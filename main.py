import csv;
import random;
import numpy as np;
import glob;
import constants

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
#import matplotlib.pyplot as plt
import torch.nn.functional as F # Contains many useful loss functions and several other utilities.
import torch.optim as optim


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

torch.set_default_dtype(torch.float64)

# Read the csv file passed in and store the specified row.
def seperateColumn (column, document):
    currList = []
    #currList.append(prefix)
    with open(document, 'r') as csvFile:
        reader = csv.reader(csvFile)
        #skip header row
        next(reader)
        #i=0
        for row in reader:
            #i = i+1
            #if(i>360):
            #    break
            #print(row)
            if column == 0:
                currList.append(float(row[0]))
            elif column == 1:
                currList.append(float(row[1]))
            else:
                currList.append(float(row[2]))
    csvFile.close()
    return currList

#all files in one directory
#add column to include target (label)
def rawDataToDs(path):
    all_files = glob.glob(path + "/*.csv")
    trainDs = []
    validDs = []
    trainTargets = []
    validTargets = []
    
    trainDs2 = None
    validDs2 = None
    trainTargets2 = None
    validTargets2 = None
    #random number to get roughly 20% of data to validate model
    rand = random.randint(0,100)
    for filename in all_files:
        #csv
        trainS = seperateColumn(2, filename)
        trainR = seperateColumn(1, filename)
        trainQ = seperateColumn(0, filename)
        temp = filename.split("\\")
        batch = (temp[1].split("_"))[2]
        batchPer = float(batch.split("P")[0])
        
        # batchPer = (float(batchPer[:-1])) # Old code.
        
        # Arrays to datasets.
        if(rand <= 20):
            for i in range(len(trainQ)):
                if trainS[i] < 0.01:
                    temp = [trainQ[i], trainR[i]]
                    validDs.append(temp)
                    validTargets.append(batchPer)
        else:
            for i in range(len(trainQ)):
                if trainS[i] < 0.01:
                    temp = [trainQ[i], trainR[i]]
                    trainDs.append(temp)
                    trainTargets.append(batchPer)
        
        tempV = []
        tempT = []
        
        if(rand <= 20):
            for i in range(len(trainQ)):
                if trainS[i] < 0.01:
                    temp = [trainQ[i], trainR[i]]
                    tempV.append(temp)
                    tempV.append(batchPer)
                    
            tempV = torch.tensor((tempV, batchPer))
            if validDs2 == None:
                validDs2 = tempV
            else:
                validDs2 = torch.cat(validDs2, tempV)
        else:
            for i in range(len(trainQ)):
                if trainS[i] < 0.01:
                    temp = [trainQ[i], trainR[i]]
                    tempT.append(temp)
                    tempT.append(batchPer)
                    
            tempT = torch.tensor((tempT, batchPer))
            if trainDs2 == None:
                trainDs2 = tempT
            else:
                trainDs2 = torch.cat(trainDs2, tempT)
        
        rand = random.randint(0,100)
    
    print(validDs2)
    print(trainDs2)
    return (trainDs, validDs, trainTargets, validTargets)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(constants.NUMROWS, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 1) # Don't want to run F.relu here.
    
    def forward(self, x):
        x = F.relu(self.fc1(x)) # F.relu (rectified linear) is an activation function.
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.softmax(x, dim=1) # Probability distribution function.

net = Net()
print(net)

optimizer = optim.Adam(net.parameters())
"""
dataQ = seperateColumn(0, "TestData/Concen_28_55P.csv")
dataR = seperateColumn(1, "TestData/Concen_28_55P.csv")
dataS = seperateColumn(2, "TestData/Concen_28_55P.csv")

data = []
for i in range(len(dataQ)):
    temp = [dataQ[i], dataR[i], dataS[i]]
    data.append(temp)

target = [55]
for i in range (len(dataQ) - 1):
    target.append(55)

data = np.array(data)

target = np.array(target)
data = torch.from_numpy(data)
target = torch.from_numpy(target)
target = target.view(-1, 1).double()
"""

allData = rawDataToDs("TrainData")
data = allData[0]
validData = allData[1]
target = allData[2]
validTargets = allData[3]

data = np.array(data)
target = np.array(target)
data = torch.from_numpy(data)
target = torch.from_numpy(target)
target = target.view(-1, 1).double()

# This slows the program down and does no considerable bonus experimentally.
# data.requires_grad_()
# target.requires_grad_()

# Define dataset.
train_ds = TensorDataset(data, target)
train_ds[0:50] # Picks first three rows of input data and output data.

# Define data loader
batch_size = 50
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# A data loader is typically used in a for-in loop, like this:
"""
for xb, yb in train_dl:
    print("batch:")
    print(xb)
    print(yb)
    break
"""

# Define model
model = nn.Linear(2, 1).double() # Can be used instead of initializing the weights & biases manually. This does it automatically.

# Parameters
list(model.parameters()) # Returns a list containing all the weights and bias matrices present in the model.

# Generate predictions
preds = model(data)

# Define loss function
loss_fn = F.mse_loss

# Compute the loss for the current predictions of our model.
loss = loss_fn(model(data), target)

# Define optimizer (used instead of manually manipulating the model's weights & biases using gradients).
# Note: SGD is short for "stochastic gradient descent". "Stochastic" indicates that samples are selected
# in batches (often with random shuffling) instead of as a single group.
opt = torch.optim.SGD(model.parameters(), lr=1e-5)

# Utility function to train the model
def fit(num_epochs, model, loss_fn, opt, train_dl):
    
    # Repeat for given number of epochs
    for epoch in range(num_epochs):
        
        # Train with batches of data
        for xb,yb in train_dl:
            
            # 1. Generate predictions
            pred = model(xb)
            
            # 2. Calculate loss
            loss = loss_fn(pred, yb)
            
            # 3. Compute gradients
            loss.backward()
            
            # 4. Update parameters using gradients
            opt.step()
            
            # 5. Reset the gradients to zero
            opt.zero_grad()
        
        # Print the progress
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

fit(30000, model, loss_fn, opt, train_dl)
preds = model(data)
print(preds)

num = torch.mean(preds)
print(num)

dataQ = seperateColumn(0, "TrainData/Concen_10_35P.csv")
dataR = seperateColumn(1, "TrainData/Concen_10_35P.csv")
dataS = seperateColumn(2, "TrainData/Concen_10_35P.csv")

data2 = []
for i in range(len(dataQ)):
    temp = [dataQ[i], dataR[i], dataS[i]]
    data2.append(temp)

data2 = np.array(data2)
data2 = torch.from_numpy(data2)

preds2 = model(data2)
num2 = torch.mean(preds2)
print(num2)