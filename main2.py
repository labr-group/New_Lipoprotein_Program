import csv;
import random;
import numpy as np;
import glob;
from datetime import datetime
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
    # all_files = glob.glob(path + "/*.csv")
    all_files = []
    for f in glob.glob("AllData/*.csv"):
        for i in constants.FILESTOTRAIN:
            if f'Concen_{i}_' in f:
                all_files.append(f)
    
    trainDs = []
    validDs = []
    trainTs = []
    validTs = []
    
    #random number to get roughly 20% of data to validate model
    rand = random.randint(0,100)
    for filename in all_files:
        trainR = seperateColumn(1, filename)
        trainQ = seperateColumn(0, filename)
        temp = filename.split("\\")
        batch = (temp[1].split("_"))[2]
        batchPer = float(batch.split("P")[0])
        
        temp = []
        
        for i in range(55, 355):
            temp.append(trainR[i])
        
        if rand <= -1:
            validDs.append(temp)
            validTs.append(batchPer)
            """
            if validDs == None:
                validDs = temp
            else:
                validDs = torch.cat(validDs, temp)
            """
        else:
            trainDs.append(temp)
            trainTs.append(batchPer)
            """
            if trainDs == None:
                trainDs = temp
            else:
                trainDs = torch.cat(trainDs, temp)
            """
        rand = random.randint(0,100)
    
    return (trainDs, trainTs)
    #return (trainDs, trainTs, validDs, validTs)

allData = rawDataToDs(constants.FOLDERFORTRAINING)
train = allData[0]
ttargets = allData[1]
# valid = allData[2]
# vtargets = allData[3]

train = np.array(train)
ttargets = np.array(ttargets)
train = torch.from_numpy(train).cuda()
ttargets = torch.from_numpy(ttargets).cuda()
ttargets = ttargets.view(-1, 1).double()

# valid = np.array(valid)
# vtargets = np.array(vtargets)
# valid = torch.from_numpy(valid).cuda()
# vtargets = torch.from_numpy(vtargets).cuda()
# vtargets = vtargets.view(-1, 1).double()

# Define dataset.
train_ds = TensorDataset(train, ttargets)
train_ds[0:100] # Picks first 50 rows of input data and output data.

# Define data loader
batch_size = 50
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# Define model
if constants.CREATEMODEL == 1:
    model = nn.Linear(300, 1).cuda() # Can be used instead of initializing the weights & biases manually. This does it automatically.
else:
    model = nn.Linear(300, 1).cuda()
    model.load_state_dict(torch.load(constants.MODELNAME))
#model.cuda()
#print(model2.weight)
#print(model2.bias)

list(model.parameters()) # Returns a list containing all the weights and bias matrices present in the model.

# Generate predictions
preds = model(train)
#print(preds)

loss_fn = F.mse_loss

# Compute the loss for the current predictions of our model.
loss = loss_fn(model(train), ttargets)
#print(loss)

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(train.device)
print(ttargets.device)
print(next(model.parameters()).is_cuda)
print(len(constants.FILESTOTRAIN))

# Define optimizer (used instead of manually manipulating the model's weights & biases using gradients).
# Note: SGD is short for "stochastic gradient descent". "Stochastic" indicates that samples are selected
# in batches (often with random shuffling) instead of as a single group.
opt = torch.optim.SGD(model.parameters(), lr=constants.LR)

# Utility function to train the model.
def fit(num_epochs, model, loss_fn, opt, train_dl):
    
    # Repeat for given number of epochs.
    for epoch in range(num_epochs):
        
        # Train with batches of data.
        for xb,yb in train_dl:
            
            # 1. Generate predictions.
            pred = model(xb)
            
            # 2. Calculate loss.
            loss = loss_fn(pred, yb)
            
            # 3. Compute gradients.
            loss.backward()
            
            # 4. Update parameters using gradients.
            opt.step()
                
            # 5. Reset the gradients to zero.
            opt.zero_grad()
        
        # Print the progress
        if (epoch+1) % constants.EPOCHSPERUPDATE == 0:
            print('Epoch [{}/{}], Loss: {:.12f}'.format(epoch+1, num_epochs, loss.item()))

fit(constants.NUMEPOCHS, model, loss_fn, opt, train_dl)
preds = model(train)
# print(ttargets)
# print(preds)

torch.save(model.state_dict(), constants.MODELNAME)

filepath = "AllData\\Concen_4_5P.csv"
temp = filepath.split("\\")
batch = (temp[1].split("_"))[2]
batchPer = batch.split("P")[0]

# Generate output filename based on current date and time.
now = datetime.now()
filename = "LogFiles\\" + now.strftime("Training_-_%Y-%m-%d_%H-%M-%S.log")
print(filename)
output = open(filename, "w")
output.write(f"Number of epochs: {constants.NUMEPOCHS}\n")
output.write(f"Model name: {constants.MODELNAME}\n")

# Print trained files.
trained_files = glob.glob(constants.FOLDERFORTRAINING + "/*.csv")
print("Files used for training:")
output.write("Files used for training:\n")
for filename in trained_files:
    print(filename.split("\\")[1])
    output.write(filename.split("\\")[1] + "\n")

all_files = glob.glob("AllData" + "/*.csv")
compilation = []

print("\n\nAll File Concentration Predictions:\n")
output.write("\n\nAll File Concentration Predictions:")
for i in range(len(all_files)):
    
    temp = all_files[i].split("\\")
    batch = (temp[1].split("_"))[2]
    batchPer = float(batch.split("P")[0])
    
    r = seperateColumn(1, all_files[i]) # Get the r column.
    r = r[55:355] # Extract the unused data.
    finalR = []
    finalR.append(r) # Essentially turn the 1D array into a 2D array.
    train = np.array(finalR) # Convert to numpy array.
    train = torch.from_numpy(train).cuda() # Turn it into a pytorch tensor to be used.
    
    list(model.parameters()) # Returns a list of the model parameters.
    preds = model(train) # Generate the predicted concentration of the file.
    value = preds.item() # Get the prediction from the preds tensor.
    
    temp = [all_files[i], round(value, 6), batchPer, round(batchPer - value, 6)]
    compilation.append(temp)

compilation.sort(key=lambda x: abs(x[3]))
    
for i in range(len(compilation)):
    print(f"File: {compilation[i][0]}:\nPredicted concentration: {compilation[i][1]}%\tActual concentration: {compilation[i][2]}%\tDifference: {compilation[i][3]}%\n")
    output.write(f"\n\nFile: {compilation[i][0]}:\nPredicted concentration: {compilation[i][1]}%\tActual concentration: {compilation[i][2]}%\tDifference: {compilation[i][3]}%")

output.close()