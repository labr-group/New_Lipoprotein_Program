# Lipoproteins Machine Learning Program
# Author: Justin Bornais

# Import general libraries.
import csv
from datetime import datetime;
import numpy as np;
import glob;
import os;

# Import ML libraries from PyTorch.
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F # Contains many useful loss functions and several other utilities.

# Define useful constants. DO NOT EDIT unless you know exactly what you are doing.
NUMROWS = 399
NUMFILES = 27
CREATEMODEL = 1 # 0 if you want to open an existing model MODELNAME[i], 1 if you want to create the MODELNAME model.
FILESTOTRAIN = [[3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,27],
                [1,3,4,5,6,8,10,12,14,16,18,20,22,24,26],
                [2,3,4,5,7,9,11,13,15,17,19,21,23,25,27],
                [1,2,3,5,7,9,11,13,15,17,19,21,23,25,27],
                [1,2,4,5,6,8,10,12,14,16,18,20,22,24,26],
                [1,2,3,4,5,7,9,11,13,15,17,19,21,27],
                [1,2,3,4,5,6,8,10,12,14,16,18,20,22],
                [1,3,4,6,8,9,10,12,13,14,16,17,18,20,21,24,25,27],
                [2,3,4,6,8,9,10,12,13,14,16,17,18,20,21,24,25,27],
                [3,4,5,6,8,9,10,12,13,14,16,17,18,20,21,24,25,27],
                [3,7,9,11,13,15,17,19,21,23,27],
                [1,3,7,9,11,13,15,17,19,21,23,27],
                [2,3,7,9,11,13,15,17,19,21,23,27],
                [3,5,7,9,11,13,15,17,19,21,23,27],
                [1,3,5,7,9,11,13,15,17,19,21,23,27],
                [1,2,3,7,9,11,13,15,17,19,21,23,27],
                [1,2,3,5,7,9,11,13,15,17,19,21,23,27],
                [4,6,8,10,12,14,16,18,20,22],
                [1,4,6,8,10,12,14,16,18,20,22],
                [2,4,6,8,10,12,14,16,18,20,22],
                [4,5,6,8,10,12,14,16,18,20,22],
                [1,4,5,6,8,10,12,14,16,18,20,22],
                [1,2,4,6,8,10,12,14,16,18,20,22],
                [1,2,4,5,6,8,10,12,14,16,18,20,22]]
NUMEPOCHS = [10_000, 100_000, 1_000_000, 5_000_000, 10_000_000, 25_000_000, 50_000_000, 100_000_000]
#NUMEPOCHS = [100, 1000, 2000]
EPOCHSPERUPDATE = 100
LR = 3.5e-1

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

torch.set_default_dtype(torch.float64)

# Read the csv file passed in and store the specified row.
def separateColumn(column, document):
    currList = []
    with open(document, 'r') as csvFile:
        reader = csv.reader(csvFile)
        next(reader) #skip header row
        for row in reader:
            if column == 0: currList.append(float(row[0]))
            elif column == 1: currList.append(float(row[1]))
            else: currList.append(float(row[2]))
    csvFile.close()
    return currList

#all files in one directory
#add column to include target (label)
def rawDataToDs(files_to_train):
    all_files = []
    for f in glob.glob("AllData/*.csv"):
        for i in files_to_train:
            if f'Concen_{i}_' in f:
                all_files.append(f)
    
    trainDs = []
    trainTs = []
    
    for filename in all_files:
        trainR = separateColumn(1, filename)
        trainQ = separateColumn(0, filename)
        temp = filename.split("\\")
        batch = (temp[1].split("_"))[2]
        batchPer = float(batch.split("P")[0])
        
        temp = []
        
        for i in range(55, 355):
            temp.append(trainR[i])
        
        # Add data to datasets.
        trainDs.append(temp)
        trainTs.append(batchPer)
    
    return (trainDs, trainTs)

# Utility function to train the model.
def fit(current_epochs, num_epochs, model, loss_fn, opt, train_dl):
    
    # Repeat for given number of epochs.
    for epoch in range(current_epochs, num_epochs):
        
        # Train with batches of data.
        for xb,yb in train_dl:
            pred = model(xb)         # 1. Generate predictions.
            loss = loss_fn(pred, yb) # 2. Calculate loss.
            loss.backward()          # 3. Compute gradients.
            opt.step()               # 4. Update parameters using gradients.
            opt.zero_grad()          # 5. Reset the gradients to zero.
        
        # Print the progress.
        if (epoch+1) % EPOCHSPERUPDATE == 0:
            print('Epoch [{}/{}], Loss: {:.12f}'.format(epoch+1, num_epochs, loss.item()))


def log_status(file, model, num_epochs):
    file.write(f"All File Concentration Predictions at {num_epochs} epochs (sorted by accuracy in descending order):\n\n")
    
    all_files = glob.glob("AllData" + "/*.csv")
    compilation = []
    
    for i in range(len(all_files)):
        
        temp = all_files[i].split("\\")
        batch = (temp[1].split("_"))[2]
        batchPer = float(batch.split("P")[0])
        
        r = separateColumn(1, all_files[i]) # Get the r column.
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

    compilation.sort(key=lambda x: abs(x[3])) # Sort by absolute difference between prediction and actual concentration.
    
    # Print the results for each file.
    for i in range(len(compilation)):
        file.write(f"File: {compilation[i][0]}:")
        
        # Formatting.
        temp = len(compilation[i][0]) + 1
        while temp < 36:
            file.write(" ")
            temp += 1
            
        file.write(f"Predicted concentration: {compilation[i][1]:.6f}%\tActual concentration: {compilation[i][2]}%")
        
        # Formatting.
        temp = len(str(compilation[i][2])) + 1
        while temp < 8:
            file.write(" ")
            temp += 1
        
        file.write(f"Difference: {compilation[i][3]}%\n")
    
    
    # Print average error for top 5 files.
    if(len(compilation) > 5):
        average_error = 0
        for i in range(5): average_error += abs(compilation[i][3])
        average_error /= 5
        file.write(f"Average error in predictions for top 5 files:\t{average_error:.6f}%\n")
    
    # Print average error for top 10 files.
    if(len(compilation) > 10):
        average_error = 0
        for i in range(10): average_error += abs(compilation[i][3])
        average_error /= 10
        file.write(f"Average error in predictions for top 10 files:\t{average_error:.6f}%\n")
    
    # Print average error for all files.
    average_error = 0
    for i in range(len(compilation)): average_error += abs(compilation[i][3])
    average_error /= len(compilation)
    file.write(f"Average error in predictions for all files:\t\t{average_error:.6f}%\n")

def train_data(files_to_train, path, num):
    
    # Create model name.
    num_models = len(glob.glob("models/*.mdx"))
    model_name = f"models/model{num_models + 1}.mdx" # Could also use len(os.listdir("models/"))
    current_epochs = 0 # Will be used to keep track of the number of epochs.
    
    # Initialize log file.
    filename = path + "\\" + f"Test {num}.log"
    output = open(filename, "w")
    output.write(f"Model name: {model_name}\n")
    output.write(f"Files Trained On: {files_to_train}\n\n\n")
    
    allData = rawDataToDs(files_to_train)
    train = allData[0]
    ttargets = allData[1]

    train = np.array(train)
    ttargets = np.array(ttargets)
    train = torch.from_numpy(train).cuda()
    ttargets = torch.from_numpy(ttargets).cuda()
    ttargets = ttargets.view(-1, 1).double()

    # Define dataset.
    train_ds = TensorDataset(train, ttargets)
    train_ds[0:100] # Picks first 50 rows of input data and output data.

    # Define data loader
    batch_size = 50
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    
    # Define model
    if CREATEMODEL == 1:
        model = nn.Linear(300, 1).cuda() # Can be used instead of initializing the weights & biases manually. This does it automatically.
    else:
        model = nn.Linear(300, 1).cuda()
        model.load_state_dict(torch.load(model_name))
    
    list(model.parameters()) # Returns a list containing all the weights and bias matrices present in the model.
    
    # Generate predictions
    loss_fn = F.mse_loss
    
    # Define optimizer (used instead of manually manipulating the model's weights & biases using gradients).
    # Note: SGD is short for "stochastic gradient descent". "Stochastic" indicates that samples are selected
    # in batches (often with random shuffling) instead of as a single group.
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    
    for i in range(len(NUMEPOCHS)):
        fit(current_epochs, NUMEPOCHS[i], model, loss_fn, opt, train_dl) # Fit the data.
        current_epochs = NUMEPOCHS[i]
        log_status(output, model, NUMEPOCHS[i])
        
        if i + 1 != len(NUMEPOCHS): output.write("\n\n")
        #preds = model(train)
    
    output.close()
    torch.save(model.state_dict(), model_name) # Save the model.


now = datetime.now() # Get the current date.
folder = "LogFiles\\" + now.strftime("Training - %Y-%m-%d_%H-%M-%S") # Name of the folder for the log files.
os.mkdir(folder) # Make the new folder for the log files.

for i in range(len(FILESTOTRAIN)):
    train_data(FILESTOTRAIN[i], folder, i) # Train on each test.