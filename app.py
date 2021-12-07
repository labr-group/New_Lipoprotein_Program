import os
import csv
from flask import Flask, flash, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
#import matplotlib.pyplot as plt
import torch.nn.functional as F # Contains many useful loss functions and several other utilities.
import torch.optim as optim

app = Flask(__name__)

UPLOAD_FOLDER = 'Uploads'
ALLOWED_EXTENSIONS = {'csv'}

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
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

@app.route('/')
def index():
    return render_template("index.html", data="sup")


@app.route('/data', methods=['GET', 'POST'])
def data():
    if request.method == 'POST': # Check if the user posted a file.
        file = request.files['csvfile'] # Get the file that was submitted in the form.
        
        # Make the "static" directory in this project if it doesn't exist.
        if not os.path.isdir('static'):
            os.mkdir('static')
            
        filepath = os.path.join('static', file.filename) # Create the filepath showcasing the file uploaded inside of the folder.
        file.save(filepath) # Save the file to this filepath.
        
        with open(filepath) as theFile: # Reopen the file.
            r = seperateColumn(1, filepath) # Get the r column.
            r = r[55:355] # Extract the unused data.
            finalR = []
            finalR.append(r) # Essentially turn the 1D array into a 2D array.
            train = np.array(finalR) # Convert to numpy array.
            train = torch.from_numpy(train).cuda() # Turn it into a pytorch tensor to be used.
            
            # Load the model.
            model = nn.Linear(300, 1).cuda()
            model.load_state_dict(torch.load("model2.mdx"))
            
            list(model.parameters()) # Returns a list of the model parameters.
            preds = model(train) # Generate the predicted concentration of the file.
            value = preds.item() # Get the prediction from the preds tensor.
            
        return render_template("index.html", data=value) # Redirect the user to the same index.html file but with the concentration.

if __name__ == "__main__":
    app.run(debug=True)