import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

from utils.dataloader import get_train_loaders, get_test_loaders
from utils.model import CustomVGG
from utils.helper import train, evaluate, predict_localize
from utils.constants import NEG_CLASS

## Parameters

batch_size = 40
target_train_accuracy = 0.98 # for early stopping of model training
test_size = 0.2 # for reduction of samples in test set
learning_rate = 0.0001
epochs = 10
class_weight = [1, 3] if NEG_CLASS == 1 else [3, 1] # Good = 1, Anomaly = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

heatmap_thres = 0.7

##  Load Training Data

data_folder = "data/"
subset_name = ["capsule", "hazelnut", "leather"] # list all objects for training
roots = []

for subset in subset_name:
    roots.append(os.path.join(data_folder, subset))

train_loader = get_train_loaders(
    roots=roots,
    batch_size=batch_size,
    random_state=42
)


# Ask user if he/she wants to load saved weights from previous training
while True:
    res = input("Do you want to load weights from previous training? (Y/N)\n")
    if res.lower() in ('y', 'n'):
        break
    else:
        print("Invalid input. Please try again.")

# Loading Saved Weights From Previous Training Results
if res.lower() == 'y': 
    model_path = f"weights/{subset_name}_model.h5"
    model = torch.load(model_path, map_location=device)

# Start Training Model
else: 
    model = CustomVGG()

    class_weight = torch.tensor(class_weight).type(torch.FloatTensor).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model = train(train_loader, model, optimizer, criterion, epochs, device, target_train_accuracy)
    torch.save(model, model_path) # save weights for next time


##  Load Testing Data

data_folder = "data/"
subset_name = ["bottle", "pill", "wood"] # list all objects for training
roots = []

for subset in subset_name:
    roots.append(os.path.join(data_folder, subset))

test_loader = get_test_loaders(
    roots=roots,
    batch_size=batch_size,
    test_size=0.9,
    random_state=42
)

# Evaluation

evaluate(model, test_loader, device)

# Visualization

predict_localize(
    model, test_loader, device, thres=heatmap_thres, n_samples=15, show_heatmap=False
)

