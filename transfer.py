import torch
from torchvision import transforms
import os, sys
from PIL import Image
import pandas as pd
from torchvision import models
import torchvision.datasets as datasets
import torch.optim as optim
from torch.optim import lr_scheduler

import warnings
warnings.filterwarnings("ignore")

transform = transforms.Compose([  # [1]
    transforms.Resize(256),  # [2]
    transforms.CenterCrop(224),  # [3]
    transforms.ToTensor(),  # [4]
    transforms.Normalize(  # [5]
        mean=[0.485, 0.456, 0.406],  # [6]
        std=[0.229, 0.224, 0.225]  # [7]
    )])

batch_size = 256

resnext = models.resnext101_32x8d(pretrained=True)

train_folder_path = os.path.abspath(sys.argv[1])
train_set = datasets.ImageFolder(train_folder_path, transform = transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

valid_folder_path = os.path.abspath(sys.argv[2])
valid_set = datasets.ImageFolder(valid_folder_path, transform = transform)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)

model_path = os.path.abspath(sys.argv[3])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device is", device)
# Move model to the device specified above
resnext.to(device)


current_weight = resnext.state_dict()["fc.weight"]

# Changing the last layer of the model!
resnext.eval()
resnext.fc = torch.nn.Linear(in_features=current_weight.size()[1], out_features=len(train_set.classes), bias=True)
resnext.fc.training=True
num_epochs = 100


criterion = torch.nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = optim.SGD(resnext.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

best_accuracy  = -1
num_steps, running_loss = 0, 0
no_improvement = 0

def validate_and_save(model_path:str):
    global labels, outputs, _, best_accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for data in valid_loader:
            images, labels = data
            outputs = resnext(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100.0 * correct / total
    print("current accuracy", accuracy)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        print("saving best accuracy", best_accuracy)
        torch.save(resnext, model_path)
        return True
    return False # no improved accuracy


for epoch in range(num_epochs):
    print("training epoch", epoch+1)
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = resnext(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        num_steps+=1

        if num_steps%100==0:
            print("epoch", epoch, "num_step", num_steps, "running_loss", running_loss/num_steps)
            improved = validate_and_save(model_path=model_path)
            no_improvement = 0 if improved else no_improvement+1
            if no_improvement>=100 and epoch>3: # no improvement for a long time, and at least 3 epochs
                print("no improvement over time--> finish")
                sys.exit(0)

validate_and_save(model_path=model_path)