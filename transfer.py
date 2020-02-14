import torch
from torchvision import transforms
import os, sys
from torchvision import models
import torchvision.datasets as datasets
import torch.optim as optim
from torch.optim import lr_scheduler
from optparse import OptionParser

import warnings

warnings.filterwarnings("ignore")


def validate_and_save(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global labels, outputs, _, best_accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = resnext(inputs)
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
    return False  # no improved accuracy

def change_learning_rate(optimizer, epoch, original_lr):
    """
    Got this from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = original_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_on_pretrained_model(train_folder_path: str, valid_folder_path: str, batch_size: int, model_path: str,
                              freeze_intermediate_layers: bool):
    global valid_loader, resnext
    train_set = datasets.ImageFolder(train_folder_path, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_set = datasets.ImageFolder(valid_folder_path, transform=transform)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    print("number of classes in trainset", len(train_set.classes))

    current_weight = resnext.state_dict()["fc.weight"]

    if freeze_intermediate_layers:
        resnext.eval()

    resnext.fc = torch.nn.Linear(in_features=current_weight.size()[1], out_features=len(train_set.classes), bias=True)
    resnext.fc.training = True
    num_epochs = 300
    criterion = torch.nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(resnext.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)

    best_accuracy = 0
    num_steps, current_steps, running_loss = 0, 0, 0
    no_improvement = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to the device specified above
    print("device is", device)
    resnext = resnext.to(device)

    for epoch in range(num_epochs):
        print("training epoch", epoch + 1)
        change_learning_rate(optimizer=optimizer, epoch=epoch, original_lr=0.1)
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = resnext(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            current_steps += 1
            num_steps += 1

            if current_steps % 100 == 0:
                print("epoch", epoch, "num_step", num_steps, "running_loss", running_loss / current_steps)
                current_steps, running_loss = 0, 0
                improved = validate_and_save(model_path=model_path)
                no_improvement = 0 if improved else no_improvement + 1
                if no_improvement >= 100 and epoch > 3:  # no improvement for a long time, and at least 3 epochs
                    print("no improvement over time--> finish")
                    sys.exit(0)
    validate_and_save(model_path=model_path)


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--train", dest="train_folder_path", help="Training data folder", metavar="FILE", default=None)
    parser.add_option("--dev", dest="valid_folder_path", help="Validation data folder", metavar="FILE", default=None)
    parser.add_option("--model", dest="model_path", help="Path to save the model", metavar="FILE", default=None)
    parser.add_option("--batch", dest="batch_size", help="Batch size", type="int", default=64)
    parser.add_option("--freeze", dest="freeze", action="store_true",
                      help="Freeze intermediate layers of the pretrained model", default=False)
    resnext = models.resnext101_32x8d(pretrained=True)
    (options, args) = parser.parse_args()

    transform = transforms.Compose([  # [1]
        transforms.Resize(256),  # [2]
        transforms.CenterCrop(224),  # [3]
        transforms.ToTensor(),  # [4]
        transforms.Normalize(  # [5]
            mean=[0.485, 0.456, 0.406],  # [6]
            std=[0.229, 0.224, 0.225]  # [7]
        )])

    model_path = os.path.abspath(sys.argv[4])
    train_on_pretrained_model(train_folder_path=options.train_folder_path, valid_folder_path=options.valid_folder_path,
                              batch_size=options.batch_size, model_path=options.model_path,
                              freeze_intermediate_layers=options.freeze)
