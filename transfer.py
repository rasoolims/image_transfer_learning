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


def train_on_pretrained_model(train_folder_path: str, valid_folder_path: str, batch_size: int, model_path: str,
                              freeze_intermediate_layers: bool, lr: float, img_size: int):
    transform = transforms.Compose([  # [1]
        transforms.Resize(img_size),  # [2]
        transforms.CenterCrop(224),  # [3]
        transforms.ToTensor(),  # [4]
        transforms.Normalize(  # [5]
            mean=[0.485, 0.456, 0.406],  # [6]
            std=[0.229, 0.224, 0.225]  # [7]
        )])
    resnext = models.resnext101_32x8d(pretrained=True)
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
    num_epochs = 100
    criterion = torch.nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(resnext.parameters(), lr=lr)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)

    best_accuracy = 0
    num_steps, current_steps, running_loss = 0, 0, 0
    no_improvement = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to the device specified above
    print("device is", device)
    resnext = resnext.to(device)

    for epoch in range(num_epochs):
        print("training epoch", epoch + 1)
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

                correct, total = 0, 0
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
                    improved = True
                else:
                    improved = False

                no_improvement = 0 if improved else no_improvement + 1
                if no_improvement >= 100 and epoch > 3:  # no improvement for a long time, and at least 3 epochs
                    print("no improvement over time--> finish")
                    sys.exit(0)
        scheduler.step(accuracy)


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--train", dest="train_folder_path", help="Training data folder", metavar="FILE", default=None)
    parser.add_option("--dev", dest="valid_folder_path", help="Validation data folder", metavar="FILE", default=None)
    parser.add_option("--model", dest="model_path", help="Path to save the model", metavar="FILE", default=None)
    parser.add_option("--batch", dest="batch_size", help="Batch size", type="int", default=64)
    parser.add_option("--lr", dest="lr", help="Learning rate", type="float", default=1e-5)
    parser.add_option("--dim", dest="img_size", help="Image dimension for transformaiton", type="int", default=128)
    parser.add_option("--freeze", dest="freeze", action="store_true",
                      help="Freeze intermediate layers of the pretrained model", default=False)
    (options, args) = parser.parse_args()

    model_path = os.path.abspath(sys.argv[4])
    train_on_pretrained_model(train_folder_path=options.train_folder_path, valid_folder_path=options.valid_folder_path,
                              batch_size=options.batch_size, model_path=options.model_path,
                              freeze_intermediate_layers=options.freeze, lr=options.lr, img_size=options.img_size)
