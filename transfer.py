import pickle
import sys
import warnings
from optparse import OptionParser

import network
import numpy as np
import torch
import torch.optim as optim
import torchvision.datasets as datasets
from dataset import TripletDataSet
from loss import TripletLoss
from torch.optim import lr_scheduler
from torchvision import models
from torchvision import transforms

warnings.filterwarnings("ignore")


def init_net(embed_dim: int, options):
    model = models.densenet161(pretrained=True)
    model.__class__ = network.DenseNetWithDropout
    model.dropout = options.dropout
    if options.freeze_intermediate_layers:
        model.eval()
    in_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(in_features=in_features, out_features=embed_dim, bias=False)
    model.classifier.training = True
    return model


def train_on_pretrained_model(options):
    transform = transforms.Compose([  # [1]
        transforms.Resize(options.img_size),  # [2]
        transforms.CenterCrop(224),  # [3]
        transforms.ToTensor(),  # [4]
        transforms.Normalize(  # [5]
            mean=[0.485, 0.456, 0.406],  # [6]
            std=[0.229, 0.224, 0.225]  # [7]
        )])
    with open(options.bert_path, "rb") as fin:
        bert_tensors = pickle.load(fin)
    print("loaded bert tensors of size", len(bert_tensors))

    train_set = datasets.ImageFolder(options.train_folder_path, transform=transform)
    embed_dim = bert_tensors[0].shape[0]

    bert_tensors_in_train = torch.tensor(np.array([bert_tensors[int(label)] for label in train_set.classes]))
    # Making sure that we do not change the BERT values.
    bert_tensors_in_train.requires_grad = False

    train_triplet_set = TripletDataSet(image_folder=train_set, bert_tensors=bert_tensors_in_train,
                                       num_neg_samples=options.neg_samples, is_train_data=True)
    train_loader = torch.utils.data.DataLoader(train_triplet_set, batch_size=options.batch_size, shuffle=True)
    valid_set = datasets.ImageFolder(options.valid_folder_path, transform=transform)
    valid_triplet_set = TripletDataSet(image_folder=valid_set, bert_tensors=bert_tensors_in_train,
                                       num_neg_samples=options.neg_samples, is_train_data=False)
    valid_loader = torch.utils.data.DataLoader(valid_triplet_set, batch_size=options.batch_size, shuffle=False)

    print("number of classes in trainset", len(train_set.classes))
    print("saving the BERT tensors")
    with open(options.model_path + ".configs", "wb") as fout:
        pickle.dump((bert_tensors_in_train, train_set.class_to_idx, options.img_size), fout)

    model = init_net(embed_dim, options)

    num_epochs = 100
    criterion = TripletLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(model.parameters(), lr=options.lr)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)

    best_loss = float("inf")
    num_steps, current_steps, running_loss = 0, 0, 0
    no_improvement = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to the device specified above
    print("device is", device)
    model = model.to(device)

    for epoch in range(num_epochs):
        print("training epoch", epoch + 1)
        for inputs, labels in train_loader:
            anchor = inputs[0].to(device)
            positive = inputs[1].to(device)
            negative = inputs[2].to(device)

            optimizer.zero_grad()
            anchor_outputs = model(anchor)

            loss = criterion(anchor=anchor_outputs, positive=positive, negative=negative)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            current_steps += 1
            num_steps += 1

            if current_steps % 100 == 0:
                print("epoch", epoch, "num_step", num_steps, "running_loss", running_loss / current_steps)
                current_steps, running_loss = 0, 0

                loss_value, total = 0, 0
                with torch.no_grad():
                    model.training = False

                    for inputs, labels in valid_loader:
                        anchor = inputs[0].to(device)
                        positive = inputs[1].to(device)
                        negative = inputs[2].to(device)

                        anchor_outputs = model(anchor)
                        loss = criterion(anchor=anchor_outputs, positive=positive, negative=negative)

                        total += anchor_outputs.size(0)
                        loss_value += loss.item()
                current_loss = 100.0 * loss_value / total
                print("current dev loss", current_loss)
                if current_loss < best_loss:
                    best_loss = current_loss
                    print("saving best dev loss", best_loss)
                    torch.save(model, options.model_path)
                    improved = True
                else:
                    improved = False

                no_improvement = 0 if improved else no_improvement + 1
                if no_improvement >= 100 and epoch > 3:  # no improvement for a long time, and at least 3 epochs
                    print("no improvement over time--> finish")
                    sys.exit(0)

            model.training = True

        scheduler.step(-current_loss)


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--train", dest="train_folder_path", help="Training data folder", metavar="FILE", default=None)
    parser.add_option("--dev", dest="valid_folder_path", help="Validation data folder", metavar="FILE", default=None)
    parser.add_option("--bert", dest="bert_path", help="File that contains bert vectors", metavar="FILE", default=None)
    parser.add_option("--model", dest="model_path", help="Path to save the model", metavar="FILE", default=None)
    parser.add_option("--batch", dest="batch_size", help="Batch size", type="int", default=32)
    parser.add_option("--sample", dest="neg_samples", help="Number of negative samples for triplet loss", type="int",
                      default=30)
    parser.add_option("--lr", dest="lr", help="Learning rate", type="float", default=1e-5)
    parser.add_option("--dropout", dest="dropout", help="Dropout", type="float", default=0.5)
    parser.add_option("--dim", dest="img_size", help="Image dimension for transformation", type="int", default=128)
    parser.add_option("--freeze", dest="freeze_intermediate_layers", action="store_true",
                      help="Freeze intermediate layers of the pretrained model", default=False)
    (options, args) = parser.parse_args()

    print(options)
    train_on_pretrained_model(options=options)
