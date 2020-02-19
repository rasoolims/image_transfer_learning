import pickle
from optparse import OptionParser

import numpy as np
import torch
import torchvision.datasets as datasets
from torchvision import transforms


def init_net(model_path, device):
    model = torch.load(model_path)
    model.eval()
    model = model.to(device)
    return model


def pairwise_distances(x, y):
    '''
    I got this code from https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--input", dest="input_path", help="Test data folder", metavar="FILE", default=None)
    parser.add_option("--output", dest="output_path", help="Test data folder", metavar="FILE", default=None)
    parser.add_option("--model", dest="model_path", help="Path to save the model", metavar="FILE", default=None)
    parser.add_option("--batch", dest="batch_size", help="Batch size", type="int", default=32)

    (options, args) = parser.parse_args()

    print(options)

    with open(options.model_path + ".configs", "rb") as fin:
        bert_tensors_in_train, class_to_idx, img_size = pickle.load(fin)

    transform = transforms.Compose([  # [1]
        transforms.Resize(img_size),  # [2]
        transforms.CenterCrop(224),  # [3]
        transforms.ToTensor(),  # [4]
        transforms.Normalize(  # [5]
            mean=[0.485, 0.456, 0.406],  # [6]
            std=[0.229, 0.224, 0.225]  # [7]
        )])

    input_set = datasets.ImageFolder(options.input_path, transform=transform)
    input_loader = torch.utils.data.DataLoader(input_set, batch_size=options.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    top_one, top_five, all = 0, 0, 0

    top_one_dict = {i: 0 for i in range(len(class_to_idx))}
    top_five_dict = {i: 0 for i in range(len(class_to_idx))}
    all_dict = {i: 0 for i in range(len(class_to_idx))}

    with torch.no_grad():
        model = init_net(model_path=options.model_path, device=device)
        for inputs, labels in input_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            # Using negative distances in order to get argmin
            neg_distances = -pairwise_distances(outputs, bert_tensors_in_train)
            topk = neg_distances.topk(k=5, dim=1)[1].numpy()
            top1 = topk[:, 0]
            all += labels.size(0)
            labels = labels.numpy()

            for i, label in enumerate(labels):
                top5 = set(topk[i])
                first = top1[i]
                label = int(label)
                if first == label:
                    top_one_dict[label] += 1
                    top_five += 1
                if label in top5:
                    top_five_dict[label] += 1
                    top_one += 1
                all_dict[label] += 1
                all += 1
    print("top_1", round(100.0 * top_one / all, 2), "top_5", round(100.0 * top_five / all, 2))

    top1_label_accuracy = {}
    top5_label_accuracy = {}

    index_to_label_dict = {value: key for key, value in class_to_idx.items()}

    for index in all_dict.keys():
        label = index_to_label_dict[index]
        all_count = all_dict[index]
        if all_count == 0:
            top1_label_accuracy[label] = "N/A"
            top5_label_accuracy[label] = "N/A"
        else:
            top1_label_accuracy[label] = top_one_dict[index] / all_count
            top5_label_accuracy[label] = top_five_dict[index] / all_count

    print("top_5_accuracy", top5_label_accuracy)
    print("top_1_accuracy", top1_label_accuracy)
