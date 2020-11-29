import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from CharlieDataset import CharlieDataset
from dataCreate import createCsv
from torch.autograd import Variable
import torch.nn.functional as F

num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Input channels = 3, output channels = 18
        self.conv1 = torch.nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 4608 input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(18 * 50 * 50, 64)

        # 64 input features, 2 output features for our 2 defined classes
        self.fc2 = torch.nn.Linear(64, 2)

    def forward(self, x):
        # Computes the activation of the first convolution
        # Size changes from (3, 32, 32) to (18, 32, 32)
        x = self.conv1(x)

        # Size changes from (18, 32, 32) to (18, 16, 16)
        x = self.pool(x)

        # Reshape data to input to the input layer of the neural net
        # Size changes from (18, 16, 16) to (1, 4608)
        # Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 18 * 50 * 50)

        # Computes the activation of the first fully connected layer
        # Size changes from (1, 4608) to (1, 64)
        x = F.relu(self.fc1(x))

        # Computes the second fully connected layer (activation applied later)
        # Size changes from (1, 64) to (1, 10)
        x = self.fc2(x)
        return (x)


if __name__ == "__main__":
    createCsv()
    transform = transforms.ToTensor()
    charlie_dataset = CharlieDataset('data.csv', 'charlies/', transform)

    loader = torch.utils.data.DataLoader(charlie_dataset)

    #print(charlie_dataset[0])
    #print(len(charlie_dataset))
    img, label = charlie_dataset[0]
    #print(img.size(), label)

    #print(type(img))
    #print(type(label))

    # npimg = img.numpy()
    # npimg = np.transpose(npimg, (1, 2, 0))
    # plt.imshow(npimg)
    # plt.show()

    # print("img", img)
    # print("label", label)

    cnn = CNN()

    optimizer = torch.optim.SGD(cnn.parameters(), lr=1e-4)
    errorFunction = nn.CrossEntropyLoss()


    for iteration in range(10000):
        for i, img in enumerate(loader):
            inputs, label = img

            optimizer.zero_grad() #Initialisation de l'optimiseur
            output = cnn(inputs)
            print('label : ', label)
            print('output : ', output)
            error = errorFunction(output, label)
            error.backward()
            optimizer.step()

            if i % 100 == 0:
                print(error)

    torch.save(cnn.state_dict(), "toto.dat") #Enregistrement du réseau dans un fichier
"""
    #Pour ouvrir dans un autre script
    cnn = CNN()
    cnn.load_state_dict(torch.load("toto.dat"))
    cnn.eval() #toujours commencer par ça pour construire le réseau
    """
