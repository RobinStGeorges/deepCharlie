import torch
import torchvision.transforms as transforms
import torch.nn as nn
from CharlieDataset import CharlieDataset
from dataCreate import createCsv
import torch.nn.functional as F

# Draws
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.input_channel = 3
        self.output_channel = 7
        self.linearLink = 64

        # Input channels = 3, output channels = 18
        self.conv1 = torch.nn.Conv2d(self.input_channel, self.output_channel, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 45000 input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(self.output_channel * 50 * 50, self.linearLink) # oldOutPutChanne * (size/2)²

        # 64 input features, 2 output for charlies labels
        self.fc2 = torch.nn.Linear(self.linearLink, 2)

    def forward(self, x):
        # (3, 100, 100) -> (18, 100, 100)
        x = self.conv1(x)
        # (18, 100, 100) -> (18, 50, 50)
        x = self.pool(x)
        # (18, 50, 50) -> (1, 45000)
        x = x.view(-1, self.output_channel * 50 * 50)
        # (1, 45000) -> (1, 64)
        x = F.relu(self.fc1(x))
        # (1, 64) -> (1, 2)
        x = F.relu(self.fc2(x))
        return (x)


if __name__ == "__main__":
    createCsv()
    transform = transforms.ToTensor()
    charlie_dataset_train = CharlieDataset('data.csv', 'charlies/', transform)

    loader = torch.utils.data.DataLoader(charlie_dataset_train)
    img, label = charlie_dataset_train[0]

    ## Draw
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

    for iteration in range(300):

        running_loss = 0.0
        for i, img in enumerate(loader):

            inputs, label = img
            optimizer.zero_grad() #Initialisation de l'optimiseur
            output = cnn(inputs)
            print('**')
            print('resultat : ', output.data.tolist()[0])
            print('**')
            error = errorFunction(output, label)
            error.backward()
            optimizer.step()

            if i % 100 == 0:
                print(error)
            running_loss += error.item()
            if i % 100 == 99:
                print('[%d, %5d] error: %.3f' %
                      (iteration + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    torch.save(cnn.state_dict(), "toto.dat") #Enregistrement du réseau dans un fichier
    print('poids sauvegardé')
