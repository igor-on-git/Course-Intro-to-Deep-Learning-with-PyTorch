from torch import nn, optim
import torch
import helper

from FashionMNIST import *
from Classifier import *
from train import train_classifier

if __name__ == '__main__':

    trainloader, testloader = FashionMNIST()

    model = Classifier()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    epochs = 5

    model = train_classifier(model, criterion, optimizer, trainloader, testloader, epochs)

    # Test out your network!
    model.eval()

    dataiter = iter(testloader)
    images, labels = next(dataiter)
    img = images[0]
    # Convert 2D image to 1D vector
    img = img.view(1, 784)

    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model.forward(img)

    ps = torch.exp(output)