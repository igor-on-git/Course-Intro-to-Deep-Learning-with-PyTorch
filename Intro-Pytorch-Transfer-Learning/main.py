import matplotlib.pyplot as plt
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

from train import *

if __name__ == '__main__':

    data_dir = '~/Documents/deep-learning-v2-pytorch-master/intro-to-pytorch/assets/Cat_Dog_data/'

    train_en = 0
    epochs = 1

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                            transforms.Resize(255),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])

    # Pass transforms in here, then run the next cell to see how the transforms look
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

    model = models.densenet121(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(1024, 256)),
        ('relu', nn.ReLU()),
        ('drop', nn.Dropout(0.2)),
        ('fc2', nn.Linear(256, 2)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    if train_en:

        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        criterion = nn.NLLLoss()
        # Only train the classifier parameters, feature parameters are frozen
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

        model, training_running_loss, testing_running_loss = train_model(model, criterion, optimizer, trainloader, testloader, device, epochs)

        plt.figure
        plt.plot(range(epochs), training_running_loss, label='Training loss')
        plt.plot(range(epochs), testing_running_loss, 'r', label='Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        inputs, labels = next(iter(testloader))
        log_ps = model(inputs)
        ps = torch.exp(log_ps)
        top_prob, top_class = ps.topk(1, dim=1)
        equals = top_class.view(labels.shape) == labels
        print("Accuracy: {:.3f}% ".format(np.mean(equals.numpy()) * 100))
        1

    else:

        model.load_state_dict(torch.load('model_epoch_0.pth'))
        inputs, labels = next(iter(testloader))
        #for ii, (inputs, labels) in enumerate(testloader):
        log_ps = model(inputs)
        ps = torch.exp(log_ps)
        top_prob, top_class = ps.topk(1, dim=1)
        equals = top_class.view(labels.shape) == labels
        print("Accuracy: {:.3f}% ".format(np.mean(equals.numpy()) * 100))
        1
