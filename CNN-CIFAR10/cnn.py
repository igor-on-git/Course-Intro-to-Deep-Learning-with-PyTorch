import numpy as np
import torch
import torch.nn as nn


class Net_v1(nn.Module):
    def __init__(self):
        super(Net_v1, self).__init__()
        dropout_prob = 0.2 #0.15
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16
            nn.Dropout2d(dropout_prob),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),  # output: 256 x 8 x 8
            nn.Dropout2d(dropout_prob),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(2, 2),  # output: 1024 x 4 x 4
            nn.Dropout2d(dropout_prob),

            nn.Flatten(),
            nn.Linear(1024 * 4 * 4, 2 * 1024),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(2 * 1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(512, 10))

    def forward(self, x):
        return self.network(x)

class Net_v2(nn.Module):
    def __init__(self):
        super(Net_v2, self).__init__()
        dropout_prob = 0.2 #0.15
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16
            nn.Dropout2d(dropout_prob),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8
            nn.Dropout2d(dropout_prob),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),  # output: 256 x 4 x 4
            nn.Dropout2d(dropout_prob),

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(512, 10))

    def forward(self, x):
        return self.network(x)

class Net_v3(nn.Module):
    def __init__(self):
        super(Net_v3, self).__init__()
        dropout_prob = 0.2 #0.15
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16
            nn.Dropout2d(dropout_prob),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8
            nn.Dropout2d(dropout_prob),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 256 x 4 x 4
            nn.Dropout2d(dropout_prob),

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(512, 10))

    def forward(self, x):
        return self.network(x)

def cnn_gen(lr, model_name):

    if model_name == 'model_cifar_v1':
        model = Net_v1()
    if model_name == 'model_cifar_v2':
        model = Net_v2()
    if model_name == 'model_cifar_v3':
        model = Net_v3()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    return model, criterion, optimizer

def cnn_train(model, criterion, optimizer, train_loader, valid_loader, device, n_epochs, model_name):

    train_loss = np.zeros(n_epochs)
    valid_loss = np.zeros(n_epochs)
    valid_loss_min = np.Inf  # track change in validation loss
    model.to(device)

    for epoch in range(n_epochs):

        ###################
        # train the model #
        ###################
        model.train()
        for data, target in train_loader:
            # move tensors to GPU if CUDA is available
            data, target = data.to(device), target.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss[epoch] += loss.item() * data.size(0)

        ######################
        # validate the model #
        ######################
        model.eval()
        for data, target in valid_loader:
            # move tensors to GPU if CUDA is available
            data, target = data.to(device), target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss
            valid_loss[epoch] += loss.item() * data.size(0)

        # calculate average losses
        train_loss[epoch] = train_loss[epoch] / len(train_loader.sampler)
        valid_loss[epoch] = valid_loss[epoch] / len(valid_loader.sampler)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch+1, train_loss[epoch], valid_loss[epoch]))

        if epoch > 2 and valid_loss[epoch] > valid_loss[epoch-1] and valid_loss[epoch-1] > valid_loss[epoch-2]:
            print('Validation loss is increasing consistently\nstopping training at Epoch #{}'.format([epoch+1]))
            break

        # save model if validation loss has decreased
        if valid_loss[epoch] <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss[epoch]))
            torch.save(model.state_dict(), model_name + '.pt')
            valid_loss_min = valid_loss[epoch]

    model.load_state_dict(torch.load(model_name + '.pt'))
    model.to('cpu')

    return model, train_loss[:epoch], valid_loss[:epoch]

def cnn_test(model, criterion, test_loader, batch_size, classes, device):
    # track test loss
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    model.to(device)

    model.eval()
    # iterate over test data
    for data, target in test_loader:
        # move tensors to GPU if CUDA is available
        data, target = data.to(device), target.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update test loss
        test_loss += loss.item() * data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not device == 'cuda' else np.squeeze(correct_tensor.cpu().numpy())
        # calculate test accuracy for each object class
        for i in range(batch_size):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # average test loss
    test_loss = test_loss / len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(10):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))