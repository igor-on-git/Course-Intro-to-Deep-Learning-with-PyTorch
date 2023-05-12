import numpy as np
import torch

def train_model(model, criterion, optimizer, trainloader, testloader, device, epochs):

    model.to(device)

    testing_running_loss = np.zeros(epochs)
    training_running_loss = np.zeros(epochs)
    for e in range(epochs):

        model.train()
        for ii, (inputs, labels) in enumerate(trainloader):

            # Move input and label tensors to the GPU
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_running_loss[e] += loss.item()

        else:

            training_running_loss[e] /= len(trainloader.dataset)
            torch.save(model.state_dict(), 'model_epoch_' + str(e) + '.pth')
            # model.load_state_dict(torch.load('model_epoch_0.pth'))
            model.eval()
            accuracy = 0
            with torch.no_grad():
                for ii, (inputs, labels) in enumerate(testloader):
                    # Move input and label tensors to the GPU
                    inputs, labels = inputs.to(device), labels.to(device)
                    log_ps = model(inputs)
                    loss = criterion(log_ps, labels)
                    testing_running_loss[e] += loss.item()
                    ps = torch.exp(log_ps)
                    top_prob, top_class = ps.topk(1, dim=1)
                    equals = top_class.view(labels.shape) == labels
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
            accuracy /= testloader.__len__()
            testing_running_loss[e] /= len(testloader.dataset)
            print("Epoch: {}/{} ".format(e + 1, epochs), "Accuracy: {:.3f}% ".format(accuracy.item() * 100),
                  "Training Loss: {:.5f} ".format(training_running_loss[e]),
                  "Testing Loss: {:.5f}".format(testing_running_loss[e]))

    if device == 'cuda':
        model.cpu()

    return model, training_running_loss, testing_running_loss