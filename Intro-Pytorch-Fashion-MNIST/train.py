import torch
import numpy as np
import matplotlib.pyplot as plt

def train_classifier(model, criterion, optimizer, trainloader, testloader, epochs):

    testing_running_loss = np.zeros(epochs)
    training_running_loss = np.zeros(epochs)
    for e in range(epochs):

        model.train()
        for images, labels in trainloader:

            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            training_running_loss[e] += loss.item()
        else:
            ## TODO: Implement the validation pass and print out the validation accuracy
            training_running_loss[e] /= len(trainloader.dataset)
            torch.save(model.state_dict(), 'model_epoch_' + str(e) + '.pth')
            #model.load_state_dict(torch.load('model_epoch_0.pth'))
            model.eval()
            accuracy = 0
            cnt = 0
            with torch.no_grad():
                for images, labels in testloader:
                    log_ps = model(images)
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

    plt.figure
    plt.plot(range(epochs), training_running_loss, label='Training loss')
    plt.plot(range(epochs), testing_running_loss, 'r', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return model