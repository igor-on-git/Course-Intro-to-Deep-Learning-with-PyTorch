import torch

def train_classifier(model, criterion, optimizer, trainloader, testloader, epochs):

    train_losses, test_losses = [], []
    for e in range(epochs):
        training_running_loss = 0
        model.train()
        for images, labels in trainloader:

            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            training_running_loss += loss.item()
        else:
            ## TODO: Implement the validation pass and print out the validation accuracy
            model.eval()
            train_losses.append(training_running_loss)
            testing_running_loss = 0
            accuracy = 0
            cnt = 0
            with torch.no_grad():
                for images, labels in testloader:
                    log_ps = model(images)
                    loss = criterion(log_ps, labels)
                    testing_running_loss += loss.item()
                    ps = torch.exp(log_ps)
                    top_prob, top_class = ps.topk(1, dim=1)
                    equals = top_class.view(labels.shape) == labels
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
                    cnt += 1
            accuracy /= cnt

            print("Epoch: {}/{} ".format(e + 1, epochs), "Accuracy: {:.3f}% ".format(accuracy.item() * 100),
                  "Training Loss: {:.5f} ".format(training_running_loss / len(trainloader.dataset)),
                  "Testing Loss: {:.5f}".format(testing_running_loss / len(testloader.dataset)))

    return model