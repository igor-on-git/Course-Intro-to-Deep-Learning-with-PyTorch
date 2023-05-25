import matplotlib.pyplot as plt
import numpy as np
import torch


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

def show_data(train_loader, classes):
    # obtain one batch of training images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    images = images.numpy()  # convert images to numpy for display
    images = images[:20, :, :, :]

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    # display 20 images
    for idx in np.arange(images.shape[0]):
        ax = fig.add_subplot(2, 20 // 2, idx + 1, xticks=[], yticks=[])
        imshow(images[idx])
        ax.set_title(classes[labels[idx]])

def show_test_results(model, test_loader, classes, device):
    # obtain one batch of test images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images.numpy()

    # move model inputs to cuda, if GPU available
    images = images.to(device)

    # get sample outputs
    output = model(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy()) if not device == 'cuda' else np.squeeze(preds_tensor.cpu().numpy())

    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 20 // 2, idx + 1, xticks=[], yticks=[])
        imshow(images[idx] if not device == 'cuda' else images[idx].cpu())
        ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                     color=("green" if preds[idx] == labels[idx].item() else "red"))

def plot_train_valid_loss(train_loss, valid_loss):
    plt.figure(3)
    plt.plot(range(train_loss.shape[0]), train_loss, label='Train Loss')
    plt.plot(range(train_loss.shape[0]), valid_loss, label='Validation Loss')
    plt.legend()