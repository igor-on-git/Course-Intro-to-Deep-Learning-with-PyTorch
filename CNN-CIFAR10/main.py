#import torch
from data_loaders import *
from visualizations import *
from cnn import *


if __name__ == '__main__':

    # define parameters
    use_saved_model = 0
    model_name = 'model_cifar_v1'
    # number of epochs to train the model
    n_epochs = 30
    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    batch_size = 50
    # percentage of training set to use as validation
    valid_size = 0.2
    # learning rate
    learning_rate = 0.01 * 4

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # define data loaders
    train_loader, valid_loader, test_loader, classes = data_laders(num_workers, batch_size, valid_size)

    # plot data
    show_data(train_loader, classes)
    # plt.show()

    # define model and loss function
    model, criterion, optimizer = cnn_gen(learning_rate, model_name)

    # train and validate
    if use_saved_model:
        model.load_state_dict(torch.load(model_name + '.pt'))
        model.to('cpu')
    else:
        model, train_loss, valid_loss = cnn_train(model, criterion, optimizer, train_loader, valid_loader, device, n_epochs, model_name)
        # plot training/ validation loss
        plot_train_valid_loss(train_loss, valid_loss)

    # test result
    cnn_test(model, criterion, test_loader, batch_size, classes, device)

    # show results
    show_test_results(model, test_loader, classes, device)
    # plt.show()