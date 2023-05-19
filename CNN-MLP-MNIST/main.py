from visualizations import *
from mlp import *

if __name__ == '__main__':

    # number of epochs to train the model
    n_epochs = 20  # suggest training between 20-50 epochs

    # number of subprocesses to use for data loading
    num_workers = 0

    # how many samples per batch to load
    batch_size = 500

    # percentage of training set to use as validation
    valid_size = 0.2

    # learning rate
    mlp_learning_rate = 0.01 * 4

    # device used for training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # data loaders
    train_loader, valid_loader, test_loader = mlp_MNIST_load(valid_size, num_workers, batch_size)

    # plot the images in the batch, along with the corresponding labels
    plot_images_labels(train_loader)

    # plot image with its intensity values
    plot_image_pixel_values(train_loader)

    # define model criterion and optimizer
    mlp_model, mlp_optimizer, mlp_criterion = mlp_gen(mlp_learning_rate)

    # train mlp
    mlp_model, train_loss, valid_loss = mlp_train(
        n_epochs, mlp_model, train_loader, valid_loader, mlp_optimizer, mlp_criterion, device)

    # plot training/ validation loss
    plot_train_valid_loss(n_epochs, train_loss, valid_loss)

    # test accuracy
    mlp_test(mlp_model, mlp_criterion, test_loader, device, batch_size)

    # demonstrate performance of trained model
    plot_model_results(mlp_model, test_loader)

    # show all plots
    plt.show()
