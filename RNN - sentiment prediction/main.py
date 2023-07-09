import numpy as np
from string import punctuation
from collections import Counter
import torch
from torch.utils.data import TensorDataset, DataLoader
from SentimentRNN import *

def pad_features(reviews_ints, seq_length):

    # getting the correct rows x cols shape
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)

    # for each review, I grab that review and
    for i, row in enumerate(reviews_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]

    return features

def tokenize_review(test_review):
    test_review = test_review.lower() # lowercase
    # get rid of punctuation
    test_text = ''.join([c for c in test_review if c not in punctuation])

    # splitting by spaces
    test_words = test_text.split()

    # tokens
    test_ints = []
    test_ints.append([vocab_to_int.get(word, 0) for word in test_words])

    return test_ints


def predict(net, test_review, sequence_length=200):
    net.eval()

    # tokenize review
    test_ints = tokenize_review(test_review)

    # pad tokenized sequence
    seq_length = sequence_length
    features = pad_features(test_ints, seq_length)

    # convert to tensor to pass into your model
    feature_tensor = torch.from_numpy(features)

    batch_size = feature_tensor.size(0)

    # initialize hidden state
    h = net.init_hidden(batch_size)

    if (train_on_gpu):
        feature_tensor = feature_tensor.cuda()

    # get the output from the model
    output, h = net(feature_tensor, h)

    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())
    # printing output value, before rounding
    print('Prediction value, pre-rounding: {:.6f}'.format(output.item()))

    # print custom response
    if (pred.item() == 1):
        print("Positive review detected!")
    else:
        print("Negative review detected.")

    # print custom response based on whether test_review is pos/neg
if __name__ == '__main__':
    with open('data/reviews.txt', 'r') as f:
        reviews = f.read()
    with open('data/labels.txt', 'r') as f:
        labels = f.read()

    ## Data preprocessing
    reviews = reviews.lower()  # lowercase, standardize
    all_text = ''.join([c for c in reviews if c not in punctuation])

    reviews_split = all_text.split('\n')
    all_text = ' '.join(reviews_split)
    words = all_text.split()

    ## Build a dictionary that maps words to integers
    counts = Counter(words)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
    ## use the dict to tokenize each review in reviews_split
    ## store the tokenized reviews in reviews_ints
    reviews_ints = []
    for review in reviews_split:
        reviews_ints.append([vocab_to_int[word] for word in review.split()])

    # 1=positive, 0=negative label conversion
    labels_split = labels.split('\n')
    encoded_labels = np.array([int(label == 'positive') for label in labels_split])

    # outlier review stats
    review_lens = Counter([len(x) for x in reviews_ints])
    print("Zero-length reviews: {}".format(review_lens[0]))
    print("Maximum review length: {}".format(max(review_lens)))

    print('Number of reviews before removing outliers: ', len(reviews_ints))

    ## remove any reviews/labels with zero length from the reviews_ints list.

    reviews_non_zero = np.array([ind for ind,x in enumerate(reviews_ints) if len(x)>0])
    reviews_ints = [reviews_ints[ind] for ind in reviews_non_zero]
    encoded_labels = np.array([encoded_labels[ind] for ind in reviews_non_zero])

    print('Number of reviews after removing outliers: ', len(reviews_ints))

    seq_length = 200

    features = pad_features(reviews_ints, seq_length=seq_length)

    split_frac = 0.8

    ## split data into training, validation, and test data (features and labels, x and y)
    split_ind = int(len(features)*split_frac)
    train_x, rest_x = features[:split_ind], features[split_ind:]
    train_y, rest_y = encoded_labels[:split_ind], encoded_labels[split_ind:]

    split_ind = len(rest_x)//2
    val_x, test_x = rest_x[:split_ind], rest_x[split_ind:]
    val_y, test_y = rest_y[:split_ind], rest_y[split_ind:]

    # create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    # dataloaders
    batch_size = 50

    # make sure to SHUFFLE your data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    # obtain one batch of training data
    dataiter = iter(train_loader)
    sample_x, sample_y = next(dataiter)

    print('Sample input size: ', sample_x.size())  # batch_size, seq_length
    print('Sample input: \n', sample_x)
    print()
    print('Sample label size: ', sample_y.size())  # batch_size
    print('Sample label: \n', sample_y)

    # First checking if GPU is available
    train_on_gpu = torch.cuda.is_available()

    if (train_on_gpu):
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')

    # Instantiate the model w/ hyperparams
    vocab_size = len(vocab_to_int) + 1  # +1 for the 0 padding + our word tokens
    output_size = 1
    embedding_dim = 400
    hidden_dim = 256
    n_layers = 2

    net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, train_on_gpu)

    print(net)

    # loss and optimization functions
    lr = 0.001

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    print(net)

    # training params

    epochs = 4  # 3-4 is approx where I noticed the validation loss stop decreasing

    counter = 0
    print_every = 100
    clip = 5  # gradient clipping

    # move model to GPU, if available
    if (train_on_gpu):
        net.cuda()

    net.train()
    # train for some number of epochs
    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)

        # batch loop
        for inputs, labels in train_loader:
            counter += 1

            if (train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            output, h = net(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for inputs, labels in valid_loader:

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    if (train_on_gpu):
                        inputs, labels = inputs.cuda(), labels.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output.squeeze(), labels.float())

                    val_losses.append(val_loss.item())

                net.train()
                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))

    # Get test data loss and accuracy

    test_losses = []  # track loss
    num_correct = 0

    # init hidden state
    h = net.init_hidden(batch_size)

    net.eval()
    # iterate over test data
    for inputs, labels in test_loader:

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        if (train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()

        # get predicted outputs
        output, h = net(inputs, h)

        # calculate loss
        test_loss = criterion(output.squeeze(), labels.float())
        test_losses.append(test_loss.item())

        # convert output probabilities to predicted class (0 or 1)
        pred = torch.round(output.squeeze())  # rounds to the nearest integer

        # compare predictions to true label
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)

    # -- stats! -- ##
    # avg test loss
    print("Test loss: {:.3f}".format(np.mean(test_losses)))

    # accuracy over all test data
    test_acc = num_correct / len(test_loader.dataset)
    print("Test accuracy: {:.3f}".format(test_acc))

    # negative test review
    test_review_neg = 'The worst movie I have seen; acting was terrible and I want my money back. This movie had bad acting and the dialogue was slow.'
    # positive test review
    test_review_pos = 'This movie had the best acting and the dialogue was so good. I loved it.'

    # call function
    # try negative and positive reviews!
    seq_length = 200
    predict(net, test_review_neg, seq_length)
    1