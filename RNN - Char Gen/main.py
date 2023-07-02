from service import *
from CharRNN import *

if __name__ == '__main__':

    # open text file and read in data as `text`
    with open('anna.txt', 'r') as f:
        text = f.read()

    print(text[:200])

    # encode the text and map each character to an integer and vice versa

    # we create two dictionaries:
    # 1. int2char, which maps integers to characters
    # 2. char2int, which maps characters to unique integers
    chars = tuple(set(text))
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}

    # encode the text
    encoded = np.array([char2int[ch] for ch in text])

    # print(encoded[:100])

    batches = get_batches(encoded, 8, 50)
    x, y = next(batches)

    # printing out the first 10 items in a sequence
    # print('x\n', x[:10, :10])
    # print('\ny\n', y[:10, :10])

    # check if GPU is available
    train_on_gpu = torch.cuda.is_available()
    if (train_on_gpu):
        print('Training on GPU!')
    else:
        print('No GPU available, training on CPU; consider making n_epochs very small.')

    n_hidden = 512
    n_layers = 2

    net = CharRNN(chars, train_on_gpu, n_hidden, n_layers)
    # print(net)

    batch_size = 128
    seq_length = 100
    n_epochs = 50  # start small if you are just testing initial behavior

    # train the model
    train(net, encoded, train_on_gpu, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.001, print_every=10)

    net = net.load_net_state()

    # print generated text
    print(sample(net, 2000, train_on_gpu, top_k=5, prime="And Levin said"))

    1
