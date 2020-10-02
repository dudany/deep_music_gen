import os
import pandas as pd
import fire as fire
import tensorflow as tf
from scipy.io.wavfile import read
from tqdm import tqdm
from collections import namedtuple
from config import checkpoint_prefix, rnn_units, embedding_dim, learning_rate, data_path, seq_length, \
    training_iterations
from model import build_model
from utils import compute_loss, PeriodicPlotter, get_batch
from config import batch_size


# @tf.function
def train_step(x, y, model, optimizer_obj):
    # Use tf.GradientTape()
    print('train step')
    with tf.GradientTape() as tape:
        y_hat = model(x)
        loss = compute_loss(y, y_hat)
    # Now, compute the gradients
    grads = tape.gradient(loss, model.trainable_variables)

    # Apply the gradients to the optimizer so it can update the model accordingly
    optimizer_obj.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def train_defined_model_per_vec(model, optimizer_obj, vectorized_data, params):
    # model params definition
    history = []
    plotter = PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')
    seq_length = params['seq_length']
    batch_size = params['batch_size']
    training_iterations = params['training_iterations']

    if hasattr(tqdm, '_instances'): tqdm._instances.clear()  # clear if it exists

    for iter in tqdm(range(training_iterations)):
        # Grab a batch and propagate it through the network
        x_batch, y_batch = get_batch(vectorized_data, seq_length, batch_size)
        loss = train_step(x_batch, y_batch, model, optimizer_obj)

        # Update the progress bar
        history.append(loss.numpy().mean())
        plotter.plot(history)

        # Update the model with the changed weights!
        if iter % 100 == 0:
            model.save_weights(checkpoint_prefix)

    # Save the trained model and the weights
    model.save_weights(checkpoint_prefix)


def train(params):
    batch_size = params['batch_size']
    rnn_units = params['rnn_units']
    embedding_dim = params['embedding_dim']
    learning_rate = params['learning_rate']

    # reading data
    MusicTup = namedtuple('MusicTup', ['rate', 'music_arr1', 'music_arr2'])
    data_path = params['data_path']
    data_list = []
    vocab_size = 0

    for i, f in enumerate(os.listdir(data_path)):  # todo Remove the i filter for full data
        if i < 1:
            path2f = os.path.join(data_path, f)
            if os.path.isfile(path2f):
                rate, music = read(path2f)
                music_df = pd.DataFrame(music)
                music_arr1 = music_df.iloc[:, 0].to_numpy()
                music_arr2 = music_df.iloc[:, 1].to_numpy()
                data_list.append(MusicTup(rate, music_arr1, music_arr2))
                vocab_size += len(music_df)
        else:
            continue

    # models definition for both vectorial data spaces
    model1 = build_model(len(data_list[0].music_arr1), embedding_dim, rnn_units, batch_size)

    # optimizer definition
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    train_defined_model_per_vec(model1, optimizer, data_list[0].music_arr1, params)


if __name__ == '__main__':
    params = {'batch_size': batch_size, 'rnn_units': rnn_units, 'embedding_dim': embedding_dim,
              'learning_rate': learning_rate, 'data_path': data_path, 'seq_length': seq_length,
              'training_iterations': training_iterations}

    fire.Fire(train(params))
