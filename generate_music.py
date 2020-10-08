import argparse
import fire
import  tensorflow as tf

from config import checkpoint_prefix, embedding_dim, rnn_units
from model import build_model


def gen_music(args)
    '''TODO: Rebuild the model using a batch_size=1'''
    model = build_model(vocab_size, args.embedding_dim, args.rnn_units, batch_size=1)  # TODO
    # model = build_model('''TODO''', '''TODO''', '''TODO''', batch_size=1)

    # Restore the model weights for the last checkpoint after training
    model.load_weights(tf.train.latest_checkpoint(args.checkpoint_prefix))
    model.build(tf.TensorShape([1, None]))

    model.summary()


def parse_args():
    parser = argparse.ArgumentParser("Train and Test Music Gen with midi files")
    parser.add_argument('--data_path', type=str, default=data_path, help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=batch_size,
                        help='how many instances for encoding and normalization at each step')
    parser.add_argument('--seq_length', type=int, default=seq_length, help='Enter sequence length for each batch')
    parser.add_argument('--embedding_dim', type=int, default=embedding_dim,
                        help='Enter the output dim for the embedding level in input layer')
    parser.add_argument('--rnn_units', type=int, default=rnn_units, help='number of RNN units in the LSTM layer')
    parser.add_argument('--optimizer', type=str, default=optimizer_rep, help='text name of the wanted optimizer')
    parser.add_argument('--learning_rate', type=float, default=learning_rate, help='optimizer learning rate')
    parser.add_argument('--training_iterations', type=int, default=training_iterations)
    parser.add_argument('--checkpoint_prefix', type=str, default=checkpoint_prefix)
    return parser.parse_args()


if __name__ == '__main__':
    parsed_args = parse_args()
    fire.Fire(gen_music(parsed_args))
