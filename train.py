import argparse
import fire as fire
from tqdm import tqdm
from config import data_path, batch_size, embedding_dim, rnn_units, optimizer_rep, learning_rate, training_iterations, \
    checkpoint_prefix, seq_length
from data_extraction import run_data_extraction, get_notes_mapping_dict, vectorize_notes_by_mapping
from model import build_model, set_optimizer, train_step
from utils import PeriodicPlotter, get_batch


def train(args):
    # First we extract the Data and vectorize it
    list_test = run_data_extraction(args.data_path)
    notes2idx, idx2note = get_notes_mapping_dict(list_test)
    notes_vec = vectorize_notes_by_mapping(list_test, notes2idx)

    # Now we set the model and the optimizer
    model = build_model(len(notes_vec), args.embedding_dim, args.rnn_units, args.batch_size)
    optimizer = set_optimizer(args.optimizer, args.learning_rate)

    # train the model
    history = []
    plotter = PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')
    if hasattr(tqdm, '_instances'): tqdm._instances.clear()  # clear if it exists

    for iter in tqdm(range(args.training_iterations)):

        # Grab a batch and propagate it through the network
        x_batch, y_batch = get_batch(notes_vec, args.seq_length, args.batch_size)
        loss = train_step(x_batch, y_batch, model, optimizer)

        # Update the progress bar
        history.append(loss.numpy().mean())
        plotter.plot(history)

        # Update the model with the changed weights!
        if iter % 100 == 0:
            model.save_weights(args.checkpoint_prefix)

    # Save the trained model and the weights
    model.save_weights(args.checkpoint_prefix)


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
    args = parse_args()
    fire.Fire(train(args))
