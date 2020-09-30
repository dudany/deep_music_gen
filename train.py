import fire as fire
import tensorflow as tf
from tqdm import tqdm

from config import checkpoint_prefix
from utils import compute_loss, PeriodicPlotter, get_batch


@tf.function
def train_step(x, y, model, optimizer_obj):
    # Use tf.GradientTape()
    with tf.GradientTape() as tape:
        y_hat = model(x)
        loss = compute_loss(y, y_hat)
    # Now, compute the gradients
    grads = tape.gradient(loss, model.trainable_variables)

    # Apply the gradients to the optimizer so it can update the model accordingly
    optimizer_obj.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def train_defined_model(model, params):
    # model params definition
    history = []
    plotter = PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')
    seq_length = params['seq_length']
    batch_size = params['batch_size']
    training_iterations = params['training_iterations']

    if hasattr(tqdm, '_instances'): tqdm._instances.clear()  # clear if it exists

    for iter in tqdm(range(training_iterations)):
        # Grab a batch and propagate it through the network
        x_batch, y_batch = get_batch(vectorized_songs, seq_length, batch_size)
        loss = train_step(x_batch, y_batch)

        # Update the progress bar
        history.append(loss.numpy().mean())
        plotter.plot(history)

        # Update the model with the changed weights!
        if iter % 100 == 0:
            model.save_weights(checkpoint_prefix)

    # Save the trained model and the weights
    model.save_weights(checkpoint_prefix)

    if __name__ == '__main__':
        fire.Fire(train_defined_model)
