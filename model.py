import tensorflow as tf

from utils import compute_loss


def LSTM(rnn_units):
    return tf.keras.layers.LSTM(
        rnn_units,
        return_sequences=True,
        recurrent_initializer='glorot_uniform',
        recurrent_activation='sigmoid',
        stateful=True,
    )


def build_model(input_dim, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential(
        [tf.keras.layers.Embedding(input_dim, embedding_dim, batch_input_shape=[batch_size, None]),
         LSTM(rnn_units), tf.keras.layers.Dense(input_dim)])

    return model


def set_optimizer(opt_type: str, learning_rate):
    optimizer = None
    if opt_type == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate)

    return optimizer


@tf.function
def train_step(x, y, model, optimizer):
    # Use tf.GradientTape()
    with tf.GradientTape() as tape:
        y_hat = model(x)

        loss = compute_loss(y, y_hat)

    # Now, compute the gradients
    grads = tape.gradient(loss, model.trainable_variables)

    # Apply the gradients to the optimizer so it can update the model accordingly
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss
