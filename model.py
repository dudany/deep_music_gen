import tensorflow as tf


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
