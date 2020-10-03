# Optimization parameters:
import os

training_iterations = 2000  # Increase this to train longer
batch_size = 4  # Experiment between 1 and 64
seq_length = 100  # Experiment between 50 and 500
learning_rate = 5e-3  # Experiment between 1e-5 and 1e-

# Model parameters:
embedding_dim = 256
rnn_units = 1024  # Experiment between 1 and 2048
optimizer_rep = 'adam'

# Data Flow
data_path = '/Users/dani/PycharmProjects/deep_music_gen/data/'
checkpoint_prefix = '/deep_music_gen/data/model_checkpoints/'



