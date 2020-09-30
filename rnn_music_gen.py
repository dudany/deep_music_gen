import pandas as pd
# import pydub
import os
import tensorflow as tf
from collections import namedtuple
from scipy.io.wavfile import read, write
from config import embedding_dim, rnn_units, batch_size, learning_rate, num_training_iterations
from utils import compute_loss, PeriodicPlotter, get_batch
from model import build_model
from tqdm import tqdm

"""**If you want to train your model on mp3 files, the following lines of code will do the trick.**"""

# converting mp3 file to wav file
# sound = pydub.AudioSegment.from_mp3("Numb_piano.mp3")
# sound.export("Numb.wav", format="wav")
# sound = pydub.AudioSegment.from_mp3("Eminem.mp3")
# sound.export("Eminem.wav", format="wav")

"""as we Have .wav data set we'll not to the conversion"""

# loading the wav files
MusicTup = namedtuple('MusicTup', ['rate', 'music_arr1', 'music_arr2'])
data_path = '/Users/dani/PycharmProjects/deep_music_gen/Full_Drum_Loops'
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

# We will make 2 RNN models for each of the data, lets the Define the RNN model class
# TODO : check about the vocab size
model1 = build_model(vocab_size, embedding_dim, rnn_units, batch_size)

optimizer = tf.keras.optimizers.Adam(learning_rate)





