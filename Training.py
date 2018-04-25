from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, CuDNNLSTM
import csv
import numpy as np
from pathlib import Path
import os
import time

from keras.utils.vis_utils import plot_model

fh = open('Data/progress.txt', 'r')
current_progress = int(fh.read())
fh.close()

batch_size = 32  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 512  # Latent dimensionality of the encoding space.
num_samples = 1024  # Number of samples to train on.
max_input_length = 1024
max_output_length = 256
# Path to the data txt file on disk.
data_path = 'Data/training_data/Reviews.csv'
model_path = 's2s.h5'

# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()

with open(data_path, encoding="utf8") as csvfile:
    reader = csv.DictReader(csvfile)
    row_counter = -1
    for row in reader:
        row_counter += 1
        if row_counter < current_progress:
            continue
        elif row_counter >= current_progress+num_samples:
            break
        elif row_counter >= current_progress:
            input_text, target_text = row['Text'], row['Summary']
            # We use "tab" as the "start sequence" character
            # for the targets, and "\n" as "end sequence" character.
            output_data_length = len(target_text)
            if output_data_length >= max_output_length-1:
                target_text = '\t' + target_text[0:max_output_length-2] + '\n'
            else:
                target_text = '\t' + target_text[0:output_data_length] + '\n'

            if len(input_text) > max_input_length:
                input_text = input_text[0:max_input_length]
            input_texts.append(input_text)
            target_texts.append(target_text)
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

print (input_characters)
print(input_token_index)

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, max_input_length),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, max_output_length),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, max_output_length),
    dtype='float32')

print(len(input_texts))
print(len(input_texts[0]))
print(len(target_texts))
print(len(target_texts[0]))

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.

if Path(model_path).exists():
    model = load_model(model_path)
    ts = int(time.time())
    os.rename(model_path, str(ts) + "_" + model_path)
else:
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, max_input_length))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, max_output_length))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(max_output_length, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.summary(line_length=300)
    # plot_model(model, to_file='model.png', show_shapes=True)
    # Run training
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    print(decoder_target_data.shape)
    print(encoder_input_data.shape)
    print(decoder_input_data.shape)
"""
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# Save model
model.save(model_path)
fh = open('progress.txt', 'w')
fh.write(str(current_progress+num_samples))
fh.close()
"""