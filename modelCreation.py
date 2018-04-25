from keras.layers import Embedding, Flatten
from keras.models import Model
from keras.layers import LSTM, Dense, Input
import numpy as np
import dataReader as dr
import dataManager as dm

class modelCreation:

    def __init__(self):
        # Reader Object
        self.reader = dr.dataReader()
        self.manager = dm.dataManager()
        # Input
        self.inputEmbeddingLayer = None
        # Output
        # Internal
        self.current_progress = 0
        self.embeddings_lookup_table = None
        # Constant change it according to training machine
        self.NUMBER_OF_LSTM = 1024
        self.NUMBER_OF_SAMPLE = 128
        self.PROGRESS_PATH = "Data/progress.txt"
        self.TRAINING_DATA_PATH = "Data/training_data/Reviews.csv"
        self.GLOVE_WEIGHT_PATH = "Data/pre_trained_GloVe/glove.6B.100d.txt"
        self.MODEL_PATH = "Data/s2s.h5"

    def createTokenizerFromTrainingData(self, training_data_path, progress_path):
        # Read the last progress
        self.current_progress = self.reader.readProgress(progress_path)
        # Read the training data start from last progress
        input_texts, target_texts = self.reader.readTrainingData(training_data_path, self.current_progress,
                                                                 self.NUMBER_OF_SAMPLE)
        # Using Tokenizer to change text into sequence
        self.manager.saveInputData(input_texts, self.embeddings_lookup_table)
        self.manager.saveOutputData(target_texts, self.embeddings_lookup_table, self.NUMBER_OF_SAMPLE)

    def loadEmbedding(self, weight_path, dimension):
        self.embeddings_lookup_table = self.reader.readWeight(weight_path)
        print(' -I- [modelCreation.loadEmbedding] Loaded %s word vectors' % len(self.embeddings_lookup_table))
        # create a weight matrix for words in training docs
        input_embedding_matrix = np.zeros((self.manager.MAX_INPUT_LENGTH, dimension))
        for word, i in self.manager.inputBagOfWords:
            embedding_vector = self.embeddings_lookup_table.get(word)
            if embedding_vector is not None:
                input_embedding_matrix[i] = embedding_vector
        self.inputEmbeddingLayer = Embedding(self.manager.MAX_INPUT_LENGTH, dimension, weights=[input_embedding_matrix],
                                             input_length=self.manager.MAX_INPUT_LENGTH, trainable=False)

    def sequenceToSequenceModel(self):
        self.loadEmbedding(self.GLOVE_WEIGHT_PATH, 100)
        self.createTokenizerFromTrainingData(self.TRAINING_DATA_PATH, self.PROGRESS_PATH)

        # Encoder
        encoder_inputs = Input(shape=(None, ))
        x = self.inputEmbeddingLayer(encoder_inputs)
        encoder_outputs, state_h, state_c = LSTM(self.NUMBER_OF_LSTM, return_state=True)(x)
        encoder_states = [state_h, state_c]

        # Decoder
        decoder_inputs = Input(shape=(None, self.manager.MAX_OUTPUT_LENGTH))
        decoder_lstm = LSTM(self.NUMBER_OF_LSTM, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

        # Output
        decoder_dense = Dense(self.manager.MAX_OUTPUT_LENGTH, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        model.summary(line_length=200)
        return model, self.manager

    def saveCurrentModelToFile(self, model):
        model.save(self.MODEL_PATH)