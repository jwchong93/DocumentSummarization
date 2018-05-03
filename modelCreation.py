from keras.layers import Embedding, Flatten
from keras.models import Model, load_model
from keras.layers import LSTM, TimeDistributed, Input, Dense
import numpy as np
import dataReader as dr
import dataManager as dm
from pathlib import Path
import os
import time

class modelCreation:

    def __init__(self):
        # Reader Object
        self.reader = dr.dataReader()
        self.manager = dm.dataManager()
        # Define data to read
        self.reader.readTrainingData = self.reader.readDUCData
        # self.reader.readTrainingData = self.reader.readBBCNews
        # self.reader.readTrainingData = self.reader.readReviews
        # Input
        self.inputEmbeddingLayer = None
        # Output
        self.outputEmbeddingLayer = None
        # Internal
        self.current_progress = 0
        self.embeddings_lookup_table = None
        self.embedding_matrix = None
        # Constant change it according to training machine
        self.NUMBER_OF_LSTM = 400
        self.NUMBER_OF_SAMPLE = 512
        self.EMBEDDING_DIMENSION = 100
        self.PROGRESS_PATH = "Data/progress.txt"
        self.TRAINING_DATA_PATH = "Data/training_data/DUC2007_Summarization_Documents/duc2007_testdocs/"
        # self.TRAINING_DATA_PATH = "Data/training_data/Reviews.csv"
        self.GLOVE_WEIGHT_PATH = "Data/pre_trained_GloVe/glove.6B.100d.txt"
        self.MODEL_PATH = "Data/s2s.h5"

    def createTokenizerFromTrainingData(self, training_data_path, progress_path, embedding_matrix):
        print (" -I- [modelCreation.createTokenizerFromTrainingData] Creating data set with respect to embedding")
        # Read the last progress
        self.current_progress = self.reader.readProgress(progress_path)
        # Read the training data start from last progress
        input_texts, target_texts = self.reader.readTrainingData(training_data_path, self.current_progress,
                                                                 self.NUMBER_OF_SAMPLE)
        self.NUMBER_OF_SAMPLE = len(input_texts)
        if input_texts == [] or target_texts == []:
            raise EOFError("Data finished")
        # Initialize the tokenizer with respect to the data that read into embedding
        self.manager.saveInputData(input_texts)
        self.manager.saveOutputData(target_texts, self.embeddings_lookup_table, self.EMBEDDING_DIMENSION)

    def loadEmbedding(self, weight_path, dimension):
        self.embeddings_lookup_table, bagOfWords = self.reader.readWeight(weight_path, dimension)
        self.manager.initializeTokenizer(self.embeddings_lookup_table, bagOfWords)
        print(' -I- [modelCreation.loadEmbedding] Loaded %s word vectors' % len(self.embeddings_lookup_table))
        # create a weight matrix for words in training docs
        embedding_matrix = np.zeros((self.manager.tokenizerSize + 1, dimension))
        for i, word in enumerate(list(self.manager.wordToIndex.keys())):
            embedding_vector = self.embeddings_lookup_table.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        self.inputEmbeddingLayer = Embedding(self.manager.tokenizerSize + 1, dimension, weights=[embedding_matrix],
                                             input_length=self.manager.MAX_INPUT_LENGTH, trainable=False,
                                             input_shape=(self.NUMBER_OF_SAMPLE, self.manager.MAX_INPUT_LENGTH))

        self.outputEmbeddingLayer = Embedding(self.manager.tokenizerSize + 1, dimension, weights=[embedding_matrix],
                                             input_length=self.manager.MAX_OUTPUT_LENGTH, trainable=False,
                                             input_shape=(self.NUMBER_OF_SAMPLE, self.manager.MAX_OUTPUT_LENGTH))
        return embedding_matrix

    def sequenceToSequenceModel(self):
        self.embedding_matrix = self.loadEmbedding(self.GLOVE_WEIGHT_PATH, self.EMBEDDING_DIMENSION)
        self.createTokenizerFromTrainingData(self.TRAINING_DATA_PATH, self.PROGRESS_PATH, self.embedding_matrix)
        if Path(self.MODEL_PATH).exists():
            print(" -I- [modelCreation.sequenceToSequenceModel] Loading model from " + self.MODEL_PATH)
            model = load_model(self.MODEL_PATH)
            return model, self.manager

        # Encoder
        encoder_inputs = Input(shape=(None, ))
        x = self.inputEmbeddingLayer(encoder_inputs)
        encoder_outputs, state_h, state_c = LSTM(self.NUMBER_OF_LSTM, return_state=True)(x)
        encoder_states = [state_h, state_c]

        # Decoder
        decoder_inputs = Input(shape=(None, ))
        x = self.outputEmbeddingLayer(decoder_inputs)
        decoder_lstm = LSTM(self.NUMBER_OF_LSTM, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(x, initial_state=encoder_states)

        # Output
        decoder_dense = TimeDistributed( Dense(self.EMBEDDING_DIMENSION, input_shape=(self.manager.MAX_OUTPUT_LENGTH,
                                                                  self.NUMBER_OF_LSTM), activation='softmax'))
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer='adam', loss='mean_squared_error')
        # model.compile(optimizer='adam', loss='cosine_proximity')
        model.summary(line_length=200)

        return model, self.manager

    def saveCurrentModelToFile(self, model):
        if Path(self.MODEL_PATH).exists():
            ts = int(time.time())
            os.rename(self.MODEL_PATH, self.MODEL_PATH + "_" + str(ts))
        model.save(self.MODEL_PATH)

    def refreshData(self):
        self.createTokenizerFromTrainingData(self.TRAINING_DATA_PATH, self.PROGRESS_PATH, self.embedding_matrix)