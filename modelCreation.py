from keras.layers import Embedding
from keras.models import Model, load_model
from keras.layers import LSTM, TimeDistributed, Input, Dense
from keras.optimizers import rmsprop

import numpy as np
import dataReader as dr
import dataManager as dm
from pathlib import Path
import os
import time
import dataWriter

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
        # self.MODEL_PATH = "Data/s2s_100d_sgd_cosine_similarity.h5"

    def createTokenizerFromTrainingData(self):
        self.NUMBER_OF_SAMPLE = 512
        print(" -I- [modelCreation.createTokenizerFromTrainingData] Creating data set with respect to embedding")
        # Read the last progress
        self.current_progress = self.reader.readProgress(self.PROGRESS_PATH)
        # Read the training data start from last progress
        input_texts, target_texts = self.reader.readTrainingData(self.TRAINING_DATA_PATH, self.current_progress,
                                                                 self.NUMBER_OF_SAMPLE)
        if input_texts == [] or target_texts == []:
            self.NUMBER_OF_SAMPLE = 512
            dataWriter.writeProgress(self.PROGRESS_PATH, 0)
            self.current_progress = self.reader.readProgress(self.PROGRESS_PATH)
            input_texts, target_texts = self.reader.readTrainingData(self.TRAINING_DATA_PATH, self.current_progress,
                                                                     self.NUMBER_OF_SAMPLE)
        self.NUMBER_OF_SAMPLE = len(input_texts)

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

    def sequenceToSequenceModelTrain(self):
        self.embedding_matrix = self.loadEmbedding(self.GLOVE_WEIGHT_PATH, self.EMBEDDING_DIMENSION)
        self.createTokenizerFromTrainingData()
        if Path(self.MODEL_PATH).exists():
            print(" -I- [modelCreation.sequenceToSequenceModelTrain] Loading model from " + self.MODEL_PATH)
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
        decoder_dense = TimeDistributed(Dense(self.EMBEDDING_DIMENSION, input_shape=(self.manager.MAX_OUTPUT_LENGTH,
                                                                  self.NUMBER_OF_LSTM)))
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        #optimizer = SGD(lr=0.1, momentum=0.7, nesterov=True)
        optimizer = rmsprop(lr=0.01)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        #model.compile(optimizer=optimizer, loss='cosine_proximity')
        model.summary(line_length=200)

        return model, self.manager

    def     saveCurrentModelToFile(self, model):
        if Path(self.MODEL_PATH).exists():
            ts = int(time.time())
            os.rename(self.MODEL_PATH, self.MODEL_PATH + "_" + str(ts))
        model.save(self.MODEL_PATH)

    def refreshData(self):
        self.createTokenizerFromTrainingData()

    def dataGenerator(self):
        input_texts, output_texts = self.reader.readTrainingData(self.TRAINING_DATA_PATH, 0,
                                                                 self.NUMBER_OF_SAMPLE)
        if input_texts == [] or output_texts == []:
            raise EOFError("Data finished")

        for input_text, output_text in zip(input_texts, output_texts):
            self.manager.inputData = np.zeros(
                (1, self.manager.MAX_INPUT_LENGTH), dtype='uint32')
            for i, word in enumerate(input_text[:self.manager.MAX_INPUT_LENGTH]):
                if word not in self.manager.wordToIndex:
                    tempWord = 'UNK'
                else:
                    tempWord = word
                self.manager.inputData[0, i] = self.manager.wordToIndex[tempWord]
            self.manager.outputData = np.zeros(
                (1, self.manager.MAX_OUTPUT_LENGTH), dtype='float32')
            target_text = ["GO"]
            for i, word in enumerate(target_text):
                if word not in self.manager.wordToIndex:
                    tempWord = 'UNK'
                else:
                    tempWord = word
                self.manager.outputData[0, i] = self.manager.wordToIndex[tempWord]
            yield self.manager.inputData, self.manager.outputData, input_text, output_text


    def sequenceToSequenceModelInference(self):
        if Path(self.MODEL_PATH).exists():
            print(" -I- [modelCreation.sequenceToSequenceInference] Loading model from " + self.MODEL_PATH)
            model = load_model(self.MODEL_PATH)
        else:
            print(" -E- [modelCreation.sequenceToSequenceInference] " + self.MODEL_PATH + " does not exist")
            return
        print(" -I- [modelCreation.sequenceToSequenceInference] Loading GloVe vector from " + self.GLOVE_WEIGHT_PATH)
        self.embedding_matrix = self.loadEmbedding(self.GLOVE_WEIGHT_PATH, self.EMBEDDING_DIMENSION)
        print(" -I- [modelCreation.sequenceToSequenceInference] Reading one data from" + self.TRAINING_DATA_PATH)
        generator = self.dataGenerator()
        for input_data, output_data, input_text, target_text in generator:
            temp_output_data = output_data
            for i in range(self.manager.MAX_OUTPUT_LENGTH):
                outputSequence = model.predict([input_data, temp_output_data])
                temp_output_data[0, i+1] = outputSequence[0, i]
                output_text = self.manager.convertVectorsToSentences(outputSequence[0], self.embeddings_lookup_table,
                                                                 chooseBestScore=False)
                print(output_text)
            input("Press Enter to Continue...")
