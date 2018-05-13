from keras.layers import Embedding
from keras.models import Model, load_model
from keras.layers import LSTM, TimeDistributed, Input, Dense
from keras.optimizers import rmsprop
import nltk
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
        # Output
        # Internal
        self.current_progress = 0
        # Constant change it according to training machine
        self.NUMBER_OF_LSTM = 400
        self.NUMBER_OF_SAMPLE = 1375
        self.PROGRESS_PATH = "Data/progress.txt"
        self.TRAINING_DATA_PATH = "Data/training_data/DUC2007_Summarization_Documents/duc2007_testdocs/"
        # self.TRAINING_DATA_PATH = "Data/training_data/Reviews.csv"
        # self.GLOVE_WEIGHT_PATH = "Data/pre_trained_GloVe/glove.6B.100d.txt"
        self.MODEL_PATH = "Data/s2s.h5"
        # self.MODEL_PATH = "Data/s2s_100d_sgd_cosine_similarity.h5"

    def createTokenizerFromTrainingData(self):
        print(" -I- [modelCreation.createTokenizerFromTrainingData] Creating data set")
        # Read the last progress
        self.current_progress = self.reader.readProgress(self.PROGRESS_PATH)
        # Read the training data start from last progress
        input_texts, target_texts = self.reader.readTrainingData(self.TRAINING_DATA_PATH, self.current_progress,
                                                                 self.NUMBER_OF_SAMPLE)
        if input_texts == [] or target_texts == []:
            dataWriter.writeProgress(self.PROGRESS_PATH, 0)
            self.current_progress = self.reader.readProgress(self.PROGRESS_PATH)
            input_texts, target_texts = self.reader.readTrainingData(self.TRAINING_DATA_PATH, self.current_progress,
                                                                     self.NUMBER_OF_SAMPLE)
        self.NUMBER_OF_SAMPLE = len(input_texts)

        #Tokenize the all the data
        self.manager.tokenizeData (input_texts + target_texts)
        # Initialize the tokenizer with respect to the data that read into embedding
        self.manager.saveInputData(input_texts)
        self.manager.saveOutputData(target_texts)

    def sequenceToSequenceModelTrain(self):
        self.createTokenizerFromTrainingData()
        if Path(self.MODEL_PATH).exists():
            print(" -I- [modelCreation.sequenceToSequenceModelTrain] Loading model from " + self.MODEL_PATH)
            model = load_model(self.MODEL_PATH)
            return model, self.manager

        # Encoder
        encoder_inputs = Input(shape=(None, self.manager.MAX_INPUT_LENGTH))
        encoder_outputs, state_h, state_c = LSTM(self.NUMBER_OF_LSTM, return_state=True)(encoder_inputs)
        encoder_states = [state_h, state_c]

        # Decoder
        decoder_inputs = Input(shape=(None, self.manager.MAX_OUTPUT_LENGTH))
        decoder_lstm = LSTM(self.NUMBER_OF_LSTM, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

        # Output
        decoder_dense = TimeDistributed(Dense(self.manager.MAX_OUTPUT_LENGTH, input_shape=(self.manager.MAX_OUTPUT_LENGTH,
                                                                  self.NUMBER_OF_LSTM)))
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        #optimizer = SGD(lr=0.1, momentum=0.7, nesterov=True)
        optimizer = rmsprop(lr=0.01)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
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
        self.embedding_matrix = self.loadEmbedding()
        print(" -I- [modelCreation.sequenceToSequenceInference] Reading one data from" + self.TRAINING_DATA_PATH)
        generator = self.dataGenerator()
        self.embeddings_lookup_table.pop('UNK')
        self.embeddings_lookup_table.pop('GO')
        self.embeddings_lookup_table.pop('PAD')
        self.embeddings_lookup_table.pop('END')
        for input_data, output_data, input_text, target_text in generator:
            temp_output_data = output_data
            for i in range(self.manager.MAX_OUTPUT_LENGTH - 1):
                outputSequence = model.predict([input_data, temp_output_data])
                model.reset_states()
                output_text = self.manager.convertVectorsToSentences(outputSequence[0], self.embeddings_lookup_table,
                                                                 cosineSimilarity=True)
                output_list = output_text.split()
                temp_output_data[0, i + 1] = self.manager.wordToIndex[output_list[i]]
                print("Loop " + str(i))
                print("Expected output: " + target_text)
                print("Real output: " + output_text)
                print("BLUE Score:" + str(nltk.translate.bleu_score.sentence_bleu([target_text], output_text)))
