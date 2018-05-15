from keras.layers import Embedding
from keras.models import Model, load_model
from keras.layers import LSTM, TimeDistributed, Input, Dense,concatenate, RepeatVector
from keras.optimizers import rmsprop
import nltk
import numpy as np
import dataReader as dr
import dataManager as dm
from pathlib import Path
import os
import time
import dataWriter
from nltk.translate.bleu_score import SmoothingFunction,sentence_bleu

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
        self.modelTrain = self.sequenceToSequenceModelTrain
        #self.modelTrain = self.recursiveModelTrain
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
        self.manager.tokenizeData (input_texts , target_texts)
        # Initialize the tokenizer with respect to the data that read into embedding
        self.manager.saveInputData(input_texts)
        self.manager.saveOutputData(target_texts)


    def modelTrain(self):
        pass

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

    def oneshotModelTraing(self):
        self.createTokenizerFromTrainingData()
        if Path(self.MODEL_PATH).exists():
            print(" -I- [modelCreation.recursiveModelTrain] Loading model from " + self.MODEL_PATH)
            model = load_model(self.MODEL_PATH)
            return model, self.manager

        # encoder input model
        inputs = Input(shape=(self.manager.MAX_INPUT_LENGTH,))
        encoder1 = Embedding(vocab_size, 128)(inputs)
        encoder2 = LSTM(128)(encoder1)
        encoder3 = RepeatVector(self.manager.MAX_OUTPUT_LENGTH)(encoder2)
        # decoder output model
        decoder1 = LSTM(128, return_sequences=True)(encoder3)
        outputs = TimeDistributed(Dense(vocab_size, activation='softmax'))(decoder1)
        # tie it together
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        return model, self.manager

    def recursiveModelTrain(self):
        self.createTokenizerFromTrainingData()
        if Path(self.MODEL_PATH).exists():
            print(" -I- [modelCreation.recursiveModelTrain] Loading model from " + self.MODEL_PATH)
            model = load_model(self.MODEL_PATH)
            return model, self.manager

        # source text input model
        inputs1 = Input(shape=(self.manager.MAX_INPUT_LENGTH,))
        am1 = Embedding(self.manager.tokenizer.__sizeof__(), 128)(inputs1)
        am2 = LSTM(128)(am1)
        # summary input model
        inputs2 = Input(shape=(self.manager.MAX_OUTPUT_LENGTH,))
        sm1 = Embedding(self.manager.tokenizer.__sizeof__(), 128)(inputs2)
        sm2 = LSTM(128)(sm1)
        # decoder output model
        decoder1 = concatenate([am2, sm2])
        outputs = Dense(self.manager.tokenizer.__sizeof__(), activation='softmax')(decoder1)
        # tie it together [article, summary] [word]
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        #model.compile(optimizer=optimizer, loss='cosine_proximity')
        model.summary(line_length=200)

        return model, self.manager

    def saveCurrentModelToFile(self, model):
        if Path(self.MODEL_PATH).exists():
            ts = int(time.time())
            os.rename(self.MODEL_PATH, self.MODEL_PATH + "_" + str(ts))
        model.save(self.MODEL_PATH)

    def refreshData(self):
        self.createTokenizerFromTrainingData()

    def dataGenerator(self):
        self.manager.reverse_input_word_map = dict(map(reversed, self.manager.inputTokenizer.word_index.items()))
        self.manager.reverse_output_word_map = dict(map(reversed, self.manager.outputTokenizer.word_index.items()))
        input_texts, output_texts = self.reader.readTrainingData(self.TRAINING_DATA_PATH, 0,
                                                                 self.NUMBER_OF_SAMPLE)
        if input_texts == [] or output_texts == []:
            raise EOFError("Data finished")
        self.manager.tokenizeData(input_texts , output_texts)
        for input_text, output_text in zip(input_texts, output_texts):
            self.manager.inputData = np.zeros(
                (1, 1,  self.manager.MAX_INPUT_LENGTH), dtype='uint32')
            splittedLines = input_text.split()
            input_sequence = self.manager.inputTokenizer.texts_to_sequences(splittedLines[0:self.manager.MAX_INPUT_LENGTH])
            for i, seq in enumerate(input_sequence):
                if seq == []:
                    seq = self.manager.inputTokenizer.texts_to_sequences(["UNK"])[0]
                self.manager.inputData[0, 0, i] = seq[0]
            self.manager.outputData = np.zeros(
                (1, 1, self.manager.MAX_OUTPUT_LENGTH), dtype='uint32')
            target_text = ["G"]
            output_sequence = self.manager.outputTokenizer.texts_to_sequences(target_text)
            for i, seq in enumerate(output_sequence):
                if seq == []:
                    seq = self.manager.outputTokenizer.texts_to_sequences(["UNK"])[0]
                self.manager.outputData[0, 0, i] = seq[0]
            yield self.manager.inputData, self.manager.outputData, input_text, output_text


    def sequenceToSequenceModelInference(self):
        if Path(self.MODEL_PATH).exists():
            print(" -I- [modelCreation.sequenceToSequenceInference] Loading model from " + self.MODEL_PATH)
            model = load_model(self.MODEL_PATH)
        else:
            print(" -E- [modelCreation.sequenceToSequenceInference] " + self.MODEL_PATH + " does not exist")
            return
        print(" -I- [modelCreation.sequenceToSequenceInference] Reading one data from" + self.TRAINING_DATA_PATH)
        self.createTokenizerFromTrainingData()
        generator = self.dataGenerator()
        BLEU_Smoothing_function = SmoothingFunction()
        for input_data, output_data, input_text, target_text in generator:
            temp_output_data = output_data
            for i in range(self.manager.MAX_OUTPUT_LENGTH - 1):
                outputSequence = model.predict([input_data, temp_output_data])
                output_text = self.manager.convertVectorsToSentences(outputSequence)
                output_list = output_text.split()
                temp_output_data[0, 0, i + 1] = self.manager.outputTokenizer.texts_to_sequences([output_list[i]])[0][0]
                print("Loop " + str(i))
                print("Expected output: " + target_text)
                print("Real output: " + output_text)
                print("BLUE Score:" + str(sentence_bleu([target_text], output_text, smoothing_function=BLEU_Smoothing_function.method4)))
            input()