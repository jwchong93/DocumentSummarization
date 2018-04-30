from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

class dataManager:
    def __init__(self):
        # Shared
        self.wordToIndex = dict()
        self.IndexToWord = dict()
        self.tokenizerSize = 0
        # Input
        self.inputData = None
        self.inputTexts = []
        # Output
        self.targetTexts = []
        self.targetData = None
        self.outputData = None
        self.outputTexts = []
        # Constant
        self.MAX_INPUT_LENGTH = 400
        self.MAX_OUTPUT_LENGTH = 80
        pass

    def initializeTokenizer (self, lookup_table, bagOfWords):
        listOfWords = list(lookup_table.keys())
        for word in bagOfWords:
            temp_index = listOfWords.index(word)
            self.wordToIndex[word] = temp_index
            self.IndexToWord[temp_index] = word
        self.tokenizerSize = len(self.wordToIndex)

    def saveInputData(self, input_texts):
        self.inputData = np.zeros(
            (len(input_texts), self.MAX_INPUT_LENGTH, self.tokenizerSize), dtype='float32')
        for t, input_text in enumerate(input_texts):
            text = input_text.lower()
            text = self.removeStopWords(text)
            text = self.replaceShortForm(text)
            text = self.removeStemming(text)
            self.inputTexts.append(text)

            input_text = text[:self.MAX_INPUT_LENGTH]
            for i, word in enumerate(input_text):
                if word not in self.wordToIndex:
                    tempWord = 'UNK'
                else:
                    tempWord = word
                index = self.wordToIndex[tempWord]
                self.inputData[t, i, index] = 1.
        print("Input Text Shape:%s" % str(self.inputData.shape))

    def saveOutputData(self, target_texts):
        self.outputData = np.zeros(
            (len(target_texts), self.MAX_OUTPUT_LENGTH, self.tokenizerSize), dtype='float32')
        self.targetData = np.zeros(
            (len(target_texts), self.MAX_OUTPUT_LENGTH, self.tokenizerSize), dtype='float32')

        for t, target_text in enumerate(target_texts):
            text = target_text.lower()
            text = self.removeStopWords(text)
            text = self.replaceShortForm(text)
            text = self.removeStemming(text)
            # Make sure the start (\t) and stop (\n) does not replace by padding
            if len(text) >= (self.MAX_OUTPUT_LENGTH - 2):
                text = "\t" + text[0:(self.MAX_OUTPUT_LENGTH - 2)] + '\n'
            else:
                text = "0" * (self.MAX_OUTPUT_LENGTH - 2 - len(text)) + "\t" + text + '\n'
            self.targetTexts.append(text)

            output_text = text[1:]
            target_text = text
            for i, word in enumerate(output_text):
                if word not in self.wordToIndex:
                    tempWord = 'UNK'
                else:
                    tempWord = word
                index = self.wordToIndex[tempWord]
                self.inputData[t, i, index] = 1.
            for i, word in enumerate(target_text):
                if word not in self.wordToIndex:
                    tempWord = 'UNK'
                else:
                    tempWord = word
                index = self.wordToIndex[tempWord]
                self.targetData[t, i, index] = 1.
        print("Output Text Shape:%s" % str(self.outputData.shape))
        print("Target Text Shape:%s" % str(self.targetData.shape))

    def replaceShortForm (self, text):
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text

    def removeStemming (self, text):
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
        return text

    def removeStopWords (self, text):
        stop_words = set(stopwords.words("english"))
        text = [w for w in text if w not in stop_words]
        text = " ".join(text)
        return text