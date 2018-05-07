import numpy as np
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
        self.MAX_OUTPUT_LENGTH = 10

    def initializeTokenizer (self, lookup_table, bagOfWords):
        listOfWords = list(lookup_table.keys())
        for word in bagOfWords:
            index = listOfWords.index(word)
            self.wordToIndex[word] = index
            self.IndexToWord[index] = word
        self.tokenizerSize = len(self.wordToIndex)

    def saveInputData(self, input_texts):
        self.inputData = np.zeros(
            (len(input_texts), self.MAX_INPUT_LENGTH), dtype='uint32')
        for t, input_text in enumerate(input_texts):
            text = input_text.lower()
            text = text.split()
            text = self.removeStemming(text)
            self.inputTexts.append(text)

            input_text = text[:self.MAX_INPUT_LENGTH]
            for i, word in enumerate(input_text):
                if word not in self.wordToIndex:
                    tempWord = 'UNK'
                else:
                    tempWord = word
                index = self.wordToIndex[tempWord]
                self.inputData[t, i] = index
        print("Input Text Shape:%s" % str(self.inputData.shape))

    def saveOutputData(self, target_texts, vector_lookup_table , dimension):
        self.outputData = np.zeros(
            (len(target_texts), self.MAX_OUTPUT_LENGTH), dtype='uint32')
        self.targetData = np.zeros(
            (len(target_texts), self.MAX_OUTPUT_LENGTH, dimension), dtype='float32')

        for t, target_text in enumerate(target_texts):
            text = target_text.lower()
            text = text.split()
            text = self.removeStemming(text)
            if len(text) >= (self.MAX_OUTPUT_LENGTH - 2):
                text = ["GO"] + text[0:(self.MAX_OUTPUT_LENGTH - 2)] + ['END']
            else:
                text = ["PAD"] * (self.MAX_OUTPUT_LENGTH - 2 - len(text)) + ["GO"] + text + ['END']
            self.targetTexts.append(text)

            output_text = text[1:]
            target_text = text
            for i, word in enumerate(output_text):
                if word not in self.wordToIndex:
                    tempWord = 'UNK'
                else:
                    tempWord = word
                index = self.wordToIndex[tempWord]
                self.outputData[t, i] = index
            for i, word in enumerate(target_text):
                if word not in self.wordToIndex:
                    tempWord = 'UNK'
                else:
                    tempWord = word
                self.targetData[t, i] = np.array(vector_lookup_table.get(tempWord), dtype='float32')
        print("Output Text Shape:%s" % str(self.outputData.shape))
        print("Target Text Shape:%s" % str(self.targetData.shape))

    def removeStemming (self, text):
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        return stemmed_words

    def convertVectorsToSentences(self, outputSequence, lookupTable, chooseBestScore = True):
        totalBestScoreWord = []
        totalLeastDiffWord = []
        for vector in outputSequence:
            bestScoreWord, leastDiffWord = self.getSimilarWords(vector, lookupTable)
            totalBestScoreWord += bestScoreWord + " "
            totalLeastDiffWord += leastDiffWord + " "

        if chooseBestScore:
            return totalBestScoreWord
        else:
            return totalLeastDiffWord

    def getSimilarWords(self, vector, table):
        bestScore = 99999
        leastDiff = 99999
        bestScoreWord = None
        leastDiffWord = None
        for word in table.keys():
            coef1MinusCoef2 = 0
            minCoef1MinusCoef2 = 99999
            lsitOfCoef1 = vector.tolist()
            listOfCoef2 = table[word].tolist()
            for coef1, coef2 in zip(lsitOfCoef1, listOfCoef2):
                different = abs(coef1 - coef2)
                coef1MinusCoef2 += different
                if different < minCoef1MinusCoef2:
                    minCoef1MinusCoef2 = different
            if minCoef1MinusCoef2 < leastDiff:
                leastDiff = minCoef1MinusCoef2
                leastDiffWord = word

            if coef1MinusCoef2 < bestScore:
                bestScore = coef1MinusCoef2
                bestScoreWord = word

        return bestScoreWord, leastDiffWord