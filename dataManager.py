from keras.preprocessing.sequence import pad_sequences
import numpy as np
class dataManager:
    def __init__(self):
        # Input
        self.inputData = None
        self.inputTexts = []
        self.inputBagOfWords = []
        # Output
        self.targetTexts = []
        self.targetData = None
        self.outputData = None
        self.outputTexts = []
        # Constant
        self.MAX_INPUT_LENGTH = 400
        self.MAX_OUTPUT_LENGTH = 80
        pass

    def saveInputData(self, input_texts, lookup_table):
        self.inputTexts = input_texts
        # Convert lookup table to list
        lookup_list = list(lookup_table.keys())

        for input_text in self.inputTexts:
            extracted_sequence = []
            for char in input_text:
                if char in lookup_list:
                    # Get the index of the word
                    extracted_sequence.append([lookup_list.index(char)])
                    if char not in self.inputBagOfWords:
                        self.inputBagOfWords.append(char)
            if self.inputData is None:
                self.inputData = pad_sequences(extracted_sequence, maxlen=self.MAX_INPUT_LENGTH, padding='post')
            else:
                self.inputData = np.vstack((self.inputData, pad_sequences(extracted_sequence, maxlen=self.MAX_INPUT_LENGTH, padding='post')))

    def saveOutputData(self, target_texts, lookup_table, data_size):
        for target_text in target_texts:
            # Make sure the start (\t) and stop (\n) does not replace by padding
            if len(target_text) >= (self.MAX_OUTPUT_LENGTH - 2):
                target_text = "\t" + target_text[0:(self.MAX_OUTPUT_LENGTH - 2)] + '\n'
            else:
                target_text = "\t" + target_text + '\n'
            self.targetTexts.append(target_text)

        for t, target_text in enumerate(self.targetTexts):
            output_sequence = []
            target_sequence = []
            for i, char in enumerate(target_text):
                vector = lookup_table.get(char)
                if vector is not None:
                    output_sequence.append(vector)
                    if i > 0:
                        target_sequence.append(vector)

            if self.outputData is None:
                self.outputData = pad_sequences(output_sequence, maxlen=self.MAX_OUTPUT_LENGTH, padding='pre', dtype='float32')
            else:
                self.outputData = np.vstack((self.outputData, (pad_sequences(output_sequence, maxlen=self.MAX_OUTPUT_LENGTH, padding='pre', dtype='float32'))))

            if self.targetData is None:
                self.targetData = pad_sequences(target_sequence, maxlen=self.MAX_OUTPUT_LENGTH, padding='pre', dtype='float32')
            else:
                self.targetData = np.vstack((self.targetData, (pad_sequences(target_sequence, maxlen=self.MAX_OUTPUT_LENGTH, padding='pre', dtype='float32'))))
        print(self.outputData)
