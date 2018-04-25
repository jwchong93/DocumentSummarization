import csv
import numpy as np

class dataReader:
    def __init__(self):
        pass

    def readProgress (self, path):
        fh = open(path, 'r')
        current_progress = int(fh.read())
        fh.close()
        return current_progress

    def readTrainingData (self, training_data_path, current_progress, sample_to_extract):
        input_texts = []
        target_texts = []
        with open(training_data_path, encoding="utf8") as csvfile:
            reader = csv.DictReader(csvfile)
            row_counter = -1
            for row in reader:
                row_counter += 1
                if row_counter < current_progress:
                    continue
                elif row_counter >= current_progress + sample_to_extract:
                    break
                elif row_counter >= current_progress:
                    input_text, target_text = row['Text'], row['Summary']
                    input_texts.append(input_text)
                    target_texts.append(target_text)
        return input_texts, target_texts

    def readWeight (self, weight_path):
        embeddings_index = dict()
        f = open(weight_path, encoding="utf8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        return embeddings_index