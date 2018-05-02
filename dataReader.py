import csv
import numpy as np
import os
from html.parser import HTMLParser
import urllib3
from bs4 import BeautifulSoup
from News_Query.FrequencySummarizer import FrequencySummarizer


class HeadingParser(HTMLParser):
    inHeading = False
    isHeadline = False
    targetText = ""
    inputText = ""

    def handle_starttag(self, tag, attrs):
        if tag == 'p':
            self.inHeading = True
        elif tag == 'headline':
            self.isHeadline = True

    def handle_endtag(self, tag):
        if tag == 'p':
            self.inHeading =False
        elif tag == 'headline':
            self.isHeadline = False

    def handle_data(self, data):
        if self.inHeading:
            self.inputText += data
        elif self.isHeadline:
            self.targetText = data



class dataReader:
    def __init__(self):
        pass

    def readProgress (self, path):
        fh = open(path, 'r')
        current_progress = int(fh.read())
        fh.close()
        return current_progress

    def readTrainingData (self, training_data_path, current_progress, sample_to_extract):
        # Remain Empty, this will part of configuration to use DUC or Reviews from Kaggle
        pass

    def readDUCData (self, training_data_path, current_progress, sample_to_extract):
        all_documents = []
        input_texts = []
        target_texts = []
        for root, subdir, files in os.walk(training_data_path):
            for file in files:
                all_documents.append(os.path.join(root, file))
        for n, document in enumerate(all_documents):
            if n < current_progress:
                continue
            elif n >= current_progress + sample_to_extract:
                break
            elif n >= current_progress:
                parser = HeadingParser()
                fh = open(document, 'rt')
                html = fh.read()
                fh.close()
                parser.feed(html)
                input_texts.append(parser.inputText.replace('\n', ' ').strip())
                target_texts.append(parser.targetText.replace('\n', ' ').strip())
        print(input_texts, target_texts)
        return input_texts, target_texts

    def readReviews (self, training_data_path, current_progress, sample_to_extract):
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

    def readBBCNews(self, training_data_path, current_progress, sample_to_extract):
        def get_only_text(url):
            """
             return the title and the text of the article
             at the specified url
            """
            page = urllib3.PoolManager().request('GET', url).data.decode('utf8')
            soup = BeautifulSoup(page, "html5lib")
            text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
            return soup.title.text, text

        feed_xml = urllib3.PoolManager().request('GET', 'http://feeds.bbci.co.uk/news/rss.xml').data
        feed = BeautifulSoup(feed_xml.decode('utf8'), "html5lib")
        to_summarize = list(map(lambda p: p.text, feed.find_all('guid')))

        fs = FrequencySummarizer()
        input_texts = []
        target_texts = []
        for article_url in to_summarize[:sample_to_extract]:
            title, input_text = get_only_text(article_url)
            target_text = ""
            for s in fs.summarize(input_text, 2):
                target_text += s
            input_texts.append(input_text)
            target_texts.append(target_text)
        return input_texts, target_texts

    def readWeight (self, weight_path, dimension):
        bagOfWords = []
        embeddings_index = dict()
        f = open(weight_path, encoding="utf8")
        for i, line in enumerate(f):
            if i >= 200000:
                break
            values = line.split()
            word = values[0].lower()
            if self.wordIsNumber(word):
                continue
            else:
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
                bagOfWords.append(word)

        f.close()
        embeddings_index['\t'] = np.full(dimension, 0.5, dtype='float32')
        embeddings_index['\n'] = np.full(dimension, 1.0, dtype='float32')
        embeddings_index['UNK'] = np.full(dimension, 0.0, dtype='float32')
        bagOfWords.append('\t')
        bagOfWords.append('\n')
        bagOfWords.append('UNK')
        return embeddings_index, bagOfWords

    def wordIsNumber (self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False