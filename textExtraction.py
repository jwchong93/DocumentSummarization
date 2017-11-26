from nltk import tokenize
from nltk.corpus import stopwords
import os


#configuration variables
class MODE :
    word,sentence =range(2)
root_directory = os.getcwd() + "/Training Database/single/single-paper-ann"
data_extension = str(".txt")
answer_extension = str(".ann")
result_extension = str(".result")
language = "english"
processing_mode = MODE.sentence
#end configuration variables

data_directory = [ x for x in os.listdir(root_directory)]
all_file_name = list(set([file_name.split(".")[0] for file_name in data_directory ]))


for file in all_file_name[0:1]:
    print ("Processing " + file)
    with open(root_directory + "/" + file + data_extension, 'r') as content:
        document = content.read()
    output_file = open(root_directory + "/" + file + result_extension ,'w')
    #tokenizing
    if processing_mode == MODE.word:
        tokenized_content = tokenize.word_tokenize(document)
    elif processing_mode == MODE.sentence:
        tokenized_content = tokenize.sent_tokenize(document)
    else:
        print ("MODE is not set, please go to the config section to edit this")

    #removing stop words
    stop_words = set (stopwords.words(language))
    filtered_token = []
    if processing_mode == MODE.word:
        filtered_token = [token for token in tokenized_content if token not in stop_words]
    elif processing_mode == MODE.sentence:
        filtered_token = [token for token in tokenized_content if token not in stop_words]
    result_document = " ".join(filtered_token)
    print ("Writing to " + root_directory + "/" + file + result_extension)
    output_file.write(result_document)

