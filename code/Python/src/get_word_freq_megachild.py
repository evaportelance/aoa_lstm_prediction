import os
import argparse

#### GET ARGUMENTS FROM PYTHON SCRIPT CALL
def get_args():
    parser = argparse.ArgumentParser(description='Given a set of pretrained models, a list of words and the set of training sentences for the models, will calculate the raw frequency of each word in each model training set.')
    script_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('-w', '--word_list', dest='word_list',
                        action='store', required=True,
                        default=script_dir,
                        help='tsv containing word list for aoa prediction')
    parser.add_argument('-d', '--data_dir', dest='data_dir',
                        action='store', required=True,
                        default=script_dir,
                        help='directory where train sets are stored')
    parser.add_argument('-r', '--result_dir', dest='result_dir',
                        action='store', required=True,
                        default=script_dir,
                        help='directory where result matrix will be stored')
    return parser.parse_args()

import csv
from keras.preprocessing.text import Tokenizer
import numpy as np


args = get_args()


#### GLOBAL VARIABLES ####
word_list = args.word_list
data_dir = args.data_dir
result_dir = args.result_dir


class AoAWord:
    def __init__(self, word, uni_lemma, vocab):
        self.word = word
        self.uni_lemma = uni_lemma
        self.id = -1
        self.count = 0
        if word in vocab:
            self.id = vocab[word]

    def get_freq_counts(self, sequences):
        if not self.id == -1:
            for seq in sequences:
                if self.id in seq:
                    for w in seq:
                        if w == self.id:
                            self.count +=1
        return self.count

def get_train_data():
    train = []
    val = []
    childname = ""
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            if ('.train.txt' in file):
                trainfile = subdir + '/' + file
                childname = file.split('.train.txt')[0]
                with open(trainfile, 'r') as f:
                    train = f.readlines()
            if ('.val.txt' in file):
                valfile = subdir + '/' + file
                with open(valfile, 'r') as f:
                    train = f.readlines()
    return childname, train, val

### MAIN METHOD ###
words = []
with open(word_list) as file:
    reader = csv.reader(file, delimiter='\t')
    reader.__next__()
    for row in reader:
        words.append([row[0], row[1]])

childname, train, val = get_train_data()
print('PREPARE DATA FOR: ' + childname + '\n')
# Get vocabulary
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(train+val)
vocab = tokenizer.word_index
#print(vocab)
seqs = tokenizer.texts_to_sequences(train)
aoa_words = list()
for word,uni_lemma in words:
    aoa_words.append(AoAWord(word, uni_lemma, vocab))

    # write all results to the result_dir
with open(result_dir + '/' + childname + '.aoa_freq.csv', 'w') as f:
    f.write("word, uni_lemma, frequency_count" + '\n')
    for w in aoa_words:
        f.write(w.word + ',' +
                w.uni_lemma + ',' +
                str(w.get_freq_counts(seqs)) + '\n')
