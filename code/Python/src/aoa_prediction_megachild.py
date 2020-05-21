import os
import sys
import argparse

#### GET ARGUMENTS FROM PYTHON SCRIPT CALL
def get_args():
    parser = argparse.ArgumentParser(description='Given a set of pretrained models, a list of words and the set of training sentences for the models, will calculate the average surprisal for each word from list accross trainning contexts for each model.')
    script_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('-w', '--word_list', dest='word_list',
                        action='store', required=True,
                        default=script_dir,
                        help='tsv containing word list for aoa prediction')
    parser.add_argument('-m', '--model_dir', dest='model_dir',
                        action='store', required=True,
                        default=script_dir,
                        help='directory where models and train sets are stored')
    parser.add_argument('-r', '--result_dir', dest='result_dir',
                        action='store', required=True,
                        default=script_dir,
                        help='directory where result matrix will be stored')
    return parser.parse_args()

import csv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


args = get_args()



#### GLOBAL VARIABLES ####
word_list = args.word_list
model_dir = args.model_dir
result_dir = args.result_dir
vocab_size =10000
epsilon = sys.float_info.epsilon

class AoAWord:
    def __init__(self, word, uni_lemma, maxlen, vocab):
        self.word = word
        self.uni_lemma = uni_lemma
        self.id = -1
        self.maxlen = maxlen
        self.contexts = []
        self.surprisals = []
        if word in vocab:
            self.id = vocab[word]

    def get_contexts_surprisals(self, sequences, model):
        if not self.id == -1:
            contexts = []
            for seq in sequences:
                if self.id in seq:
                    context = []
                    for w in seq:
                        if w == self.id:
                            break
                        context.append(w)
                    context.append(self.id)
                    contexts.append(context)
            contexts = pad_sequences(contexts, maxlen=self.maxlen, padding='pre')
            self.contexts= np.array(contexts)
            X, y = self.contexts[:,:-1],self.contexts[:,-1]
            p_pred = model.predict(X)
            for i, prob in enumerate(p_pred):
                print(prob[y[i]])
                self.surprisals.append(-np.log(prob[y[i]]+epsilon))


    def get_avg_surprisal(self, sequences, model):
        score = 0.0
        if not self.contexts:
            self.get_contexts_surprisals(sequences, model)
        if len(self.surprisals) == 0:
            return "NA"
        else:
            for surprisal in self.surprisals:
                score += surprisal
            print(self.word + ' total: '+ str(score))
            score = score/len(self.surprisals)
            print(self.word + ' nb contexts: '+ str(len(self.surprisals)))
            #print('surprisals: ')
            #print(*self.surprisals, sep = ", ")
            return score

def get_model_train_test():
    data = []
    for subdir, dirs, files in os.walk(model_dir):
        for file in files:
            if ('.h5' in file):
                model_file = subdir + '/' + file
                this_model = load_model(model_file)
                childname = file.split('_model.h5')[0]
                trainfile = subdir +'/train/' + childname + '.train.txt'
                valfile = subdir +'/train/' + childname + '.val.txt'
                testfile = subdir + '/test/' + childname + '.test.txt'
                with open(trainfile, 'r') as f:
                    train = f.readlines()
                with open(valfile, 'r') as f:
                    val = f.readlines()
                with open(testfile, 'r') as f:
                    test = f.readlines()
                data.append((childname, this_model, train, val, test))
    return data


### MAIN METHOD ###
words = []
with open(word_list) as file:
    reader = csv.reader(file, delimiter='\t')
    reader.__next__()
    for row in reader:
        words.append([row[0], row[1]])

data = get_model_train_test()
aoa_corpus = dict()
for childname, model, train, val, test in data:
    print('PREPARE DATA FOR: ' + childname + '\n')
    # Get vocabulary
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(train + val + test)
    vocab = tokenizer.word_index
    #print(vocab)
    seqs = tokenizer.texts_to_sequences(train + val)
    maxlen = max([len(seq) for seq in seqs])
    aoa_words = list()
    for word,uni_lemma in words:
        aoa_words.append(AoAWord(word, uni_lemma, maxlen, vocab))
    aoa_corpus[childname] = aoa_words

        # write all results to the result_dir
    with open(result_dir + '/' + childname + '.aoa_result.csv', 'w') as f:
        csvwriter = csv.writer(f, delimiter=',')
        csvwriter.writerow(["word", "uni_lemma", "avg_surprisal"])
        for w in aoa_words:
            row= [w.word,w.uni_lemma,w.get_avg_surprisal(seqs, model)]
            csvwriter.writerow(row)

    with open(result_dir + '/' + childname + '.aoa_all_surprisals.csv', 'w') as f:
        csvwriter = csv.writer(f, delimiter=',')
        for w in aoa_words:
            row= [w.word]+[w.surprisals]
            csvwriter.writerow(row)
