#for getting args from script call
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Train LSTM models for all child transcripts in data_dir. Save models in model_dir and results for production scores in result_dir. If test, use random 60% of child utterances for training and test on 40% remaining. ')
    script_dir = os.path.dirname(os.path.realpath(__file__))

    parser.add_argument('-d', '--data_dir', dest='transcript_dir',
                        action='store', required=True,
                        default=script_dir,
                        help='directory where transcripts are stored')
    parser.add_argument('-w', '--word_list', dest='word_list',
                        action='store', required=True,
                        default=script_dir,
                        help='tsv containing word list for aoa prediction')
    parser.add_argument('-m', '--model_dir', dest='model_dir',
                        action='store', required=True,
                        default=script_dir,
                        help='directory where models are written')
    parser.add_argument('-r', '--result_dir', dest='result_dir',
                        action='store', required=True,
                        default=script_dir,
                        help='directory where results are written')
    parser.add_argument('-t', '--test', dest='train_all_data',
                        action='store_true',
                        help='include 60% of child utterances sentences in training')
    return parser.parse_args()

args = get_args()

#libs for helper functions
import re
import math
import csv
#import multiprocessing
import random
# Keras model functions and classes
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# My classes
from my_data_generator import DataGenerator
from my_decoder_generator import DecoderGenerator


#### HYPERPARAMETERS ####


# Nb of epochs (iterations through whole train set)
epochs=100
# Mini-batch size necessary for initializing data generators
batch_size = 32
# Size of word vectors
output_size = 100
#vocab_size
vocab_size =10000
# Nb of hidden neurons in 1 layer of LSTM
hidden_size = 50
#use regularizer
reg = L1L2(l1=0.01, l2=0.01)
# Shuffle order in which sentences are seen
shuffle = True


#### GLOBAL VARIABLES ####

transcript_dir = args.transcript_dir
word_list = args.word_list
model_dir = args.model_dir
result_dir = args.result_dir
train_all_data = args.train_all_data


#### HELPER FUNCTIONS ####

# A biased (p) coin flip to determine whether a child utterance will be part of train or test set
def is_test_sent(p):
    return True if random.random() < p else False


# Retrieve train and test sets for all child transcripts
def get_data_from_files():
    train = []
    val = []
    test = []
    for subdir, dirs, files in os.walk(transcript_dir):
        for file in files:
            if ('.capp' in file):
                textfile = subdir+'/'+file
                with open(textfile,'r') as f :
                    lines = f.readlines()
                for sent in lines :
                    if '*CHI:' in sent :
                        sent = re.sub('\*[A-Z]+: ', '', sent)
                        # if training on random 60% of child utterances and testing on 40% remaining
                        if train_all_data:
                            if is_test_sent(0.4):
                                test.append(sent)
                            else:
                                train.append(sent)
                        # else train only on child-directed and test on all child utterances
                        else:
                            test.append(sent)
                    else :
                        sent = re.sub('\*[A-Z]+: ', '', sent)
                        if is_test_sent(0.2):
                            val.append(sent)
                        else:
                            train.append(sent)

    # save test and train split in case we need to rerun model
    with open(model_dir+'/train/Mega_child.train.txt','w') as f :
        for line in train:
            f.write(line)
    with open(model_dir+'/train/Mega_child.val.txt','w') as f :
        for line in val:
            f.write(line)
    with open(model_dir+'/test/Mega_child.test.txt','w') as f :
        for line in test:
            f.write(line)

    return train, val, test


#### MAIN METHOD : MODEL TRAINING ####

train, val, test = get_data_from_files()

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train+val+test)
vocab = tokenizer.word_index

# transform text strings into sequences of int (representing the word's
# index in vocab)
train_seqs = tokenizer.texts_to_sequences(train)
val_seqs = tokenizer.texts_to_sequences(val)
# get the maximum length of sequences - this is needed for data generator
maxlen = max([len(seq) for seq in (train_seqs+val_seqs)])

print('vocab_size = '+str(vocab_size))
print('train_maxlen = '+str(maxlen))
print('INITIALIZE DATA GENERATORS...\n')

# Create data generators for train and test sequences
train_generator = DataGenerator(seqs = train_seqs,
                                   vocab = vocab,
                                   vocab_size = vocab_size,
                                   maxlen = maxlen,
                                   batch_size = batch_size,
                                   shuffle = shuffle)
val_generator = DataGenerator(seqs = val_seqs,
                                   vocab = vocab,
                                   vocab_size = vocab_size,
                                   maxlen = maxlen,
                                   batch_size = batch_size,
                                   shuffle = shuffle)

print('BUILDING MODEL...\n')
# initialize model
model = tf.keras.Sequential()
# add initial embedding layer
model.add(Embedding(input_dim = vocab_size,  # vocabulary size
                    output_dim = output_size,  # size of embeddings
                    input_length = maxlen-1))  # length of the padded sequences minus the last output word

#add LSTM layers (2 LAYERS)
model.add(LSTM(hidden_size, return_sequences=True))
#second layer with weight regularization
model.add(LSTM(hidden_size, bias_regularizer=reg))
# add layer regular densely connected layer to reshape to output size and use softmax activation for output layer
model.add(Dense(vocab_size, activation='softmax'))
# use RMSprop for optimization (could also use Adam or Adagrad) and cross entropy for loss function
model.compile('rmsprop', 'categorical_crossentropy', metrics=['accuracy'])

# checkpoint save best
checkpoint = ModelCheckpoint(model_dir+"/checkpoints/", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

print('TRAINING MODEL...\n')
# Train LSTM
model.fit_generator(train_generator,
                    steps_per_epoch = len(train_generator),
                    epochs = epochs,
                    verbose=2,
                    validation_data=val_generator,
                    validation_steps= len(val_generator),
                    validation_freq = 2,
                    callbacks=[checkpoint, earlystop])

# Save trained model for future use
model.save(str(model_dir+'/Mega_child_model.h5'))




del model
