import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random
import numpy as np

# Data generator class for keras Sequential model
class DataGenerator(Sequence):
    def __init__(self, seqs, vocab, vocab_size, maxlen=60, batch_size=32, shuffle=False):
        self.seqs = seqs
        self.vocab = vocab
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.seqs) / self.batch_size))

# Generate one batch of data
    def __getitem__(self, index):
        """Generate one batch of data"""
        # get indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Get sequences
        seqs_temp = [self.seqs[k] for k in indexes]

        # Generate data for model X are input contexts and y are output layers
        X, y = self.__data_generation(seqs_temp)

        return X, y

# update indexes after each epoch if shuffle is true
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.seqs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


#Generate input, output data for model training given n=batch_size sequences
    def __data_generation(self, seqs_temp):
        sequences = list()
        #create all sub sequences, e.g. seq = [1,2,3], then sequences = [[1],[1,2],[1,2,3]]
        for seq in seqs_temp:
            for i in range(1, len(seq)):
                sequence = seq[:i+1]
                sequences.append(sequence)
        # pad sequences, e.g. if maxlen = 4, then sequences = [[0,0,0,1],[0,0,1,2],[0,1,2,3]]
        sequences = pad_sequences(sequences, maxlen=self.maxlen, padding='pre')
        sequences = np.array(sequences)
        # create context and output split, e.g. [[0,0,0],[0,0,1],[0,1,2]], [[1],[2],[3]]
        X, y = sequences[:,:-1],sequences[:,-1]
        # create one hot vector for output category layer
        y = tf.keras.utils.to_categorical(y, num_classes=self.vocab_size)

        return X,y
