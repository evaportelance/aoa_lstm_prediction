import copy
import numpy as np
from keras.preprocessing.sequence import pad_sequences
# Class which contains all necessary functions for decoders for production score calculation
class DecoderGenerator():
    def __init__(self, model, generator, k):
        # A Sequential model
        self.model = model
        #DataGenerator
        self.generator = generator
        # nb of beams for beam search
        self.k = k


    # get all sub sequences for each sequence in train and pad
    def prepare_seq(self, seq):
        sequences = list()
        #create all sub sequences, e.g. seq = [1,2,3], then sequences = [[1],[1,2],[1,2,3]]
        for i in range(1, len(seq)):
            sequence = seq[:i+1]
            sequences.append(sequence)
        # pad sequences, e.g. if maxlen = 4, then sequences = [[0,0,0,1],[0,0,1,2],[0,1,2,3]]
        sequences = pad_sequences(sequences, maxlen=self.generator.maxlen, padding='pre')
        sequences = np.array(sequences)
        # create context and output split, e.g. [[0,0,0],[0,0,1],[0,1,2]], [[1],[2],[3]]
        X, y = sequences[:,:-1],sequences[:,-1]

        return X, y


    # Returns the log probability of a sequence of words given the current context and the next possible word.
    def get_seq_prob(self, word, context):
        # create copy of context and add next word
        sub_seq = copy.deepcopy(context)
        sub_seq.append(word)
        # prepare sequence, such that x represents the contexts for each word y in the sequence, e.g. for sequence [1,2,3], x,y = [[],[1],[1,2]], [[1],[2],[3]]
        x, y = self.prepare_seq(sub_seq)
        # get output layers for each state in x
        p_pred = self.model.predict(x)
        # accumulate probability at each state to return probability of whole sequence
        log_p_seq = 0.0
        for i, prob in enumerate(p_pred):
            prob_word = prob[y[i]]
            log_p_seq += np.log(prob_word)

        return log_p_seq

    # Performs beam search decoder estimation for sequence and then returns 1 if original sequence is in final k beams else returns 0 .
    def beam_search_decoder(self, seq):
        result = 0
        # create "bag of words" from original sequence
        vocab = list(seq)
        # beams are composed of a context, the remaining vocab, and a score
        beams = [[list(), vocab, np.log(1.0)]]

        for i in range(len(seq)):
            # keep track of all possible candidates for beams at each state
            candidates = []
            for (context, vocab, score) in beams:
                # for each beam, find all possible next states and their scores
                for v in range(len(vocab)):
                    score = self.get_seq_prob(vocab[v], context)
                    # remove item from vocab and add it to the context for the new candidate beam
                    new_vocab = vocab[:v] + vocab[(v + 1):]
                    new_context = copy.deepcopy(context)
                    new_context.append(vocab[v])
                    candidates.append([new_context, new_vocab, score])
            # order all candidate beams next state according to their scores
            ordered = sorted(candidates, key=lambda prob: prob[2], reverse=True)
            # keep top k beams for the next iteration
            if self.k < len(ordered):
                beams = ordered[:self.k]
            else:
                beams = ordered
        for context,vocab,score in beams:
            if context == seq:
                result = 1

        return result


    # Performs the greedy decoder estimation and then returns 1 if it is equal to the original sequence else returns 0 .
    def greedy_decoder(self,seq):
        result = 0
        # create "bag of words" from sequence
        vocab = list(seq)
        context = []
        # while there are still words in the bag of words
        while vocab:
            # find the most probable next word add it to the current context and remove it from the bag of words
            (next_word, max_prob) = max([(v, self.get_seq_prob(v, context)) for v in vocab],
                                        key=lambda prob: prob[1])
            context.append(next_word)
            vocab.remove(next_word)
        # if the greedy sequence is the same as the original return 1 else 0
        if context == seq:
            result = 1

        return result

    # Returns the number of correct predictions and the overall number of test utterances for each sequence length for a given model decoder is either 'greedy' or 'beam'
    def get_performance_bylength(self, decoder):

        # Returns all sequences with less than 17 words organized by sequence length
        def get_seq_bylength(seqs):
            seqs_bylength = dict()
            for seq in seqs:
                seqlen = len(seq)
                if 1 < seqlen < 17:
                    if seqlen in seqs_bylength:
                        seqs_bylength[seqlen].append(seq)
                    else:
                        seqs_bylength[seqlen] = [seq]
            return seqs_bylength
        # organize sequences by length
        seqs_bylength = get_seq_bylength(self.generator.seqs)
        results_bylength = dict()
        for length, seqs in seqs_bylength.items():
            # for each length get the nb of correct predictions and the total nb of test utterances
            results_bylength[length] = [0, len(seqs)]
            print(str(length))
            for seq in seqs:
                # use greedy decoder
                if (decoder == 'greedy'):
                    results_bylength[length][0] += self.greedy_decoder(seq)
                # use beam search decoder
                else:
                    results_bylength[length][0] += self.beam_search_decoder(seq)

        return results_bylength
