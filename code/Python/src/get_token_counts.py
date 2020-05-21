import os
import argparse
import re
from keras.preprocessing.text import text_to_word_sequence


def get_args():
    parser = argparse.ArgumentParser(description='Retrieve the train and test token counts for all transcripts in transcript_dir and write counts to result_file.')
    script_dir = os.path.dirname(os.path.realpath(__file__))

    parser.add_argument('-t', '--transcript_dir', dest='transcript_dir',
                        action='store', required=True,
                        default=script_dir,
                        help='directory where transcripts are stored')
    parser.add_argument('-r', '--result_file', dest='result_file',
                        action='store', required=True,
                        default=script_dir,
                        help='result file name where counts will be written')
    return parser.parse_args()

args = get_args()


#### GLOBAL VARIABLES ####

transcript_dir = args.transcript_dir
result_file = args.result_file

#### MAIN METHOD : RETRIEVE TOKEN COUNTS FOR EACH CORPUS ####

with open(result_file,'w') as f :
    f.write('file,train_tokens,test_tokens\n')

for subdir, dirs, files in os.walk(transcript_dir):
    for file in files:
        if ('.capp' in file):
            textfile = subdir+'/'+file
            print(textfile)
            with open(textfile,'r') as f :
                lines = f.readlines()
            # collect all child-directed utterances
            train = ''
            # collect all child utterances
            test = ''
            for sent in lines :
                if '*CHI:' in sent :
                    sent = re.sub('\*[A-Z]+: ', '', sent)
                    test = test+sent
                else :
                    sent = re.sub('\*[A-Z]+: ', '', sent)
                    train = train+sent
            print("Tokenize...")
            train_tokens = text_to_word_sequence(train)
            test_tokens = text_to_word_sequence(test)
            # collect token counts for both train and test and write to file
            res = str(file.split('.capp')[0]+','+str(len(train_tokens))+','+str(len(test_tokens))+'\n')
            print(res)
            with open(result_file,'a') as f :
                f.write(res)
