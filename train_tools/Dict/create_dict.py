from utils.Preprocess import Preprocess
from tensorflow.keras import preprocessing
import pickle

def read_corpus_data(filename):

    with open(filename, 'r', encoding='utf-8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
    return data

corpus_data = read_corpus_data('./train_tools/Dict/corpus.txt')


p = Preprocess(word2index_dic = './train_tools/Dict/chatbot_dict.bin',userdic='./utils/user_dic.tsv')

dict = []

for c in corpus_data:
    pos  = p.pos(c[1])
    for k in pos:
        dict.append(k[0])

tokenizer = preprocessing.text.Tokenizer(oov_token='OOV')
tokenizer.fit_on_texts(dict)
word_index = tokenizer.word_index

f = open('./train_tools/Dict/chatbot_dict.bin', 'wb')

try:
    pickle.dump(word_index, f) # Memory --> file 

except Exception as e:
    print(e)

finally:
    f.close()