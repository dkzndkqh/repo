import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from utils.Preprocess import Preprocess
import os

this_program_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(this_program_directory)

def read_file(file_name):
    sents = []
    with open(file_name,'r', encoding='utf-8') as f:
        lines = f.readlines()
        for idx, i in enumerate(lines):
            if i[0] == ';' and lines[idx + 1][0] == '$':
                this_sent= []
            elif i[0] == '$' and lines[idx - 1][0] == ';':
                continue
            elif i[0] == '\n':
                sents.append(this_sent)
            else:
                this_sent.append(tuple(i.split() ))
    return sents

corpus = read_file('./ner_train.txt')
pre = Preprocess(word2index_dic='../../train_tools/Dict/chatbot_dict.bin', userdic='../../utils/user_dic.tsv')

#print(corpus)

sentences, tags = [],[]
for t in corpus:
    tagged_sentence =[]
    sentence, bio_tag = [], []

    for w in t:
        tagged_sentence.append((w[1],w[3]))
        sentence.append(w[1])
        bio_tag.append(w[3])

    sentences.append(sentence)
    tags.append(bio_tag)

print("샘플 크기 : \n", len(sentences))
print("0번 째 샘플 단어  시퀀스 : \n", sentences[0])
print("0번 째 샘플 bio  태그 : \n", tags[0])
print("샘플 단어 시퀀스 최대 길이 : ", max(len(i)for i in sentences))
print("샘플 단어 시퀀스 평균 길이 : ", (sum(map(len, sentences))/len(sentences)))

#토크나이저
tag_tokenizer = preprocessing.text.Tokenizer(lower = False)
tag_tokenizer.fit_on_texts(tags)

vocab_size = len(pre.word_index) + 1
tag_size = len(tag_tokenizer.word_index) + 1

print("BIO 태그 사전 크기 : ", tag_size)
print("단어 사전 크기 : ", vocab_size)

x_train = [pre.get_wordidx_sequence(sent) for sent in sentences]
y_train = tag_tokenizer.texts_to_sequences(tags)


index_to_ner = tag_tokenizer.index_word
index_to_ner [0] = 'PAD'

# 패딩
max_len = 40
x_train = preprocessing.sequence.pad_sequences(x_train, padding='post', maxlen=max_len)
y_train = preprocessing.sequence.pad_sequences(y_train, padding='post', maxlen=max_len)

x_train, x_test, y_train, y_test = train_test_split(x_train,y_train,test_size =0.2, random_state= 1234)


#one - hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes = tag_size)
y_test = tf.keras.utils.to_categorical(y_test, num_classes = tag_size)

print('학습 샘플 시퀀스 형상 :{0}' .format(x_train.shape))
print('학습 샘플 레이블 형상 :{0}' .format(y_train.shape))
print('테스트 샘플 시퀀스 형상 :{0}' .format(x_test.shape))
print('테스트 샘플 레이블 형상 :{0}' .format(y_test.shape))

# Bi-LSTM Model

model = Sequential()
model.add(Embedding(input_dim = vocab_size, output_dim = 30,
                   input_length = max_len, mask_zero = True))

model.add (Bidirectional (LSTM(200, return_sequences = True, dropout = 0.5, recurrent_dropout = 0.25)))
model.add (TimeDistributed(Dense(tag_size, activation='softmax')))

model.compile(loss = 'categorical_crossentropy', optimizer = Adam(0.01), metrics = ['accuracy'])

model.fit (x_train, y_train, batch_size = 128, epochs = 10)

print('평가 결과 : {0}'.format(model.evaluate(x_test, y_test)[1]))

model.save('./ner_model.h5')

# f1 - score 


def seqeunces_to_tag(sequences):
    result = []
    for sequence in sequences:
        temp = []
        for pred in sequence: # [ 0.1,... 0. 2 ]
            pred_index = np.argmax(pred)
            temp.append(index_to_ner[pred_index].replace('PAD','O'))
        result.append(temp)
    return result
from seqeval.metrics import f1_score, classification_report
y_predicted = model.predict(x_test)
pred_tag = seqeunces_to_tag(y_predicted) # 예측된 NER
test_tag = seqeunces_to_tag(y_test) #  말뭉치에서 추출된 실제 NER

print('Classfication Report')
print(classification_report(test_tag,pred_tag))
print('f1-Score {0}'.format(f1_score(test_tag,pred_tag)))