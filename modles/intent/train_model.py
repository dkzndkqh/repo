import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Conv1D, GlobalMaxPool1D, concatenate
import pandas as pd
import os
from utils.Preprocess import Preprocess




this_program_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(this_program_directory)


train_file = "./total_train_data.csv"
data = pd.read_csv(train_file, delimiter = ',')
queries = data['query'].tolist()
intents = data['intent'].tolist()

p = Preprocess(word2index_dic='../../train_tools/Dict/chatbot_dict.bin', userdic='../../utils/user_dic.tsv')

sequences = []
for sentence in queries:
    pos = p.pos(sentence)
    keywords = p.get_keywords(pos, without_tag = True)
    seq = p.get_wordidx_sequence(keywords)
    sequences.append(seq)


#padding
from config.GlobalParams import MAX_SEQ_LEN
padded_seqs = preprocessing.sequence.pad_sequences(sequences,maxlen=MAX_SEQ_LEN, padding='post')

print('padded_seqs.shape = {0}'.format(padded_seqs.shape))
print('length of intent = {0}'.format(len(intents)))


# train : vali : test = 7:2:1

ds = tf.data.Dataset.from_tensor_slices((padded_seqs, intents))

ds = ds.shuffle((len(queries)))

train_size = int(len(padded_seqs) * 0.7)
val_size = int(len(padded_seqs) * 0.2)
test_size = int(len(padded_seqs) * 0.1)



train_ds = ds.take(train_size).batch(20)
val_ds = ds.skip(train_size).take(val_size).batch(20)
test_ds = ds.skip(train_size + val_size).take(test_size).batch(20)

dropout_prob = 0.5
EMB_SIZE = 128
EPOCH = 5
VOCAB_SIZE = len(p.word_index) + 1

# CNN Model 

input_layer = Input(shape = (MAX_SEQ_LEN, ) )
embedding_layer = Embedding(VOCAB_SIZE,EMB_SIZE, input_length = 15)(input_layer)
dropout_emb = Dropout(rate=dropout_prob)(embedding_layer)

# 3-gram
conv1 = Conv1D(filters = 128, kernel_size = 3, padding = 'valid', activation = tf.nn.relu)(dropout_emb)
pool1 = GlobalMaxPool1D()(conv1)

# 4-gram

conv2 = Conv1D(filters = 128, kernel_size = 4, padding = 'valid', activation = tf.nn.relu)(dropout_emb)
pool2 = GlobalMaxPool1D()(conv2)

# 5-gram

conv3 = Conv1D(filters = 128, kernel_size = 5, padding = 'valid', activation = tf.nn.relu)(dropout_emb)
pool3 = GlobalMaxPool1D()(conv3)

concat = concatenate([pool1,pool2,pool3])

#classfier

hidden = Dense(128, activation=tf.nn.relu)(concat)
dropout_hidden = Dropout(rate = dropout_prob)(hidden)

logits = Dense(5, name='logits')(dropout_hidden)
predictions = Dense(5, activation=tf.nn.softmax)(logits)

model = Model(inputs = input_layer, outputs = predictions)

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(train_ds, validation_data = val_ds, epochs = EPOCH, verbose = 1)

loss, accuracy = model.evaluate(test_ds, verbose=1)

print('Accuracy : {0}%'.format(accuracy * 100))
print('loss : {0}'.format(loss))

model.save('./intent_model.h5')

#y_pred = model.predict(padded_seqs)
#intents_tag = 
#pred_tag = 

#from seqeval.metrics import f1_score, classfication_report
#print(classfication_report(intents_tag, pred_tag))

#print('f1-score{0}'.format(f1_score(intents_tag, pred_tag)))