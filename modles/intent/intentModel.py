import tensorflow as tf
from tensorflow.keras import preprocessing # padding
from tensorflow.keras.models import Model, load_model

from config.GlobalParams import MAX_SEQ_LEN

class IntentModel:
    def __init__(self, model_name, preprocess):
        self.labels = { 0 :'인사', 1: '욕설', 2:'주문', 3:'예약', 4:'기타'}

        self.model = load_model(model_name)
        self.preprocess = preprocess



    def predict_class(self, query): # query 입력 input
        pos = self.preprocess.pos(query)
        keywords = self.preprocess.get_keywords(pos, without_tag = True)
        sequences = [self.preprocess.get_wordidx_sequence(keywords)]

        padded_seqs = preprocessing.sequence.pad_sequences(sequences,maxlen=MAX_SEQ_LEN, padding='post')

        predict = self.model.predict(padded_seqs)
        predict_class = tf.math.argmax(predict, axis = 1)
        predict_class_acc = tf.gather(predict, predict_class, axis=1)

       

        

        return str((predict_class_acc.numpy()[0][0] * 100).round()) + '%', predict_class.numpy()[0]
    
#사용법
#p = preprocessing(word2index_dic='',)
#intent = IntentModel('..')
#predict = intent.predict_class("입력하고 싶은 말들")
#predict_label =intent.labels[predict]
#print('label of class : {0} - {1}.format(predict, predict_label))