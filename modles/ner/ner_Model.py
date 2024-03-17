import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import preprocessing
import numpy as np

class NerModel:
    def __init__(self,model_name,preprocess ):
    
        self.index_to_ner = { 1 : 'O', 2 : 'B_DT', 3 : 'B_FOOD', 4 : 'I', 5 : 'B_OG', 6 : 'B_PS',
                             7 : 'B_LC', 8 : 'NNP', 9 : 'B_TI',0 : 'PAD'} #Training pickle 저장된거 로드

        self.model = load_model(model_name)
        self.preprocess =  preprocess
    def predict(self, query):
        pos = self.preprocess.pos(query)
        keywords = self.preprocess.get_keywords(pos, without_tag = True)
        sequences = [self.preprocess.get_wordidx_sequence(keywords)]

        max_len= 40
        padded_seqs = preprocessing.sequence.pad_sequences(sequences,maxlen=max_len, padding='post',value = 0)

        predict = self.model.predict(np.array( [padded_seqs[0] ] ))
        predict_class = tf.math.argmax(predict, axis = -1) #classindex 값이 넘어옴
        tags = [self.index_to_ner[idx ] for idx in predict_class.numpy()[0]]
        if len(keywords) == 0 : return None
        return list(zip(keywords,tags))  # [( keyword, BIO_tag)] 형태로 나옴 zip으로하면 tuple형태로나옴

    def predict_tag(self, query): #tag 정보만 나오게
        pos = self.preprocess.pos(query)
        keywords = self.preprocess.get_keywords(pos, without_tag = True)
        sequences = [self.preprocess.get_wordidx_sequence(keywords)]

        max_len= 40
        padded_seqs = preprocessing.sequence.pad_sequences(sequences,maxlen=max_len, padding='post',value = 0)

        predict = self.model.predict(np.array( [padded_seqs[0] ] ))
        predict_class = tf.math.argmax(predict, axis = -1) #classindex 값이 넘어옴

        tags = []
        for tag_index in predict_class.numpy()[0]:
            if tag_index == 1 : 
                continue
            tags.append(self.index_to_ner[tag_index])

        if len(tags ) == 0: return None

        return tags# [BIO_tag]
