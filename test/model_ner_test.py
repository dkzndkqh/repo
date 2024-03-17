import tensorflow as tf
from tensorflow.keras import preprocessing # padding
from tensorflow.keras.models import Model, load_model
from modles.ner.ner_Model import NerModel
from utils.Preprocess import Preprocess
import os

this_program_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(this_program_directory)

p = Preprocess(word2index_dic='../train_tools/Dict/chatbot_dict.bin', userdic='../utils/user_dic.tsv')

model = NerModel('../modles/ner/ner_model.h5', p)

query = '오늘 오전 13시 2분에 탕수육 주문 하고 싶어요'

answer = model.predict(query)
answer_Bio = model.predict_tag(query)

print(answer)
print(answer_Bio)
