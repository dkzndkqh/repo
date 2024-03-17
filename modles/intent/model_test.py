import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import preprocessing
from modles.intent.intentModel import IntentModel
from utils.Preprocess import Preprocess
import os

this_program_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(this_program_directory)

p = Preprocess(word2index_dic='../../train_tools/Dict/chatbot_dict.bin', userdic='../../utils/user_dic.tsv')

model = IntentModel('./intent_model.h5', p)

query = [ "오늘 탕수육 주문 가능한가요?", "안녕하세요?" ]

acc, answer = [model.predict_class(queries) for queries in query]

print(acc , answer)
