import threading
import json

from config.DatabaseConfig import *
from utils.Database import Database
from utils.BotServer import BotServer
from utils.Preprocess import Preprocess
from modles.intent.intentModel import IntentModel
from modles.ner.ner_Model import NerModel
from utils.FindAnswer import FindAnswer

import os 

this_program_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(this_program_directory)

p = Preprocess(word2index_dic='./train_tools/Dict/chatbot_dict.bin', userdic='./utils/user_dic.tsv')

intent = IntentModel(model_name = './modles/intent/intent_model.h5', preprocess= p)
ner = NerModel(model_name = './modles/ner/ner_model.h5', preprocess= p)

def to_client(conn, addr, params): #thread function
    db = params['db']
    
    try:
        db.connect()

        read = conn.recv(2048)
        print('=' * 20)
        print ('Connection from {0}'.format(str(addr)))

        if read is None or not read: #disconnect from client or Error
            print('Disconnection from client')
            exit(0) # terminated Thread


        recv_json_data = json.loads(read.decode())
        print ('-' * 20)
        print('Recved data : {0}'.format(recv_json_data))
        query = recv_json_data['Query']

        intent_predict = intent.predict_class(query)
        intent_name = intent.labels[intent_predict]
        print('Intent Label : {0}'.format(intent_name))

        ner_predicts = ner.predict(query)
        ner_tags = ner.predict_tag(query)

        print('Ner Predict : {0}'.format(ner_predicts))

        try:
            f = FindAnswer(db)
            answer_text, answer_image = f.search(intent_name, ner_tags)
            answer = f.tag_to_word(ner_predicts, answer_text)

        except:
            answer = '죄송합니다. 무슨말인지 모르겠어요'

        send_json_data_str = {
            'Query': query,
            'Answer' : answer,
            'AnswerImageUrl' : answer_image,
            'Intent' : intent_name,
            'NER' : str(ner_predicts)
            }
        message = json.dumps(send_json_data_str)
        conn.send(message)

    except Exception as ex:
        print(ex)

    finally:
        if db is not None:
            db.close()
        conn.close()
if __name__ == '__main__':
    db = Database(host = DB_HOST, user = DB_USER, password = DB_PASSWORD, db_name = DB_NAME)

    print('DB Connection!')

    port = 5050

    listen = 100

    bot = BotServer(port, listen)
    bot.create_socket()

    print('Bot Server Start!')

    while True:
        conn, addr = bot.ready_for_client()
        params = {"db" : db }

        #to_client()
        client_thread = threading.Thread(target = to_client,  #function name
                                         args = (conn, addr, params)) # parameters

        client_thread.start()


