import pickle
from utils.Preprocess import Preprocess
import konlpy

f = open('./train_tools/Dict/chatbot_dict.bin', 'rb')

word_index = pickle.load(f)
f.close()

sent = '내일 오전 10시에 탕수육 주문하고 싶어 ㅋㅋ'

p = Preprocess(userdic='./utils/user_dic.tsv')
pos = p.pos(sent)

keywords = p.get_keywords(pos, without_tag = True)
for word in keywords:
    try:
        print('{0} : {1}'.format(word, word_index[word]))

    except KeyError:
        print('{0} : {1}'.format(word, word_index['OOV']))