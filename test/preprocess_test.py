from utils.Preprocess import Preprocess

sent = "내일 오전10시에 짬뽕 주문하고 싶어ㅋㅋ"
p = Preprocess(word2index_dic ='./train_tools/Dict/chatbot_dict.bin',userdic='./utils/user_dic.tsv')

pos = p.pos(sent)

print(pos)

keywords = p.get_keywords(pos, without_tag=False)

print(keywords)

seq = p.get_wordidx_sequence(keywords)

print(seq)