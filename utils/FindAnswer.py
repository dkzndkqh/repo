class FindAnswer:
    def __init__(self,db):
        self.db = db

    def _make_query(self, intent_name, ner_tags): #create sql query
        sql = 'select *from chatbot_train_data'

        if intent_name != None and ner_tags == None:
            sql = sql + "where intent = '{}' ".format(intent_name)  
        elif intent_name != None and ner_tags !=None:
            where = "where intent = '{}' ".format(intent_name)
            if(len(ner_tags) > 0):
                where += "and ("
                for ner in ner_tags:
                    where += "ner like '%{}%'or".format(ner)

                where = where [:-3]+ ")"
        sql = sql + where
        
        sql = sql + "order by rand() limit 1"
        return sql

    def search(self, intent_name, ner_tags):
        sql = _make_query(intent_name, ner_tags)
        answer = self.db.select_one(sql)

        if answer is None:
            sql = _make_query(intent_name, None)
            answer = self.db.select_one(sql)

        return (answer['answer'], answer['answer_image'])

    def tag_to_word(self,ner_predicts, answer): #짜장면을 주문할게요 --> B_FOOD -- > {B_FOOD } 주문 감사합니다. -- > 자장면 주문 감사합니다.
        for word, tag in ner_predicts:
            if tag == 'B_FOOD' or tag == 'B_DT' or tag == 'B_TI': #응답에 필요한 BIO 태그를 추가
                answer = answer.replace(tag, word)


        answer = answer.replace('{','')
        answer = answer.replace('}','')




        return answer
    