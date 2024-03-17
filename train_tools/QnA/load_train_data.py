import pymysql
import openpyxl
from config.DatabaseConfig import * # DB 접속 정보 불러오기
# 학습 데이터 초기화
def all_clear_train_data(db):
 # 기존 학습 데이터 삭제
    sql = '''
    delete from chatbot_train_data
    '''
    with db.cursor() as cursor:
        cursor.execute(sql)

        # auto increment 초기화
    sql = '''
    ALTER TABLE chatbot_train_data AUTO_INCREMENT=1
    '''
    with db.cursor() as cursor:
        cursor.execute(sql)

def inser_data(db, xls_row):
    intent, ner, query, answer, answer_img_url = xls_row

    sql = '''
    INSERT chatbot_train_data(intent, ner, query, answer, answer_image) 
    values(
    '%s', '%s', '%s', '%s', '%s' )
    ''' % (intent.value, ner.value, query.value, answer.value, answer_img_url.value)
    sql = sql.replace("'None'", "null")

    with db.cursor() as cursor:
        cursor.execute(sql)
        print('{} 저장'.format(query.value))
        db.commit()

train_file = './train_tools/QnA/train_data.xlsx'

db = None

try:
    db = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        passwd=DB_PASSWORD,
        db=DB_NAME,
        charset='utf8'
    )

    all_clear_train(db)

    wb= openpyxl.load_workbook(train_file)
    sheet = wb['Sheet1']


    for row in sheet.iter_rows(min_row=2):
        insert_data(db, row)

    wb.close()

except Exception as e:
    print(e)


finally:
    if db is not None:
        db.close()