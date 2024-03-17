import pymysql
import pymysql.cursors
import logging


class Database:
    def __init__(self, host, user, password, db_name, charset = 'utf8' ):
        self.host = host
        self.user = user
        self.password= password
        self.charset = charset
        self.db_name = db_name
        self.conn = None

 
        
    def connect(self):
        if self.conn != None:
            return 
        self.conn = pymysql.connect(host = self.host, user = self.user,
                                    password = self.password, db = self.db_name,
                                    charset = self.charset)
    def close(self):
        if self.conn is None:
            return 
        if not self.conn.open:
            self.conn = None
            return 

        self.conn.close()
        self.conn = None


    def execute(self, sql):

        last_row_id = -1
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(sql)
            self.conn.commit()
            last_row_id = cursor.lastrowid
            logging.debug('excute last row id : {0}'.format(last_row_id))
        except Exception as ex:
            logging.debug(ex)

        finally:
            return last_row_id


    def select_one(self, sql): #SELECT --> ONE
        result = None

        try:
            with self.conn.cursor() as cursor:
                cursor.execute(sql)
                result = cursor.fetchon()

        except Exception as ex:
            logging.debug(ex)
        finally:
            return result

    def select_all(self, sql): #SELECT --> ALL
        result = None
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(sql)
                result = cursor.fetchall()

        except Exception as ex:
            logging.debug(ex)
        finally:
            return result