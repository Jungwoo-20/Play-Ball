import threading
import pymysql
import run_bert_answer


def detec_bot():
    try:
        conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='1234', db='playball', charset='utf8')
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        sql = 'SELECT * FROM playball.board where answer is null order by date asc limit 1;'
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            question = row['question']
            no = row['No']
        run_bert_answer.run_bert_answer(no, question)
    except:
        pass
    finally:
        cursor.close()
        threading.Timer(7, detec_bot).start()