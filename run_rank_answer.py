import pymysql
import re

# no, question, keyword, match

def run_rank_answer(no, question, keyword, match):
    try:
        # mysql 연동 및 select
        conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='1234', db='playball', charset='utf8')
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        sql = 'SELECT * FROM playball.rank where title = "' + keyword + '" and rank_no = ' + match
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            answer = row['name']
            title = row['info']
        #insert
        sql = 'insert into playball.answers (board_no, answer, article_title) values(%s,%s,%s)'
        cursor.execute(sql, (no, answer, title))
        sql = "UPDATE playball.board SET answer = \'답변완료\' WHERE (No =\'" + str(no) + "\')"
        cursor.execute(sql)
        conn.commit()
    except:
        print("예외처리")