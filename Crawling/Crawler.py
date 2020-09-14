from selenium import webdriver
import numpy as np
from collections import OrderedDict
import json
from datetime import date, timedelta
from elasticsearch import Elasticsearch
import threading
from datetime import datetime

# from panda._libs import json
# 객체 생성 및 엘라스틱 서치 접속
es = Elasticsearch(hosts="127.0.0.1", port=9200, timeout=30, max_retries=10, retry_on_timeout=True)
# chromedriver연결
driver = webdriver.Chrome('./chromedriver')


def Crawler():
    # 일자별 수집
    today = date.today()
    tomorrow = date.today() + timedelta(1)
    today = today.strftime("%Y%m%d")
    tomorrow = tomorrow.strftime("%Y%m%d")
    print('오늘 날짜 : ' + str(today))
    # file open
    f = open(
        "C:/Users/DeepLearning_5/PycharmProjects/Play-Ball/Crawling/CrawlingData_TEXT/" + today + ".txt",
        'w')

    teamList = np.array(['두산', '키움', 'SK', 'LG', 'NC', 'KT', 'KIA', '삼성', '한화', '롯데'])
    article_data = OrderedDict()

    def switch(x):
        return {'두산': 'OB', '키움': 'WO', 'SK': 'SK', 'LG': 'LG', 'NC': 'NC', 'KT': 'KT', 'KIA': 'HT', '삼성': 'SS',
                '한화': 'HH', '롯데': 'LT', }.get(x, '오류')

    for team in teamList:
        cntTeam = switch(team)
        for i in range(int(today), int(tomorrow)):
            flag = True
            page_flag = True
            for j in range(1, 20):
                if page_flag == False:
                    page_flag = True
                    break
                url = 'https://sports.news.naver.com/kbaseball/news/index.nhn?isphoto=N&type=team&team=' + cntTeam + '&date=' + str(
                    i) + '&page=' + str(j)
                driver.get(url)
                if flag:
                    for k in range(1, 21):
                        try:
                            current_page = driver.find_element_by_css_selector('#_pageList > strong').text
                            if j == int(current_page):
                                title_url = driver.find_element_by_css_selector(
                                    '#_newsList > ul > li:nth-child(' + str(k) + ') > div > a').get_attribute('href')
                                title_url = title_url + '!' + team + '\n'
                                f.write(title_url)
                            else:
                                flag = False
                                break
                        except Exception as ex:
                            try:
                                title_url = driver.find_element_by_css_selector(
                                    '#_newsList > ul > li:nth-child(' + str(k) + ') > div > a').get_attribute('href')
                                title_url = title_url + '!' + team + '\n'
                                f.write(title_url)
                                if k == 20:
                                    page_flag = False
                            except Exception as ex1:
                                flag = False
                                break
                else:
                    break

    f.close()

    def get_article_info(line):
        url = line.split('!')[0]
        team = line.split('!')[1][:-1]
        driver.get(url)
        flag = True
        try:
            article_title = driver.find_element_by_class_name('title').text
            article_date = driver.find_element_by_css_selector(
                '#content > div > div.content > div > div.news_headline > div > span:nth-child(1)').text
            # date format
            article_date = article_date[5:15].replace('.', '-')
        except Exception as ex:
            flag = False
            return {"flag": flag}
        try:
            article_img = driver.find_element_by_css_selector('#newsEndContents > span > img').get_attribute('src')
        except Exception as ex:
            article_img = None
        finally:
            article_content = driver.find_element_by_id('newsEndContents').text
            article_content = article_content.replace('\n', ' ')
            return {"article_title": article_title, "article_date": article_date, "article_img": article_img,
                    "article_content": article_content, "article_url": url, "article_team": team, "flag": flag}

    resultList = []

    f = open(
        "C:/Users/DeepLearning_5/PycharmProjects/Play-Ball/Crawling/CrawlingData_TEXT/" + today + ".txt",
        'r')
    line = f.readline()
    while True:
        line = f.readline()
        if not line: break
        article_info = get_article_info(line)
        flag = article_info['flag']
        if flag == True:
            result = {
                "article_title": article_info['article_title'],
                "article_date": article_info['article_date'],
                "article_img": article_info['article_img'],
                "article_content": article_info['article_content'],
                "article_url": article_info['article_url'],
                "article_team": article_info['article_team']
            }
            resultList.append(result)

    with open(
            'C:/Users/DeepLearning_5/PycharmProjects/Play-Ball/Crawling/CrawlingData_JSON/' + today + '.json',
            'w', encoding="utf-8") as make_file:
        json.dump(resultList, make_file, ensure_ascii=False, indent="\t")
    f.close()

    # 데이터 삽입
    def insertData():
        with open(
                "C:/Users/DeepLearning_5/PycharmProjects/Play-Ball/Crawling/CrawlingData_JSON/" + today + ".json",
                "r", encoding="utf-8") as fjson:
            data = json.loads(fjson.read())
            for n, i in enumerate(data):
                doc = {"article_title": i["article_title"],
                       "article_date": i["article_date"],
                       "article_img": i["article_img"],
                       "article_content": i["article_content"],
                       "article_url": i["article_url"],
                       "article_team": i["article_team"]}
                es.index(index="article_data", id=i["article_url"], doc_type="_doc", body=doc)

    insertData()
    print(datetime.today().strftime("%Y/%m/%d %H:%M:%S"))
    threading.Timer(1800, Crawler).start()


Crawler()
