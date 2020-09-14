from elasticsearch import Elasticsearch
from datetime import datetime, date, timedelta
import os
import numpy as np
import re
import keras as keras
from keras import backend as K
from keras.layers import Layer
from keras_bert import Tokenizer, load_trained_model_from_checkpoint
from keras_radam import RAdam
import tensorflow as tf
import pymysql
import pickle

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# Elasticsearch
# 객체 생성 및 엘라스틱 서치 접속
es = Elasticsearch(hosts="127.0.0.1", port=9200)
# Database connect
conn = pymysql.connect(host='localhost', user='root', password='1234', db='playball', charset='utf8')
# Model_path, Date type
model_path = "C:/Users/DeepLearning_5/PycharmProjects/Play-Ball/Bert"
config_path = os.path.join(model_path, 'bert_config.json')
check_path = os.path.join(model_path, 'model.ckpt-45305')
vocab_path = os.path.join(model_path, 'Play_Ball_QA_Vocab')
today = date.today()
yesterday = date.today() - timedelta(1)
today = today.strftime("%Y-%m-%d")
yesterday = yesterday.strftime("%Y-%m-%d")
lastweek = date.today() - timedelta(7)
lastweek = lastweek.strftime("%Y-%m-%d")
team = ['두산', '키움', 'SK', 'LG', 'NC', 'KT', 'KIA', '삼성', '한화', '롯데']

with open(vocab_path, 'rb') as vocabHandle:
    Token = pickle.load(vocabHandle)


class inherit_Tokenizer(Tokenizer):
    def _tokenize(self, text):
        if not self._cased:
            text = text.lower()
        spaced = ''
        for ch in text:
            if self._is_punctuation(ch) or self._is_cjk_character(ch):
                spaced += ' ' + ch + ' '
            elif self._is_space(ch):
                spaced += ' '
            elif ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                continue
            else:
                spaced += ch
        tokens = []
        for word in spaced.strip().split():
            tokens += self._word_piece_tokenize(word)
        return tokens


def get_bert_finetuning_model(model):
    inputs = model.inputs[:2]
    bert_transformer = model.layers[-1].output
    x = NonMasking()(bert_transformer)
    outputs_start, outputs_end = answer_predict(512)(x)
    bert_model = keras.models.Model(inputs, [outputs_start, outputs_end])
    optimizer_warmup = RAdam(learning_rate=1e-5, warmup_proportion=0.1, epsilon=1e-6, weight_decay=0.01)
    bert_model.compile(
        optimizer=optimizer_warmup,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return bert_model


# start, end idx predict
def bert_gelu(x):
    cdf = 0.5 * (1.0 + K.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * K.pow(x, 3)))))
    return x * cdf


class NonMasking(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(NonMasking, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape = input_shape

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        return x

    def get_output_shape_for(self, input_shape):
        return input_shape


class answer_predict(Layer):
    def __init__(self, seq_len, **kwargs):
        self.seq_len = seq_len
        super(answer_predict, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='kernel',
                                 shape=(input_shape[2], 2),
                                 initializer='uniform',
                                 trainable=True)
        super(answer_predict, self).build(input_shape)

    def call(self, x):
        x = K.reshape(x, shape=(-1, self.seq_len, K.shape(x)[2]))
        x = K.dot(x, self.W)
        x = K.permute_dimensions(x, (2, 0, 1))
        self.start_logits, self.end_logits = x[0], x[1]
        self.start_logits = bert_gelu(self.start_logits)
        self.end_logits = bert_gelu(self.end_logits)
        self.start_logits = K.softmax(self.start_logits, axis=-1)
        self.end_logits = K.softmax(self.end_logits, axis=-1)
        return [self.start_logits, self.end_logits]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.seq_len), (input_shape[0], self.seq_len)]


# voacb
tokenizer = inherit_Tokenizer(Token)
reverse_token_dict = {v: k for k, v in Token.items()}
# bert_model load
model = load_trained_model_from_checkpoint(
    config_path,
    check_path,
    seq_len=512, )
bert_model = get_bert_finetuning_model(model)
h5_path = "C:/Users/DeepLearning_5/PycharmProjects/Play-Ball/Bert/Play_Ball_QA_Model.h5"
bert_model.load_weights(h5_path)
graph = tf.get_default_graph()

def model_reinforce(data_x, data_y):
    with graph.as_default():
        bert_model.fit(data_x, data_y, epochs=10, verbose=2, batch_size=2, shuffle=True)
    bert_model.save_weights('C:/Users/DeepLearning_5/PycharmProjects/Play-Ball/Bert/Play_Ball_QA_Model.h5')


def predict_letter(no, question, title, content, url):
    global tokenizer
    indices_token, segments_token = [], []
    ind, seg = tokenizer.encode(question, content, max_len=512)
    indices_token.append(ind)
    segments_token.append(seg)
    indices = np.array(indices_token)
    segment = np.array(segments_token)
    question_input = [indices, segment]
    with graph.as_default():
        answer_start, answer_end = bert_model.predict(question_input)
    indexes = tokenizer.encode(question, content, max_len=512)[0]
    start_idx = np.argmax(answer_start, axis=1).item()
    end_idx = np.argmax(answer_end, axis=1).item()
    unrefined_result = []
    for i in range(start_idx, end_idx + 1):
        token_based_word = reverse_token_dict[indexes[i]]
        unrefined_result.append(token_based_word)
    answer = ''
    for word_piece in unrefined_result:
        if word_piece.startswith('##'):
            answer += word_piece.replace('##', '')
        else:
            answer += '' + word_piece
    if len(answer) > 30:
        answer = answer[:30]
    if answer == '[CLS]' or answer == '':
        answer = '정답을 찾지 못했습니다. 기사를 참고해 주세요.'
    try:
        with conn.cursor() as curs:
            sql = 'insert into playball.answers (board_no, answer, article_title, article_url) values(%s,%s,%s,%s)'
            curs.execute(sql, (no, answer, title, url))
            # board table answer update
            try:
                sql = "UPDATE playball.board SET answer = \'답변완료\' WHERE (No =\'" + str(no) + "\')"
                curs.execute(sql)
                conn.commit()
            except:
                pass
        conn.commit()
    except:
        conn.close()


# 데이터 검색(날짜 입력이 들어오는 경우)
def searchDateAPI(no, question, date, question_team):
    # 질의문
    result_list = []
    # 기사 본문을 통한 검색
    if question_team == None:
        docs = es.search(index='article_data',
                         body={
                             "query": {
                                 "bool": {
                                     "must": [
                                         {"match": {"article_content.nori_discard": question}}
                                     ],
                                     "filter": [
                                         {"range": {"article_date": {"gte": date, "lte": "now"}}}
                                     ]
                                 }
                             }
                         })
    else:
        docs = es.search(index='article_data',
                         body={
                             "query": {
                                 "bool": {
                                     "must": [
                                         {"match": {"article_content.nori_discard": question}}
                                     ],
                                     "filter": [
                                         {"term": {"article_team": question_team}},
                                         {"range": {"article_date": {"gte": date, "lte": "now"}}}
                                     ]
                                 }
                             }
                         })

    # 결과 추출
    for result in docs['hits']['hits']:
        idx = result['_source']
        result = {
            'article_title': idx['article_title'],
            'article_date': idx['article_date'],
            'article_img': idx['article_img'],
            'article_content': idx['article_content'],
            'article_url': idx['article_url'],
            'article_team': idx['article_team'],
        }
        result_list.append(result)
    # 문서를 찾지 못한 경우 board에 답변완료 기입, answer에 '문서를 찾지 못했습니다.' 만 기록
    for i in range(3):
        try:
            predict_letter(no, question, result_list[i]['article_title'], result_list[i]['article_content'],
                           result_list[i]['article_url'])
        except:
            with conn.cursor() as curs:
                answer = '정답을 찾지 못했습니다.'
                title = ''
                url = ''
                sql = 'insert into playball.answers (board_no, answer, article_title, article_url) values(%s,%s,%s,%s)'
                curs.execute(sql, (no, answer, title, url))
                # board table answer update
                try:
                    sql = "UPDATE playball.board SET answer = \'답변완료\' WHERE (No =\'" + str(no) + "\')"
                    curs.execute(sql)
                    conn.commit()
                except:
                    pass
            conn.commit()


# 데이터 검색(기본 검색)
def searchAPI(no, question, question_team):
    # 질의문
    result_list = []
    if question_team == None:
        # 기사 본문을 통한 검색
        docs = es.search(index='article_data',
                         body={
                             "query": {
                                 "match": {
                                     "article_content.nori_discard": question
                                 }
                             }
                         })
    else:
        docs = es.search(index='article_data',
                         body={
                             "query": {
                                 "bool": {
                                     "must": [
                                         {"match": {"article_content.nori_discard": question}}
                                     ],
                                     "filter": [
                                         {"term": {"article_team": question_team}}
                                     ]
                                 }
                             }
                         })
    # 결과 추출
    for result in docs['hits']['hits']:
        idx = result['_source']
        result = {
            'article_title': idx['article_title'],
            'article_date': idx['article_date'],
            'article_img': idx['article_img'],
            'article_content': idx['article_content'],
            'article_url': idx['article_url'],
            'article_team': idx['article_team'], }
        result_list.append(result)

    for i in range(3):
        predict_letter(no, question, result_list[i]['article_title'], result_list[i]['article_content'],
                       result_list[i]['article_url'])

# ==========================Main==============================
def run_bert_answer(no, question):
    question_team = None
    for i in team:
        if (i in question):
            question_team = i
    # 년,월,일 전부 있는 경우
    if re.search(r'\d{4}년\d{1,2}월\d{1,2}일', question):
        match = re.search(r'\d{4}년\d{1,2}월\d{1,2}일', question)
        date = datetime.strptime(match.group(), '%Y년%m월%d일').date()
        searchDateAPI(no, question, date, question_team)
    # 년, 월 만 있는 경우
    elif re.search(r'\d{4}년\d{1,2}월', question):
        match = re.search(r'\d{4}년\d{1,2}월', question)
        date = datetime.strptime(match.group(), '%Y년%m월').date()
        date = str(date)
        searchDateAPI(no, question, date, question_team)
    # 월, 일만 있는 경우(2020년으로 만듬) - 일만 검색 불가(최소 월, 일은 입력해야함)
    elif re.search(r'\d{1,2}월\d{1,2}일', question):
        match = re.search(r'\d{1,2}월\d{1,2}일', question)
        date = datetime.strptime(match.group(), '%m월%d일').date()
        date = str(date)
        date = date.replace('1900', '2020')
        searchDateAPI(no, question, date, question_team)
    # 년만 있는 경우(2020년으로 만듬)
    elif re.search(r'\d{4}년', question):
        match = re.search(r'\d{4}년', question)
        date = datetime.strptime(match.group(), '%Y년').date()
        date = str(date)
        searchDateAPI(no, question, date, question_team)
    # 월만 있는 경우(2020년으로 만듬)
    elif re.search(r'\d{1,2}월', question):
        match = re.search(r'\d{1,2}월', question)
        date = datetime.strptime(match.group(), '%m월').date()
        date = str(date)
        date = date.replace('1900', '2020')
        searchDateAPI(no, question, date, question_team)
    elif '오늘' in question:
        date = str(today)
        searchDateAPI(no, question, date, question_team)
    elif '어제' in question:
        date = str(yesterday)
        searchDateAPI(no, question, date, question_team)
    elif '최근' in question:
        date = str(lastweek)
        searchDateAPI(no, question, date, question_team)
    else:
        date = str(lastweek)
        searchDateAPI(no, question, date, question_team)

# Reference
# https://www.elastic.co/guide/en/elasticsearch/reference/current/search-search.html
# https://github.com/kimwoonggon
# https://github.com/CyberZHG/keras-bert
# https://github.com/google-research/bert