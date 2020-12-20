from elasticsearch import Elasticsearch
from datetime import datetime, date, timedelta
import os
import numpy as np
import re
import pandas as pd
import keras as keras
from keras import backend as K
from keras.layers import Layer
from keras_bert import Tokenizer, load_trained_model_from_checkpoint
from keras_radam import RAdam
import tensorflow as tf
import pymysql
import pickle
from konlpy.tag import *
import run_rank_answer
import json
from tqdm import tqdm

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# 객체 생성 및 엘라스틱 서치 접속
es = Elasticsearch(hosts="127.0.0.1", port=9200)
# Database connect
conn = pymysql.connect(host='localhost', user='root', password='1234', db='playball', charset='utf8')
# Model_path, Date type
model_file = "C:/Users/DeepLearning_5/PycharmProjects/Play-Ball/nonBert"
config = 'C:/Users/DeepLearning_5/PycharmProjects/Play-Ball/nonBert/bert_config.json'
ckpt = 'C:/Users/DeepLearning_5/PycharmProjects/Play-Ball/nonBert/model.ckpt-45305'
vocab = 'C:/Users/DeepLearning_5/PycharmProjects/Play-Ball/nonBert/Play_Ball_QA_Vocab'
today = date.today()
yesterday = date.today() - timedelta(1)
today = today.strftime("%Y-%m-%d")
yesterday = yesterday.strftime("%Y-%m-%d")
lastweek = date.today() - timedelta(7)
lastweek = lastweek.strftime("%Y-%m-%d")
team = ['두산', '키움', 'SK', 'LG', 'NC', 'KT', 'KIA', '삼성', '한화', '롯데']
rank = ['다승', '자책점', '평균자책', '탈삼진', '세이브', '타율', '타점', '홈런', '도루', 'WHIP', 'OPS', '투수 WAR', '타자 WAR']
twit = Twitter()

with open(vocab, 'rb') as vocabHandle:
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
    bert_qa_model = keras.models.Model(inputs, [outputs_start, outputs_end])
    optimizer_warmup = RAdam(learning_rate=1e-5, warmup_proportion=0.1, epsilon=1e-6, weight_decay=0.01)
    bert_qa_model.compile(
        optimizer=optimizer_warmup,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return bert_qa_model


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
    config,
    ckpt,
    seq_len=512, )
FurtherLearningModel = load_trained_model_from_checkpoint(
    config,
    ckpt,
    seq_len=512, )
bert_qa_model = get_bert_finetuning_model(model)
h5_path = "C:/Users/DeepLearning_5/PycharmProjects/Play-Ball/nonBert/nonModel.h5"
bert_qa_model.load_weights(h5_path)
graph = tf.get_default_graph()


def model_reinforce(data_x, data_y, answers_list):
    with graph.as_default():
        bert_qa_model.fit(data_x, data_y, epochs=10, verbose=2, batch_size=2, shuffle=True)
    bert_qa_model.save_weights('C:/Users/DeepLearning_5/PycharmProjects/Play-Ball/nonBert/testModel.h5')
    score = bert_model_verify()
    # 모델 저장 로드해서 성능 테스트
    with conn.cursor() as curs:
        sql = "select * from model_score;"
        curs.execute(sql)
        rows = curs.fetchall()
        for row in rows:
            f1_score = row[0]
    if (score > f1_score):
        os.remove('C:/Users/DeepLearning_5/PycharmProjects/Play-Ball/nonBert/nonModel.h5')
        os.renames('C:/Users/DeepLearning_5/PycharmProjects/Play-Ball/nonBert/testModel.h5',
                   'C:/Users/DeepLearning_5/PycharmProjects/Play-Ball/nonBert/nonModel.h5')
        # 새로운 모델로 변경, 변경 사항 DB수정
        for answer_no in answers_list:
            sql = "UPDATE playball.answers SET reinforce = '1' WHERE (answer_no =" + answer_no + ")"
            curs.execute(sql)
            sql = 'UPDATE playball.model_score SET f1_score = ' + str(
                score) + ' WHERE (f1_score = ' + f1_score + ');'
            curs.execute(sql)
        conn.commit()
        return str(score)
    else:
        os.remove('C:/Users/DeepLearning_5/PycharmProjects/Play-Ball/nonBert/testModel.h5')
        return "n"


# 문장 / 최대 길이 받아서 마침표 단위로 자름
def split_content_by_comma(content, length):
    splited_content = content.split('.')

    temp = ""
    before_temp = ""
    cnt = 0
    result = [""]

    while True:
        before_temp = temp
        temp = temp + splited_content[0] + '.'
        del splited_content[0]
        if splited_content == []:
            break
        if len(result[cnt] + temp) + 1 >= length:
            result.append(before_temp)
            cnt += 1
            temp = ""
            before_temp = temp

    del result[0]

    return result


def QA_predict(no, question, title, content, url):
    global tokenizer
    target = 512 - len(question)
    splited_content = split_content_by_comma(content, target)

    # 예비 답변 위치, 본문단어 토큰 인덱스들 저장용 리스트
    start_percent_list = []
    start_idx_list = []
    end_idx_list = []
    indexes_list = []

    # 나눠진 기사 본문 리스트 원소 별로 답변 위치 추출
    for num, cont in enumerate(splited_content):
        indices_token, segments_token = [], []

        ind, seg = tokenizer.encode(question, cont, max_len=512)
        indices_token.append(ind)
        segments_token.append(seg)
        indices = np.array(indices_token)
        segment = np.array(segments_token)
        question_input = [indices, segment]
        with graph.as_default():
            answer_start, answer_end = bert_qa_model.predict(question_input)
        indexes_candidate = tokenizer.encode(question, cont, max_len=512)[0]

        start_idx_candidate = np.argmax(answer_start, axis=1).item()
        start_percent_candidate = np.max(answer_start, axis=1).item()
        end_idx_candidate = np.argmax(answer_end, axis=1).item()

        # 현재 원소에서 도출한 답변 위치와 확률, 본문단어 토큰 인덱스 리스트에 저장
        start_idx_list.append(start_idx_candidate)
        end_idx_list.append(end_idx_candidate)
        start_percent_list.append(start_percent_candidate)
        indexes_list.append(indexes_candidate)

    # 확률이 최대인 위치에서 도출한 시작, 종료, 본문단어 토큰 인덱스 위치 사용
    ar = np.array(start_percent_list)
    idx = np.argmax(ar)
    start_idx = start_idx_list[idx]
    end_idx = end_idx_list[idx]
    indexes = indexes_list[idx]

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

    # 형태소 분석기 morphs -> 불용어 체크 -> 복구
    Morpheme_analysis = twit.morphs(answer)
    r = open('stopword.txt', mode='rt', encoding='utf-8')
    stop_words = r.read()
    stop_words = stop_words.split(' ')
    stop_words_result = []
    for w in Morpheme_analysis:
        if w not in stop_words:
            stop_words_result.append(w)
    answer = ''
    for i in stop_words_result:
        answer += i
    if len(answer) > 30:
        answer = answer[:30]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if answer == '[CLS]':
        answer = '정답을 찾지 못했습니다.'
    try:
        with conn.cursor() as curs:
            sql = "insert into playball.answers (board_no, answer, article_title, article_url, date) values(%s,%s,%s,%s,%s)"
            curs.execute(sql, (no, answer, title, url, now))
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
def article_search_date(no, question, date, question_team):
    # 질의문
    result_list = []
    # 기사 본문을 통한 검색
    if question_team == None:
        docs = es.search(index='article_data',
                         body={
                             "query": {
                                 "bool": {
                                     "must": [
                                         {"match": {"article_content": question}}
                                     ],
                                     "filter": [
                                         {"range": {"article_date": {"gte": date, "lte": "now"}}}
                                     ]
                                 }
                             },
                             "size": 3
                         })
    else:
        docs = es.search(index='article_data',
                         body={
                             "query": {
                                 "bool": {
                                     "must": [
                                         {"match": {"article_content": question}}
                                     ],
                                     "filter": [
                                         {"term": {"article_team": question_team}},
                                         {"range": {"article_date": {"gte": date, "lte": "now"}}}
                                     ]
                                 }
                             },
                             "size": 3
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
            QA_predict(no, question, result_list[i]['article_title'], result_list[i]['article_content'],
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
def article_search(no, question, question_team):
    # 질의문
    result_list = []
    if question_team == None:
        # 기사 본문을 통한 검색
        docs = es.search(index='article_data',
                         body={
                             "query": {
                                 "match": {
                                     "article_content": question
                                 }
                             },
                             "size": 3
                         })
    else:
        docs = es.search(index='article_data',
                         body={
                             "query": {
                                 "bool": {
                                     "must": [
                                         {"match": {"article_content": question}}
                                     ],
                                     "filter": [
                                         {"term": {"article_team": question_team}}
                                     ]
                                 }
                             },
                             "size": 3
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
        try:
            QA_predict(no, question, result_list[i]['article_title'], result_list[i]['article_content'],
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


# ==========================Main==============================
def run_bert_answer(no, question):
    question_team = None
    for i in team:
        if (i in question):
            question_team = i
    # Rank 조회
    try:
        for i in rank:
            if (i in question):
                if i == '자책점':
                    keyword = '평균자책'
                    rank_flag1 = True
                else:
                    keyword = i
                    rank_flag1 = True
        if re.search(r'\d{4}년', question):
            rank_flag3 = True
        if re.search(r'\d{1}위', question):
            rank_flag2 = True
        if (rank_flag1 == True and rank_flag2 == True and rank_flag3 == True):
            match = re.search(r'\d{1,2}위', question)
            match = (match.group(), '%d')[0]
            run_rank_answer.run_rank_answer(no, question, keyword, match[0])
    except:
        # 년,월,일 전부 있는 경우
        if re.search(r'\d{4}년\d{1,2}월\d{1,2}일', question):
            match = re.search(r'\d{4}년\d{1,2}월\d{1,2}일', question)
            date = datetime.strptime(match.group(), '%Y년%m월%d일').date()
            article_search_date(no, question, date, question_team)
        # 년, 월 만 있는 경우
        elif re.search(r'\d{4}년\d{1,2}월', question):
            match = re.search(r'\d{4}년\d{1,2}월', question)
            date = datetime.strptime(match.group(), '%Y년%m월').date()
            date = str(date)
            article_search_date(no, question, date, question_team)
        # 월, 일만 있는 경우(2020년으로 만듬) - 일만 검색 불가(최소 월, 일은 입력해야함)
        elif re.search(r'\d{1,2}월\d{1,2}일', question):
            match = re.search(r'\d{1,2}월\d{1,2}일', question)
            date = datetime.strptime(match.group(), '%m월%d일').date()
            date = str(date)
            date = date.replace('1900', '2020')
            article_search_date(no, question, date, question_team)
        # 년만 있는 경우(2020년으로 만듬)
        elif re.search(r'\d{4}년', question):
            match = re.search(r'\d{4}년', question)
            date = datetime.strptime(match.group(), '%Y년').date()
            date = str(date)
            article_search_date(no, question, date, question_team)
        # 월만 있는 경우(2020년으로 만듬)
        elif re.search(r'\d{1,2}월', question):
            match = re.search(r'\d{1,2}월', question)
            date = datetime.strptime(match.group(), '%m월').date()
            date = str(date)
            date = date.replace('1900', '2020')
            article_search_date(no, question, date, question_team)
        elif '오늘' in question:
            date = str(today)
            article_search_date(no, question, date, question_team)
        elif '어제' in question:
            date = str(yesterday)
            article_search_date(no, question, date, question_team)
        elif '최근' in question:
            date = str(lastweek)
            article_search_date(no, question, date, question_team)
        else:
            article_search(no, question, question_team)


# model_reinforce_testing
def bert_model_verify():
    with graph.as_default():
        verify_model = get_bert_finetuning_model(FurtherLearningModel)
    h5_path = 'C:/Users/DeepLearning_5/PycharmProjects/Play-Ball/nonBert/testModel.h5'
    with graph.as_default():
        verify_model.load_weights(h5_path)

    def korquad_json_to_dataframe_dev(input_file_path, record_path=['data', 'paragraphs', 'qas', 'answers'],
                                      verbose=1):
        if verbose:
            pass
        file = json.loads(open(input_file_path).read())
        if verbose:
            pass
        js = pd.io.json.json_normalize(file, record_path)
        m = pd.io.json.json_normalize(file, record_path[:-1])
        r = pd.io.json.json_normalize(file, record_path[:-2])

        idx = np.repeat(r['context'].values, r.qas.str.len())
        m['context'] = idx
        main = m[['id', 'question', 'context', 'answers']].set_index('id').reset_index()
        main['c_id'] = main['context'].factorize()[0]
        if verbose:
            print("shape of the dataframe is {}".format(main.shape))
            print("Done")
        return main

    input_file_path = 'C:/Users/DeepLearning_5/PycharmProjects/Play-Ball/nonBert/Dev.json'
    record_path = ['data', 'paragraphs', 'qas', 'answers']
    dev = korquad_json_to_dataframe_dev(input_file_path=input_file_path, record_path=record_path)

    dev['answer_len'] = dev['answers'].map(lambda x: len(x))

    def get_text(text_len, answers):
        texts = []
        for i in range(text_len):
            texts.append(answers[i]['text'])
        return texts

    dev['texts'] = dev.apply(lambda x: get_text(x['answer_len'], x['answers']), axis=1)
    TEXT_COLUMN = 'texts'

    DATA_COLUMN = "context"
    QUESTION_COLUMN = "question"
    SEQ_LEN = 512

    def convert_data(data_df):
        global tokenizer
        indices, segments, target_start, target_end = [], [], [], []

        for i in tqdm(range(len(data_df))):
            que, _ = tokenizer.encode(data_df[QUESTION_COLUMN][i])
            doc, _ = tokenizer.encode(data_df[DATA_COLUMN][i])
            doc.pop(0)

            que_len = len(que)
            doc_len = len(doc)

            if que_len > 64:
                que = que[:63]
                que.append(102)

            if len(que + doc) > SEQ_LEN:
                while len(que + doc) != SEQ_LEN:
                    doc.pop(-1)

                doc.pop(-1)
                doc.append(102)

            segment = [0] * len(que) + [1] * len(doc) + [0] * (SEQ_LEN - len(que) - len(doc))
            if len(que + doc) <= SEQ_LEN:
                while len(que + doc) != SEQ_LEN:
                    doc.append(0)

            ids = que + doc

            texts = data_df[TEXT_COLUMN][i]
            for text_element in texts:
                text = tokenizer.encode(text_element)[0]

                text_slide_len = len(text[1:-1])
                for j in range(0, (len(doc))):
                    exist_flag = 0
                    if text[1:-1] == doc[j:j + text_slide_len]:
                        ans_start = j + len(que)
                        ans_end = j + text_slide_len - 1 + len(que)
                        exist_flag = 1
                        break

                if exist_flag == 0:
                    ans_start = SEQ_LEN
                    ans_end = SEQ_LEN

            indices.append(ids)
            segments.append(segment)
            target_start.append(ans_start)
            target_end.append(ans_end)

        indices_x = np.array(indices)
        segments = np.array(segments)
        target_start = np.array(target_start)
        target_end = np.array(target_end)

        del_list = np.where(target_start != SEQ_LEN)[0]
        not_del_list = np.where(target_start == SEQ_LEN)[0]
        indices_x = indices_x[del_list]
        segments = segments[del_list]

        target_start = target_start[del_list]
        target_end = target_end[del_list]

        return [indices_x, segments], del_list

    def load_data(pandas_dataframe):
        data_df = pandas_dataframe
        data_df[DATA_COLUMN] = data_df[DATA_COLUMN].astype(str)
        data_df[QUESTION_COLUMN] = data_df[QUESTION_COLUMN].astype(str)
        data_x, data_y, del_list = convert_data(data_df)

        return data_x, data_y, del_list

    dev_bert_input = convert_data(dev)
    dev_bert_input, del_list = dev_bert_input[0], dev_bert_input[1]
    dev = dev.iloc[del_list]
    dev = dev.reset_index(drop=True)

    indexes = dev_bert_input[0]

    with graph.as_default():
        bert_predictions = verify_model.predict(dev_bert_input, verbose=1, batch_size=128)

    start_indexes = np.argmax(bert_predictions[0], axis=-1)
    end_indexes = np.argmax(bert_predictions[1], axis=-1)
    not_del_list = np.where(start_indexes <= end_indexes)[0]

    start_indexes = start_indexes[not_del_list]
    end_indexes = end_indexes[not_del_list]
    indexes = indexes[not_del_list]

    # dev 데이터셋 재조정
    dev = dev.iloc[not_del_list].reset_index(drop=True)
    # length : dev 데이터의 길이
    length = len(dev)

    sentences = []

    untokenized = []

    for j in range(len(start_indexes)):
        sentence = []
        for i in range(start_indexes[j], end_indexes[j] + 1):
            token_based_word = reverse_token_dict[indexes[j][i]]
            sentence.append(token_based_word)
            # 문장이 토큰화된 단어 하나 하나를 sentence에 저장

        sentence_string = ""

        for word_piece in sentence:

            if word_piece.startswith("##"):
                word_piece = word_piece.replace("##", "")
                # ## 제거
            else:
                word_piece = " " + word_piece
            sentence_string += word_piece
        if sentence_string.startswith(" "):
            sentence_string = "" + sentence_string[1:]
        untokenized.append(sentence_string)
        sentences.append(sentence)

    dev_answers = []
    for i in range(length):
        dev_answer = []
        texts_dict = dev['answers'][i]

        for j in range(len(texts_dict)):
            dev_answer.append(texts_dict[j]['text'])
        dev_answers.append(dev_answer)

    dev_tokens = []
    for i in dev_answers:
        dev_tokened = []
        for j in i:
            temp_token = tokenizer.tokenize(j)
            # 정답을 토큰화
            temp_token.pop(0)
            # [CLS] 제거
            temp_token.pop(-1)
            # [SEP] 제거
            dev_tokened.append(temp_token)
        dev_tokens.append(dev_tokened)

    dev_answer_lists = []
    for dev_answers in dev_tokens:
        dev_answer_list = []
        for dev_answer in dev_answers:
            dev_answer_string = " ".join(dev_answer)
            dev_answer_list.append(dev_answer_string)
        dev_answer_lists.append(dev_answer_list)

    dev_strings_end = []
    for dev_strings in dev_answer_lists:
        dev_strings_processed = []
        for dev_string in dev_strings:
            # 문장 합치기
            dev_string = dev_string.replace(" ##", "")
            dev_strings_processed.append(dev_string)
        dev_strings_end.append(dev_strings_processed)
    dev_answers = dev_strings_end

    from collections import Counter
    import string

    # F1 SCORE / EXACT MATCH
    def normalize_answer(s):

        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def f1_score(prediction, ground_truth):
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)

    f1_score_result = 0
    for i in range(len(untokenized)):
        f1 = metric_max_over_ground_truths(f1_score, untokenized[i], dev_answers[i])
        f1_score_result += f1
    return f1_score_result / length
