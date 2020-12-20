from collections import OrderedDict
from run_bert_answer import *
import time
import numpy as np

file_data = OrderedDict()
resultList = []

model_flag = True


def get_content_by_URL(url):
    es = Elasticsearch(hosts="127.0.0.1", port=9200)

    docs = es.search(index='article_data',
                     body={
                         "query": {
                             "match": {
                                 "article_url": url
                             }
                         }
                     })

    result_list = []

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

    return result_list[0]['article_content']


def get_QA_info(line):
    question = line.split('§')[2]
    answer = line.split('§')[3]
    context = line.split('§')[1]
    title = line.split('§')[0]
    answer_start = line.split('§')[4]
    answer_start = answer_start.replace('\n', '')
    return {"question": question, "answer": answer, "context": context, "title": title, "answer_start": answer_start}

def create_json(file_name,answers_list):
    f1score=""
    model_flag = True
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    timestamp = timestamp + '.json'
    f = open(file_name, 'r', -1, encoding='UTF-8')
    i = 0
    flag = True
    while True:
        line = f.readline()
        if not line: break
        QA_info = get_QA_info(line)
        # return 값을들 나누어 처리
        file_data["data"] = [{"paragraphs": [
            {"qas": [{"answers": [{"text": QA_info['answer'], 'answer_start': QA_info['answer_start']}], 'id': i,
                      'question': QA_info['question']}],
             'context': QA_info['context']}],
            'title': QA_info["title"]}]

        if flag == True:
            result = {
                "version": "KorQuAD_v1.0_dev",
                "data": file_data["data"]
            }
            flag = False
            resultList.append(result)
        else:
            result = {
                "data": file_data["data"]
            }
            resultList.append(result)
        i += 1

    with open('C:/Users/DeepLearning_5/PycharmProjects/Play-Ball/ReinforcementLearning/ReinforcementFile/' + timestamp,
              'w', encoding="utf-8") as make_file:
        json.dump(resultList, make_file, ensure_ascii=False, indent="\t")
    try:
        reinforce_path = 'C:/Users\DeepLearning_5/PycharmProjects/Play-Ball/ReinforcementLearning/ReinforcementFile/' + timestamp
        record_path = ['data', 'paragraphs', 'qas', 'answers']
        reinforce_train_file = korquad_json_to_dataframe_train(input_file_path=reinforce_path, record_path=record_path)
        data_x, data_y = load_data(reinforce_train_file)
        f1score = model_reinforce(data_x, data_y,answers_list)
    except Exception as e:
        f1score="n"
    return f1score

def korquad_json_to_dataframe_train(input_file_path, record_path):
    with open(input_file_path, "r", encoding='utf-8') as fjson:
        file = json.loads(fjson.read())
    js = pd.io.json.json_normalize(file, record_path)
    m = pd.io.json.json_normalize(file, record_path[:-1])
    r = pd.io.json.json_normalize(file, record_path[:-2])

    idx = np.repeat(r['context'].values, r.qas.str.len())
    ndx = np.repeat(m['id'].values, m['answers'].str.len())
    m['context'] = idx
    js['q_idx'] = ndx
    main = pd.concat([m[['id', 'question', 'context']].set_index('id'), js.set_index('q_idx')], 1,
                     sort=False).reset_index()
    main['c_id'] = main['context'].factorize()[0]
    return main


def convert_data(data_df):
    global tokenizer
    indices, segments, target_start, target_end = [], [], [], []

    for i in range(len(data_df)):
        que, _ = tokenizer.encode(data_df['question'][i])
        doc, _ = tokenizer.encode(data_df['context'][i])
        doc.pop(0)

        que_len = len(que)
        # doc_len = len(doc)

        if que_len > 64:
            que = que[:63]
            que.append(102)

        if len(que + doc) > 512:
            while len(que + doc) != 512:
                doc.pop(-1)

            doc.pop(-1)
            doc.append(102)

        segment = [0] * len(que) + [1] * len(doc) + [0] * (512 - len(que) - len(doc))
        if len(que + doc) <= 512:
            while len(que + doc) != 512:
                doc.append(0)

        ids = que + doc

        texts = data_df['text'][i]
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
                ans_start = 512
                ans_end = 512

        indices.append(ids)
        segments.append(segment)
        target_start.append(ans_start)
        target_end.append(ans_end)

    indices_x = np.array(indices)
    segments = np.array(segments)
    target_start = np.array(target_start)
    target_end = np.array(target_end)

    del_list = np.where(target_start != 512)[0]
    indices_x = indices_x[del_list]
    segments = segments[del_list]

    target_start = target_start[del_list]
    target_end = target_end[del_list]
    return [indices_x, segments], [target_start, target_end]


def load_data(data_df):
    data_df['context'] = data_df['context'].astype(str)
    data_df['question'] = data_df['question'].astype(str)
    data_x, data_y = convert_data(data_df)
    return data_x, data_y
