# API
from __future__ import unicode_literals


from flask import Flask
from flask_restful import Resource, Api, reqparse

import sys
sys.path.insert(0, sys.path[0]+"/../")
# from answer_generation.model_utils import question_answer, rag_full_pipe
import hashlib

import logging

import json
import settings as settings
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)
api = Api(app)


# logging
handler = logging.FileHandler('./Frontend/logs/api.log', encoding='UTF-8')

logging_format = logging.Formatter(
            '%(asctime)s - %(levelname)s- %(funcName)s - %(lineno)s \n - %(message)s')
handler.setFormatter(logging_format)
app.logger.addHandler(handler)


class Getanswer(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('question', type=str, required=True)
        parser.add_argument('start', type=str, required=True)
        parser.add_argument('end', type=str, required=True)
        args = parser.parse_args()
        
        question = args['question']
        start_date = args['start']
        end_date = args['end']

        answer = self.get_answer(question, start_date, end_date)

        qa_pair = f'{question} {answer}'
        qa_id = hashlib.sha256(qa_pair.encode('utf-8')).hexdigest()   
        
        m= {'question':question,'answer':answer,'qa_id': qa_id}
        return json.dumps(m, ensure_ascii=False)


    def get(self):
        return json.dumps({}, ensure_ascii=False)
    
    def get_answer(self, question, start_date, end_date):
        if datetime.strptime(start_date,'%Y-%m-%d')>datetime.strptime(end_date,'%Y-%m-%d'):
            return None
        start_date = start_date.replace('-', '/')
        end_date = end_date.replace('-', '/')
        last_path = os.getcwd()
        file_path = f"{last_path}/Frontend/QA_mock_data.xlsx"
        df = pd.read_excel(file_path,engine='openpyxl')
        result = df['query'][df['query'].isin([question.replace('\r','').replace('\n','').strip()])]
        if len(result)>0:
            ind =result.index.tolist()
            answer = df['result'][ind[0]]
            # answer = question_answer(question)
            # answer = rag_full_pipe(start_date, end_date, question)['result']
            return answer
        else:
            return 'There is no answer available.'

api.add_resource(Getanswer, '/answer_search/')

if __name__ == '__main__':
    app.run(host='127.0.0.1',port=8010,debug=True)
    #app.run(host='0.0.0.0')