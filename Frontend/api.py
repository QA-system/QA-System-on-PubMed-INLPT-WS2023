# API
from __future__ import unicode_literals


from flask import Flask
from flask_restful import Resource, Api, reqparse

import sys
sys.path.insert(0, sys.path[0]+"/../")
from answer_generation.model_utils import question_answer
import hashlib

import logging

import json
import settings as settings


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
        args = parser.parse_args()
        
        question = args['question']

        answer = self.get_answer(question)

        qa_pair = f'{question} {answer}'
        qa_id = hashlib.sha256(qa_pair.encode('utf-8')).hexdigest()   
        
        m= {'question':question,'answer':answer,'qa_id': qa_id}
        return json.dumps(m, ensure_ascii=False)


    def get(self):
        return json.dumps({}, ensure_ascii=False)
    
    def get_answer(self, question):
        answer = question_answer(question)
        return answer

api.add_resource(Getanswer, '/answer_search/')

if __name__ == '__main__':
    app.run(host='127.0.0.1',port=8010,debug=True)
    #app.run(host='0.0.0.0')