from flask import Flask, request, render_template, redirect, session, flash

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import requests
import json
import settings as settings

import traceback
import logging

app = Flask(__name__)  # type:Flask
app.secret_key = "!@#$%^&*()sdbvdff"

interface_url = settings.INTERFACE_URL

# flow limit
limiter = Limiter(app=app, key_func=get_remote_address, default_limits=["1000/day, 10/minute, 1/2second"])

#logging
handler = logging.FileHandler('./Frontend/logs/app.log', encoding='UTF-8')

logging_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(funcName)s - %(lineno)s \n - %(message)s')
handler.setFormatter(logging_format)
app.logger.addHandler(handler)



# white list
@limiter.request_filter
def filter_white_list():
    path_url = request.path
    white_list = ['/main/', '/answer_search/']
    if path_url in white_list:
        return True
    else:
        return False


@app.errorhandler(500)
def not_found(e):
    return redirect('/main/')


# redirect
@app.route('/', methods=['GET', 'POST'])
def ma():
    return redirect('/main/')


# Main page
@app.route('/main/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        session['question_session']=''
        try:
            return render_template('index.html')
        except Exception as e:
            app.logger.exception(e)
            traceback.print_exc()
            flash('Please try again later.')

    else:
        question = request.form.get('question')
        start_date = request.form.get('trip-start')
        end_date = request.form.get('trip-end')
        # check illegal character
        question_str = str(question)

        if question_str == '':
            return render_template('index.html')
       
        result_basic = requests.post('http://127.0.0.1'+interface_url+'answer_search/', 
                                     json={'question': question, 'start': start_date, 'end': end_date}).json()
        result_basic = json.loads(result_basic)
        print(result_basic)
        if result_basic['answer'] == None:
            err = 'Illegal date, please retry'
            return render_template('index.html', msg=err)
        session['question_session'] = question_str
        return render_template('basic.html', **result_basic)


if __name__ == "__main__":
    # local testing
    app.run(host='127.0.0.1', port=8080, debug=True)
    requests.adapters.DEFAULT_RETRIES = 5
    s = requests.session()
    s.keep_alive = False
