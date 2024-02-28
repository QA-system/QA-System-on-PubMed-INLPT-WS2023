from copy import deepcopy
import pandas as pd
import os
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer

pd.options.mode.copy_on_write = True

def formatting_data():
    last_path = os.getcwd()
    with open(f'{last_path}/testing_QA/reformatting_testing_qa.csv','r',encoding='utf-8') as f:
        df_c = pd.read_csv(f)

    df_latest = deepcopy(df_c)

    col_name=df_latest.columns.tolist() 
    col_name.insert(0, 'id')

    df_latest = df_latest.reindex(columns=col_name)  

    for i in range(len(df_c['answer'])):
        new_context = f"{df_c['article_title'][i]}\n{df_c['author'][i]}\n{df_c['context'][i]}\n{df_c['article_date'][i]}\n{df_c['PMID'][i]}"
        if not df_c['start_point'][i].isnumeric():
            start_point = new_context.index(' '.join(df_c['start_point'][i].split()[0:2]))
            df_latest.replace({"start_point": {
                    df_c['start_point'][i]: str([start_point])
                }}, inplace=True)
            temp_answer = {'text': [df_c['answer'][i]], 'answer_start': [start_point]}
        else:
            temp_answer = {'text': [df_c['answer'][i]], 'answer_start': [df_c['start_point'][i]]}
        df_latest.replace({"answer": {
                    df_c['answer'][i]: str(temp_answer)
                }}, inplace=True)

        df_latest.replace({"context": {
                    df_c['context'][i]: new_context
                }}, inplace=True)
        
        df_latest.at[i, 'id']= str(int(i))

    return df_latest


def generate_context():
    df = formatting_data()
    context = ' '.join(df['context'])
    return context

def get_model(model_hf: str="choichoi/transformer_qa_with_batch"):
    model = AutoModelForQuestionAnswering.from_pretrained(model_hf)
    return model

def question_answer(question, model_checkpoint: str="distilbert-base-uncased")->str:
    if question is None or question.strip()=='':
        return None
    context = generate_context()
    model = get_model()
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)

    result = pipe({"question": question, 'context': context}, function_to_apply="none")
    if result['answer'] is not None and len(result['answer'])>0:
        return result['answer']
    else:
        return None


if __name__=="__main__":
    result = question_answer('2')
    print(result)