from copy import deepcopy
import pandas as pd
import os
from transformers import (
    pipeline, 
    AutoModelForQuestionAnswering, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    AutoConfig,
    AutoModelForCausalLM,
)
from dotenv import load_dotenv, find_dotenv

import locale
import json
from transformers import LlamaTokenizer
from tqdm.auto import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import BM25Retriever
from torch import bfloat16
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

locale.getpreferredencoding = lambda: "UTF-8"
pd.options.mode.copy_on_write = True
_ = load_dotenv(find_dotenv())
hf_auth = os.environ.get('HF_AUTH')

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



# RAG model
def load_dataset():
    last_path = os.getcwd()
    file_path = f"{last_path}/data/pubmed_intelligence.json"
    with open(file_path, "r", encoding="utf-8") as file:
        docs = json.load(file)
    return docs

def get_tokenizer():
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf",use_auth_token=hf_auth)
    return tokenizer

def token_len(text):
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text)
    return len(tokens)

def dataset_processing(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=token_len,
        separators=['\n\n', '\n', ' ', '']
    )
    documents = []

    for doc in tqdm(docs[:10]):
        uid = doc['PMID']
        chunks = text_splitter.split_text(doc['Abstract'])
        for i, chunk in enumerate(chunks):
            documents.append({
                'id': f'{uid}-{i}',
                'text': chunk,
                'source': doc
            })

    data = pd.DataFrame(documents)
    return data


def load_embedding_model():
    embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'
    device = 'cuda:0'

    embed_model = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': device},
        encode_kwargs={'device': device, 'batch_size': 32}
    )
    return embed_model

def semantic_search(data, filtered_docs):
    embed_model = load_embedding_model()
    vectordb = Chroma.from_texts(texts=list(data['text']), embedding=embed_model, persist_directory="chroma_db")
    faiss_retriever = vectordb.as_retriever(search_kwargs={"k":5,})
    bm25_retriever = BM25Retriever.from_texts(filtered_docs)
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever,faiss_retriever],
                                       weights=[0.5,0.5])
    return ensemble_retriever
    
def facet_search(docs, arg):
    author = 'Hamrouni K'
    start_date = '2013/01/01'
    end_date = '2023/01/01'

    # filtered_docs = [doc['Abstract'] for doc in docs if (author in doc['Authors']) and (doc['ArticleDate']>start_date and doc['ArticleDate']<end_date)]
    filtered_docs = [doc['Abstract'] for doc in docs if (doc['ArticleDate']>start_date and doc['ArticleDate']<end_date)]

    return filtered_docs

def get_rag_model():
    model_id = 'meta-llama/Llama-2-13b-chat-hf'
    bitsAndBites_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16
        )
    model_config = AutoConfig.from_pretrained(
        model_id,
        use_auth_token=os.environ.get('HF_AUTH')
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bitsAndBites_config,
        device_map='auto',
        token=os.environ.get('HF_AUTH')
    )
    model.eval()
    return model

def generate_rag_pipeline(model, tokenizer, ensemble_retriever):
    generate_text = pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=True,
        task='text-generation',
        temperature=0.01,
        max_new_tokens=512,
        repetition_penalty=1.1
    )

    llm = HuggingFacePipeline(pipeline=generate_text)
    rag_pipeline = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        verbose=True,
        retriever=ensemble_retriever,
        chain_type_kwargs={
            "verbose": True },
    )
    return rag_pipeline

def rag_full_pipe(query:str = 'What are the three successive steps involved in the novel mass detection process described for identifying breast abnormalities on mammographic images, according to the paper?'):
    docs = load_dataset()
    tokenizer = get_tokenizer()
    data = dataset_processing(docs)
    filtered_docs = facet_search(docs)
    ensemble_retriever = semantic_search(data, filtered_docs)
    rag_model = get_rag_model()
    rag_pipeline = generate_rag_pipeline(rag_model, tokenizer, ensemble_retriever)
    
    return rag_pipeline
    # result = rag_pipeline(query)
    # return result


if __name__=="__main__":
    result = rag_full_pipe()
    # result = question_answer('2')
    print(result)