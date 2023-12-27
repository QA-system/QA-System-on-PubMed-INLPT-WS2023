'''
Data Collection:

1. install EDirect
sh -c "$(curl -fsSL https://ftp.ncbi.nlm.nih.gov/entrez/entrezdirect/install-edirect.sh)"

2. fetch all PubMed ids
esearch -db pubmed -query "intelligence[tiab] AND 2013:2023[dp] AND hasabstract" | efetch -format uid > pmids.csv

3. fetch abstract of all the articles based on PMID as below
'''

import logging
import pandas as pd
from datetime import datetime

from Bio import Entrez, Medline

from config import index_name
from connection import client

logging.basicConfig(filename='nohup.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

Entrez.email = "huqiaowen0104@gmail.com"  # Set your email address

pmids_file_path = 'data/pmids.csv'


def fetch_pubmed_record(pmid):
    try:
        # Fetch the record from PubMed
        handle = Entrez.efetch(db="pubmed", id=pmid, rettype="medline", retmode="text")
        records = Medline.parse(handle)
        record = next(records)

        # Extract information
        pubmed_id = record.get("PMID", "")
        title = record.get("TI", "")
        abstract = record.get("AB", "")
        keywords = record.get("OT", [])
        authors = record.get("AU", [])
        pub_date_edat = record.get("EDAT", "")
        article_date = record.get("CRDT", "")
        journal = record.get("JT", "")

        return {
            'PMID': pubmed_id,
            'Title': title,
            'Abstract': abstract,
            'Keywords': keywords,
            'Authors': authors,
            'PubDateEDAT': pub_date_edat.split(" ")[0],
            'ArticleDate':article_date[0].split(" ")[0],
            'Journal': journal
            
        }
    except Exception as e:
        logging.error(f"Error: {e}")
        return None
    
if __name__ == "__main__":
    # fetch and store data
    pmids_df = pd.read_csv(pmids_file_path,header=None,names=['pmid'])

    batch_size = 10
    start_index = 0
    for i in range(start_index,len(pmids_df),batch_size):
        data_batch = []
        pmids = pmids_df['pmid'][i:i+batch_size].tolist()
        start_t = datetime.now()
        for pmid in pmids:
            record_data = fetch_pubmed_record(pmid)
            record_index = {"index": {"_index": index_name, "_id": record_data["PMID"]}}
            data_batch.append(record_index)
            data_batch.append(record_data)
        # Bulk index the data
        client.bulk(body=data_batch, index=index_name)
        end_t = datetime.now()
        logging.info(f"{i+batch_size}-th record has been stored into opensearch. Cost {(end_t-start_t).total_seconds()} seconds")