from connection import client
from config import index_name

# create index
with open("pubmed_intelligence_mappings.json", "r") as mapping_file:
    mapping_json = mapping_file.read()
try:
    response = client.indices.create(index_name,body=mapping_json)
    print("Creating index:")
    print(response)
except Exception as e:
    print(f"Error: {e}")