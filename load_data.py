import json

from connection import client
from config import index_name


# Set up the scroll query to retrieve all documents
scroll_size = 1000
scroll = "1m"  

# Initial search to get the first batch of results
response = client.search(
    index=index_name,
    scroll=scroll,
    size=scroll_size,
    body={"query": {"match_all": {}}}
)

hits = response["hits"]["hits"]
all_documents = hits

# Continue scrolling through the results until there are no more hits
while hits:
    response = client.scroll(scroll_id=response["_scroll_id"], scroll=scroll)
    hits = response["hits"]["hits"]
    all_documents.extend(hits)

# Extract the source data from the documents
source_data = [doc["_source"] for doc in all_documents]

# Save the data to a JSON file
output_file = "data/pubmed_intelligence.json"
with open(output_file, "w") as json_file:
    json.dump(source_data, json_file, indent=2)

print(f"All data has been loaded and stored in {output_file}")
