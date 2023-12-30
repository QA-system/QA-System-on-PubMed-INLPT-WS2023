def process_query(query,nlp):
    
    # Tokenize the query
    tokens = [token.text.lower() for token in nlp(query)]
    tokens = [token for token in tokens if not nlp.vocab[token].is_stop]

    # Perform Named Entity Recognition (NER)
    entities = [(ent.text, ent.label_) for ent in nlp(query).ents]

    return tokens, entities


def construct_elasticsearch_query(tokens, entities, N=20, pub_date_range=None, authors=None):
    title_should_clauses = [{"match": {"Title": {"query": token, "boost": 2}}} for token in tokens]

    abstract_should_clauses = [{"match": {"Abstract": {"query": token, "boost": 1}}} for token in tokens]

    title_should_clauses.extend([{"match": {"Title": {"query": entity[0], "boost": 3}}} for entity in entities])

    abstract_should_clauses.extend([{"match": {"Abstract": {"query": entity[0], "boost": 2}}} for entity in entities])

    # Combine should clauses into a bool query for "Title" and "Abstract" fields
    title_bool_query = {"bool": {"should": title_should_clauses}}
    abstract_bool_query = {"bool": {"should": abstract_should_clauses}}

    bool_query = {"bool": {"should": [title_bool_query, abstract_bool_query]}}

    # Add filters for both "pub date" and "author" facets
    filter_clauses = []

    if pub_date_range:
        filter_clauses.append({"range": {"PubDateEDAT": {"gte": pub_date_range[0], "lte": pub_date_range[1]}}})

    if authors:
        filter_clauses.append({"terms": {"Authors.keyword": authors}})

    if filter_clauses:
        bool_query["bool"]["filter"] = filter_clauses

    # Create a query with the bool query
    es_query = {"query": bool_query, "size": N, "sort": [{"_score": {"order": "desc"}}]}

    return es_query