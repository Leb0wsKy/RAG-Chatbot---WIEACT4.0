# src/embedder.py
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity



def embed_texts(texts,model):
    embeddings = [model.encode(text, show_progress_bar=False, convert_to_numpy=True) for text in texts]
    return embeddings

def cos_sim(query,texts) :
    res = []
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embed = model.encode(query, show_progress_bar=False, convert_to_numpy=True)
    embeddings = embed_texts(texts,model)
    for i in range (len(embeddings)):
        res_embed = cosine_similarity([query_embed], [embeddings[i]])
        res.append((texts[i], res_embed[0][0]))    
    
    res = [res_item for res_item in res if res_item[1] > 0.2]
    return res

    

query = "What is the impact of climate change on agriculture?"
texts = ["Climate change affects agriculture by altering rainfall patterns and increasing the frequency of extreme weather events.","Oussema is stupid"]
res = cos_sim(query,texts)
for result in res:
    print(result[0],result[1])