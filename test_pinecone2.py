import os
import json
import urllib.request
from pinecone import Pinecone

req = urllib.request.Request('http://127.0.0.1:8000/api/v1/embed', data=json.dumps({'text': 'what is article 11'}).encode('utf-8'), headers={'Content-Type': 'application/json'})
res = urllib.request.urlopen(req)
embedding = json.loads(res.read().decode('utf-8'))['embedding']

pc = Pinecone(api_key='pcsk_6e695v_G98gDsvKpTprBDdZzwnbbxKuxAuu72BtRU9xvjP9D3nv8UpBVx5tCv6nkU6WW7o')
index = pc.Index('legalmind-index')
query_res = index.query(vector=embedding, top_k=3, include_metadata=True, namespace='')

for match in query_res['matches']:
    text = match['metadata'].get('text', '')[:100]
    print(f"Score: {match['score']}, Text snippet: {text}...")
