from fastapi import FastAPI
from rag import Rag

app = FastAPI()
rag = None
session_info = {}
chat_history = []

@app.get('/')
def hello(name: str = 'Prabhu'):
    return f'Hello {name}'

@app.post('/upload')
def upload_file(path: str):
    global rag
    session_info['file_path'] = path
    rag = Rag(file_path=path)
    return session_info

@app.post('/query')
def query(prompt: str, chat_history=chat_history):
    global rag
    if rag is None:
        return {"error": "RAG model not initialized. Please upload a file first."}
    
    res = rag.query(prompt=prompt, chat_history=chat_history)
    chat_history.append((prompt, res))
    return res

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
