import os
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

def load_vectorstore():
    """ Make sure to run build.py first, to instantiate and create embeddings """
    hf_embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2')
    db_dir = os.path.join(os.getcwd(),'data_store','db')
    db = Chroma(persist_directory=db_dir, embedding_function = hf_embedding)
    return db

def add_to_vectorstore(db,split_documents):
    """ Pass in db from load_vectorstore """
    db.add(split_documents)
    print("Add complete")
    return
