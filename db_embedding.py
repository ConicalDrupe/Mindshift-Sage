import os
from open_ai_key import open_ai_key
from utils import download_and_load 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import CSVLoader

### TO DO
    # Create replace dowload and load, with simple langchain load
    # Create local, persistent vector data base
    # Improve text splitter / loader, think about data sets in the future
    # add more data sets

def load_quotes():
    quotes_path = os.path.join(os.getcwd(),'data_store','quotes')
    loader = CSVLoader(os.path.join(quotes_path,'philosopher-quotes.csv')) # Is there a way to store author and tags into metadata for textsplitter?
    document = loader.load()
    return document

# The below needs tweeking - see TO DO above
def query_quotes(query,k=3):
    """Connects to OpenAI embedding model, loads/splits documents into Chroma DB 
        then implements similarity search based on user query
    """
    os.environ["OPENAI_API_KEY"] = open_ai_key
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    
    data_dir = os.path.join(os.getcwd(),'data_store','quotes')
    docs = download_and_load(data_dir)
    
    # Lets split the document using Chracter splitting. 
    splitter = CharacterTextSplitter(separator = "\n",
                                    chunk_size=700, 
                                    chunk_overlap=0,
                                    length_function=len)
    documents = splitter.split_documents(docs)
    
    chromadb = Chroma.from_documents(documents, embeddings) 
    
    results = chromadb.similarity_search(query,k=k)

    return results
