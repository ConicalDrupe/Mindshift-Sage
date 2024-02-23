import os
import errno
from huggingface_hub import hf_hub_download
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from open_ai_key import open_ai_key
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel

# Setup file paths
# Download Files
# create embeddings

def create_data_paths():
    dp = os.path.join(os.getcwd(),'data_store')
    
    if not os.path.exists(dp):
        try:
            os.mkdir(dp)
            os.mkdir(os.path.join(dp,'db'))
            os.mkdir(os.path.join(dp,'quotes'))
            os.mkdir(os.path.join(dp,'llm_cache'))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    print("Data paths created!")
    return

def download_quotes(download_path='quotes',repo_id='datastax/philosopher-quotes',filename='philosopher-quotes.csv',repo_type='dataset', check=False):
    """Input download path,HF repoid,filename,type""" 
    target_dir = os.path.join(os.getcwd(),'data_store',download_path)
    if len(os.listdir(target_dir)) == 0: #checking if target path is empty
        hf_hub_download(repo_id=repo_id, filename=filename,repo_type=repo_type,local_dir=target_dir)
        print("New files located in ")
        for (root, dirs, file) in os.walk(download_path,topdown=True):
            print(root)
            print(dirs)
            print(file)
            print("---------------Next-Root-----------------")
        print("download complete")
    else:
        print("Nothing Downloaded, file already exists")
    return

# load documents -> split documents -> create embedding into persistent directory
def load_and_split_quotes():
    """ Make sure to run build.py first, to download quotes """
    quotes_path = os.path.join(os.getcwd(),'data_store','quotes')
    loader = CSVLoader(os.path.join(quotes_path,'philosopher-quotes.csv')) # Is there a way to store author and tags into metadata for textsplitter?
    document = loader.load()
    split_documents = rec_split_documents(document)
    return split_documents

def rec_split_documents(documents):
    """ Takes in loaded documents, outputs split documents """
    splitter = RecursiveCharacterTextSplitter(
            chunk_size = 625,
            chunk_overlap = 0,
            length_function = len,
            is_separator_regex = False)
    split_documents = splitter.split_documents(documents)
    return split_documents

def create_hf_vectorstore(split_documents):
    hf_embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2')
    db_dir = os.path.join(os.getcwd(),'data_store','db')
    Chroma.from_documents(documents=split_documents, embedding=hf_embedding, persist_directory=db_dir)
    return 

def create_openai_vectorstore(split_documents):
    """ TO DO - Update interactive open_ai_key """
    db_dir = os.path.join(os.getcwd(),'data_store','db')
    os.environ["OPENAI_API_KEY"] = open_ai_key
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    db = Chroma.from_documents(split_documents, embedding_function=embeddings, persist_directory=db_dir)
    return

def download_hf_model(repo_id="tiiuae/falcon-7b-instruct"):
    model_dir = os.path.join(os.getcwd(),'data_store','llm_cache')

    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModel.from_pretrained(repo_id)

    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir)
    print("Download complete")
    return

if __name__ == "__main__":
    # create_data_paths()
    # download_quotes()
    # split_documents = load_and_split_quotes()
    # create_hf_vectorstore(split_documents)
    # print("Chroma Vectorstore created!")
    download_hf_model(repo_id="tiiuae/falcon-7b-instruct")

