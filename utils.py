from huggingface_hub import hf_hub_download
from langchain.document_loaders import CSVLoader
import os

def download_and_load(download_path,repo_id='datastax/philosopher-quotes',filename='philosopher-quotes.csv',repo_type='dataset', check=False):
    """Input download path,HF repoid,filename,type, and ruturns CSV loaded document"""
    target_dir = os.path.join(os.getcwd(),'data_store',download_path)
    if len(os.listdir(target_dir)) == 0: #checking if target path is empty
        hf_hub_download(repo_id=repo_id, filename=filename,repo_type=repo_type,local_dir=download_path)
        print("New files located in ")
        for (root, dirs, file) in os.walk(download_path,topdown=True):
            print(root)
            print(dirs)
            print(file)
            print("---------------Next-Root-----------------")

    else:
        pass
        #print("Nothing Downloaded, file already exists")

    # Load with CSVLoader
    loader = CSVLoader(os.path.join(target_dir,'philosopher-quotes.csv')) # Is there a way to store author and tags into metadata for textsplitter?
    document = loader.load()

    if check:
        doc_len = len(document)
        print(f"Document contains {doc_len} docs")
        for i, doc in enumerate(document):
            print(f"Page {i} page content ", doc.page_content)
            print(f"Page {i} metadata ", doc.metadata)
            if i == 2: break
        pass

    return document
