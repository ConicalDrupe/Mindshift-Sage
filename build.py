import os
import errno
from huggingface_hub import hf_hub_download

# Setup file paths
# Download Files
# create embeddings

def create_data_paths():
    dp = os.path.join(os.getcwd(),'data_dir')
    
    if not os.path.exists(dp):
        try:
            os.mkdir(dp)
            os.mkdir(os.path.join(dp,'db'))
            os.mkdir(os.path.join(dp,'quotes'))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def download_quotes(download_path='quotes',repo_id='datastax/philosopher-quotes',filename='philosopher-quotes.csv',repo_type='dataset', check=False):
    """Input download path,HF repoid,filename,type""" 
    target_dir = os.path.join(os.getcwd(),'data_store',download_path)
    if len(os.listdir(target_dir)) == 0: #checking if target path is empty
        hf_hub_download(repo_id=repo_id, filename=filename,repo_type=repo_type,local_dir=download_path)
        print("New files located in ")
        for (root, dirs, file) in os.walk(download_path,topdown=True):
            print(root)
            print(dirs)
            print(file)
            print("---------------Next-Root-----------------")
        print("download complete")
    else:
        print("Nothing Downloaded, file already exists")

if __name__ == "__main__":
    create_data_paths()
    download_quotes()
