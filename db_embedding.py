from utils import load_vectorstore 
from langchain.vectorstores import Chroma

### TO DO
    # Create replace dowload and load, with simple langchain load
    # Create local, persistent vector data base
    # Improve text splitter / loader, think about data sets in the future
    # add more data sets


def new_query_quotes(query,k=3):
    # make sure database is build with data (verify)
    # load_vector store to memory
    # send query, with top k
    # return response
    db = load_vectorstore()
    hits = db.similarity_search(query=query,k=k)
    return hits

# The below needs tweeking - see TO DO above
# def query_quotes(query,k=3):
#     """Connects to OpenAI embedding model, loads/splits documents into Chroma DB 
#         then implements similarity search based on user query
#     """
#     os.environ["OPENAI_API_KEY"] = open_ai_key
#     embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
#     
#     data_dir = os.path.join(os.getcwd(),'data_store','quotes')
#     docs = download_and_load(data_dir)
#     
#     # Lets split the document using Chracter splitting. 
#     splitter = CharacterTextSplitter(separator = "\n",
#                                     chunk_size=700, 
#                                     chunk_overlap=0,
#                                     length_function=len)
#    documents = splitter.split_documents(docs)
#     
#     chromadb = Chroma.from_documents(documents, embeddings) 
#     
#     results = chromadb.similarity_search(query,k=k)
#
#     return results
