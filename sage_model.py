import os
from open_ai_key import open_ai_key
from hf_api_key import hf_api_key
from langchain.llms import OpenAI 
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from db_embedding import new_query_quotes
from transformers import AutoTokenizer, AutoModel, pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

# Insert output parser for response 

def retrieve_hf_insight(user_input):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_key
    # model_dir = os.path.join(os.getcwd(),'data_store','llm_cache')
    cache_dir = '~/.cache/huggingface/hub/'
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct",cache_dir=cache_dir)
    model = AutoModel.from_pretrained("tiiuae/falcon-7b-instruct",cache_dir=cache_dir)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    hf = HuggingFacePipeline(pipeline=pipe)

    template = """
    You are a wise robot, well studied in philosophy. You are giving advice to a human,
    who is giving you a situation in their life. Output a friendly sentiment to their situation and a
    relevant philosophy quote from a list of quotes. Cite the author.
    
    Human Situation: {user_input}
    
    List of Quotes: {list_of_quotes}
    
    Seperate your output by newlines.
    """
    # prompt_template = PromptTemplate(template=template,input_variables=["user_input","list_of_quotes"])
    prompt_template = PromptTemplate.from_template(template=template)

    quotes_list = new_query_quotes(user_input, k=1)

    llm_chain = prompt_template | hf

    response = llm_chain.invoke({"user_input":user_input,"list_of_quotes":quotes_list})
    print("LLM Chain ran")

    return response

# def retrieve_open_ai_insight(user_input):
#     # Set openai key environment variable
#     os.environ["OPENAI_API_KEY"] = open_ai_key
#     
#     template = """
#     You are a wise robot, well studied in philosophy. You are giving advice to a human,
#     who is giving you a situation in their life. Output a friendly sentiment to their situation and a
#     relevant philosophy quote from a list of quotes. Cite the author.
#     
#     Human Situation: {user_input}
#     
#     List of Quotes: {list_of_quotes}
#     
#     Seperate your output by newlines.
#     """
#     test = """    Example:
#         Human Situation: My dog died the other day.
#         I am so sorry to hear about your dog, losing something close to you is never easy.
#         (Philosopher's name) has something to say about this...
#     
#         (Philosopher quote here) -(Philosopher's name)
#     """
#     prompt_template = PromptTemplate(template=template,input_variables=["user_input","list_of_quotes"])
#     
#     
#     llm = OpenAI(model="text-davinci-003",temperature=0.1)
#     
#     
#     llm_chain = LLMChain(prompt=prompt_template,llm=llm)
#     quotes_list = query_quotes(user_input, k=3)
#     response = llm_chain.run({"user_input":user_input,"list_of_quotes":quotes_list})
#
#     return response

if __name__ == "__main__":
    prompt_q = "I had a hard day at work."
    print("Input: ", prompt_q)
    response = retrieve_hf_insight(prompt_q)
    print(response)
