import os
from open_ai_key import open_ai_key
from hf_api_key import hf_api_key
from langchain.llms import OpenAI 
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from db_embedding import new_query_quotes

# Insert output parser for response 

def retrieve_hf_insight(user_input):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_key

    template = """
    You are a wise robot, well studied in philosophy. You are giving advice to a human,
    who is giving you a situation in their life. Output a friendly sentiment to their situation and a
    relevant philosophy quote from a list of quotes. Cite the author.
    
    Human Situation: {user_input}
    
    List of Quotes: {list_of_quotes}
    
    Seperate your output by newlines.
    """
    prompt_template = PromptTemplate(template=template,input_variables=["user_input","list_of_quotes"])

    repo_id = "tiiuae/falcon-7b"
    model_kwargs = {'temperature':0.1} 
                    #,
                    #'max_length':200}

    llm = HuggingFaceHub(repo_id=repo_id,model_kwargs=model_kwargs)
    
    
    llm_chain = LLMChain(prompt=prompt_template,llm=llm)
    quotes_list = new_query_quotes(user_input, k=3)
    print("Quotes : ", quotes_list)
    response = llm_chain.run({"user_input":user_input,"list_of_quotes":quotes_list})

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
