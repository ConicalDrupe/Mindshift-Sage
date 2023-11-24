import os
from open_ai_key import open_ai_key
from langchain.llms import OpenAI 
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from db_embedding import query_quotes


def retrieve_insight(user_input):
    # Set openai key environment variable
    os.environ["OPENAI_API_KEY"] = open_ai_key
    
    template = """
    You are a wise robot, well studied in philosophy. You are giving advice to a human,
    who is giving you a situation in their life. Output a friendly sentiment to their situation and a
    relevant philosophy quote from a list of quotes. Cite the author.
    
    Human Situation: {user_input}
    
    List of Quotes: {list_of_quotes}
    
    Seperate your output by newlines.
    """
    test = """    Example:
        Human Situation: My dog died the other day.
        I am so sorry to hear about your dog, losing something close to you is never easy.
        (Philosopher's name) has something to say about this...
    
        (Philosopher quote here) -(Philosopher's name)
    """
    prompt_template = PromptTemplate(template=template,input_variables=["user_input","list_of_quotes"])
    
    
    llm = OpenAI(model="text-davinci-003",temperature=0.1)
    
    
    llm_chain = LLMChain(prompt=prompt_template,llm=llm)
    quotes_list = query_quotes(user_input, k=3)
    response = llm_chain.run({"user_input":user_input,"list_of_quotes":quotes_list})

    return response

if __name__ == "__main__":
    response = retrieve_insight("I had a hard day at work.")
    print(response)
