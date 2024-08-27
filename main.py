from urllib.request import Request
import streamlit as st

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI 
from langchain_community.agent_toolkits import GmailToolkit
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain import hub
from google_auth_oauthlib.flow import InstalledAppFlow
import os
import pickle 

from langchain_community.tools.gmail.utils import (
    build_resource_service,
    get_gmail_credentials,
)

from dotenv import load_dotenv

load_dotenv()



os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

SCOPES = [
    'https://www.googleapis.com/auth/gmail.modify',  
    'https://www.googleapis.com/auth/gmail.compose',
    # 'https://www.googleapis.com/auth/gmail.readonly'  
]

def get_gmail_credentials():
    creds = None
    token_path = 'token.pickle'
    
    # Check if token.pickle exists and load it
    if os.path.exists(token_path):
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)
    
    # If credentials are not valid, re-authenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Start the OAuth flow to get new credentials
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the new credentials
        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)
    
    return creds

# In your Streamlit app

credentials = get_gmail_credentials()


# credentials = get_gmail_credentials(
#     token_file="token.json",
#     scopes=["https://mail.google.com/"],
#     client_secrets_file="credentials.json",
# )

api_resource = build_resource_service(credentials=credentials)
toolkit = GmailToolkit(api_resource=api_resource)

tools = toolkit.get_tools()


instructions = """You are an assistant that creates email drafts. when a user asks to modify to change the last email. They are talking about the 
                    the most recent draft. You are also able to delete drafts when the user asks"""
base_prompt = hub.pull("langchain-ai/openai-functions-template")
prompt = base_prompt.partial(instructions=instructions)
llm = ChatOpenAI(temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)

memory = ConversationBufferMemory(memory_key="chat_history")

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    
    verbose=True,
    return_intermediate_steps=True  # Whether to return the agent’s trajectory of intermediate steps at the end in addition to the final output.
)




def invoke_agent(prompt, history):
    
    response = agent_executor.invoke(
        {
            "input": f"{prompt}. This is the last message I sent to you",
            "chat_history": history
            
        }
    )
    
   
    return response

if __name__ == "__main__":
    
    
    response = None
    
    st.write("AI assistant")
    with st.form("Question",clear_on_submit=True):
        request = st.text_input("Ask a question:")
        submitted = st.form_submit_button("Submit")
        if submitted:
                
            if 'history' not in st.session_state:
                st.session_state['history'] = [HumanMessage(content=request)]
                
            
            else:
                # prev_message = st.session_state['history'][-2]
                st.session_state['history'].append(HumanMessage(content=request))
                
            
            response = invoke_agent(request, history = st.session_state['history'])
            
            
            if 'history' in st.session_state:
                st.session_state['history'].append(AIMessage(content=response['output']))
    
    
    if response is not None:
        st.write(response['output'])

