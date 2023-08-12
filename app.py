import openai
import streamlit as st
from plotly.graph_objects import Figure
import mysql.connector as connection 
import plotly.express as px
import plotly.graph_objs as go
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
import os
load_dotenv()
from analyze import AnalyzeGPT, ChatGPT_Handler


# # Only load the settings if they are running local and not in Azure
# if os.getenv('WEBSITE_SITE_NAME') is None:
#     env_path = Path('.') / 'secrets.env'
#     load_dotenv(dotenv_path=env_path)

def load_setting(setting_name, session_name,default_value=''):  
    """  
    Function to load the setting information from session  
    """  
    if session_name not in st.session_state:  
        if os.environ.get(setting_name) is not None:
            st.session_state[session_name] = os.environ.get(setting_name)
        else:
            st.session_state[session_name] = default_value  

load_setting("AZURE_OPENAI_CHATGPT_DEPLOYMENT","chatgpt","gpt-3.5-turbo") 
if 'show_settings' not in st.session_state:  
    st.session_state['show_settings'] = False


def saveOpenAI():
    st.session_state.chatgpt = st.session_state.txtChatGPT
    st.session_state['show_settings'] = False

def toggleSettings():
    st.session_state['show_settings'] = not st.session_state['show_settings']

openai.api_key =  os.getenv('OPENAI_API_KEY')
gpt_engine = 'gpt-3.5-turbo'

st.set_page_config(page_title="GenBI", page_icon=":memo:", layout="wide")
col1, col2  = st.columns((3,1)) 

with st.sidebar:  
    custom_css_image = """
    <style>
            width: 90%;
            margin-bottom:0px;
        }
    </style>
    """
    custom_css_h1 = """
    <style>
    h1 {
        color: #FFF39F;
        margin-top: 0;
    }
    </style>
    """
    custom_css_h5 = """
    <style>
    h5 {
        color: #AAAAAA;
        margin-top: 0;
    }
    </style>
    """
    st.markdown(custom_css_image, unsafe_allow_html=True)
    image = Image.open('.\\image\\PwC Logo.png')
    left_co, cent_co,last_co = st.columns(3)
    with cent_co:
        st.image(image , width=100,)
    st.markdown(custom_css_h1, unsafe_allow_html=True)
    st.markdown(custom_css_h5, unsafe_allow_html=True)
    st.markdown('<h1 align="center">Generative BI</h1>', unsafe_allow_html=True)
    #st.markdown('<h5 align="center">©️PwC Advanced Analytics Team</h5>', unsafe_allow_html=True)
    options = (["Data Analysis Assistant"])
    index = st.radio("Choose the app", range(len(options)), format_func=lambda x: options[x])
    system_message="""
        You are a smart AI assistant to help answer business questions based on analyzing data. 
        You can plan solving the question with one more multiple thought step. At each thought step, you can write python code to analyze data to assist you. Observe what you get at each step to plan for the next step.
        You are given following utilities to help you retrieve data and commmunicate your result to end user.
        1. execute_sql(sql_query: str): A Python function can query data from the <<data_sources>> given a query which you need to create. The query has to be syntactically correct for {sql_engine} and only use tables and columns under <<data_sources>>. The execute_sql function returns a Python pandas dataframe contain the results of the query.
        2. Use plotly library for data visualization. 
        3. Use observe(label: str, data: any) utility function to observe data under the label for your evaluation. Use observe() function instead of print() as this is executed in streamlit environment. Due to system limitation, you will only see the first 10 rows of the dataset.
        4. To communicate with user, use show() function on data, text and plotly figure. show() is a utility function that can render different types of data to end user. Remember, you don't see data with show(), only user does. You see data with observe()
            - If you want to show  user a plotly visualization, then use ```show(fig)`` 
            - If you want to show user data which is a text or a pandas dataframe or a list, use ```show(data)```
            - Never use print(). User don't see anything with print()
        5. Lastly, don't forget to deal with data quality problem. You should apply data imputation technique to deal with missing data or NAN data.
        6. Always follow the flow of Thought: , Observation:, Action: and Answer: as in template below strictly. 

        """
    few_shot_examples="""
        <<Template>>
        Question: User Question
        Thought 1: Your thought here.
        Action: 
        ```python
        #Import neccessary libraries here
        import numpy as np
        #Query some data 
        sql_query = "SOME SQL QUERY"
        step1_df = execute_sql(sql_query)
        # Replace NAN with 0. Always have this step
        step1_df['Some_Column'] = step1_df['Some_Column'].replace(np.nan,0)
        #observe query result
        observe("some_label", step1_df) #Always use observe() instead of print
        ```
        Observation: 
        step1_df is displayed here
        Thought 2: Your thought here
        Action:  
        ```python
        import plotly.express as px 
        #from step1_df, perform some data analysis action to produce step2_df
        #To see the data for yourself the only way is to use observe()
        observe("some_label", step2_df) #Always use observe() 
        #Decide to show it to user.
        fig=px.line(step2_df)
        #visualize fig object to user.  
        show(fig)
        #you can also directly display tabular or text data to end user.
        show(step2_df)
        ```
        Observation: 
        step2_df is displayed here
        Answer: Your final answer and comment for the question. Also use Python for computation, never compute result youself.
        <</Template>>

        """

    extract_patterns=[("Thought:",r'(Thought \d+):\s*(.*?)(?:\n|$)'), ('Action:',r"```python\n(.*?)```"),("Answer:",r'([Aa]nswer:) (.*)')]
    extractor = ChatGPT_Handler(extract_patterns=extract_patterns)

    st.button("Settings",on_click=toggleSettings)
    if st.session_state['show_settings']:  
        # with st.expander("Settings",expanded=expandit):
        with st.form("AzureOpenAI"):
            st.markdown("<b>Azure OpenAI Settings</b>",  unsafe_allow_html=True)
            st.text_input("ChatGPT deployment name:", value=st.session_state.chatgpt,key="txtChatGPT")  
            st.form_submit_button("Submit",on_click=saveOpenAI)

    chat_list=[]
    if st.session_state.chatgpt != '':
        chat_list.append("ChatGPT")

    show_code = st.checkbox("Show code", value=False)  
    show_prompt = st.checkbox("Show prompt", value=False)
    question = st.text_area("Ask me a question")
  
    if st.button("Submit"):  
        analyzer = AnalyzeGPT(extract_patterns=extract_patterns, content_extractor= extractor, 
                              system_message=system_message, few_shot_examples=few_shot_examples,st=st)  

        analyzer.run(question,show_code,show_prompt, col1)  
    # else:
    #     st.error("Not implemented yet!")
    custom_css_footnote = """
    <style>
    .footnote {
        font-size: 14px;
        color: #777777;
        position: absolute;
        bottom: 10px;
        right: 10px;
    }
    .footnote a {
    text-decoration: none;
    color: #AAAAAA;
    }
    </style>
    """
    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown(custom_css_footnote, unsafe_allow_html=True)
    st.markdown('<div class="footnote">Developed by: <a href="mailto:krishnendu.dey@pwc.com">PwC Advanced Analytics Team</a></div>', unsafe_allow_html=True)
