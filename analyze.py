import os
import re
import json
import time
import openai
import string
import pandas as pd
import numpy as np 
from urllib import parse
from dotenv import load_dotenv
from sqlalchemy import create_engine  
import warnings
warnings.filterwarnings('ignore')
from plotly.graph_objects import Figure
import mysql.connector as connection 
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


def createDBConnector():
    '''
        Creates connection string for MySQL Database
    '''
    connector = connection.connect(
                    host='localhost',
                    user= os.getenv('DB_USERNAME'),
                    password=os.getenv('DB_PASSWORD'),
                    database= os.getenv('DB_NAME'),
                    use_pure=True)
    return connector


def execute_sql_query(query, limit=10000):  
    '''
        Returns the table/tables in the form of a dataframe. 
        limit = 10000 is there for the sake of memory consumption.
    '''
    db_connector = createDBConnector()
    result = pd.read_sql_query(query, db_connector)
    result = result.infer_objects()
    for col in result.columns:  
        if 'date' in col.lower():  
            result[col] = pd.to_datetime(result[col], errors="ignore")  

    if limit is not None:  
        result = result.head(limit)  
    return result  

def get_table_schema():
    '''
    Fetches Schema information from the database as mentioned
    '''
    sql_query = F"""  
        SELECT C.TABLE_NAME, C.COLUMN_NAME, C.DATA_TYPE, T.TABLE_TYPE, T.TABLE_SCHEMA  
        FROM INFORMATION_SCHEMA.COLUMNS C  
        JOIN INFORMATION_SCHEMA.TABLES T ON C.TABLE_NAME = T.TABLE_NAME AND C.TABLE_SCHEMA = T.TABLE_SCHEMA  
        WHERE T.TABLE_SCHEMA = '{os.getenv('DB_NAME')}' 
        """  
    df = execute_sql_query(sql_query, limit=None)
    output=[]
    current_table = ''  
    columns = []  
    for index, row in df.iterrows():
        table_name = f"{row['TABLE_SCHEMA']}.{row['TABLE_NAME']}"  
        column_name = row['COLUMN_NAME']  
        data_type = row['DATA_TYPE']  
        if " " in table_name:
            table_name= f"[{table_name}]" 
        column_name = row['COLUMN_NAME']  
        if " " in column_name:
            column_name= f"[{column_name}]" 
            # If the table name has changed, output the previous table's information  
        if current_table != table_name and current_table != '':  
            output.append(f"table: {current_table}, columns: {', '.join(columns)}")  
            columns = []  
        
        # Add the current column information to the list of columns for the current table  
        columns.append(f"{column_name} {data_type}")  
        
        # Update the current table name  
        current_table = table_name  
    
    # Output the last table's information  
    output.append(f"table: {current_table}, columns: {', '.join(columns)}")
    output = "\n ".join(output)
    return output

class ChatGPT_Handler:
    def __init__(self, extract_patterns):
        self.max_response_tokens = 1250
        self.token_limit= 4096
        self.temperature=0
        self.extract_patterns=extract_patterns
        
    def _call_llm(self,prompt, stop):
        response = openai.ChatCompletion.create(
                model ='gpt-3.5-turbo', 
                messages = prompt,
                temperature=self.temperature,
                max_tokens=self.max_response_tokens,
                stop=stop
                )
            
        llm_output = response['choices'][0]['message']['content']
        return llm_output
    
    def extract_output(self, text_input):
            output={}
            if len(text_input)==0:
                return output
            for pattern in self.extract_patterns: 
                if "sql" in pattern[1]:

                    sql_query=""
                    sql_result = re.findall(pattern[1], text_input, re.DOTALL)

                    if len(sql_result)>0:
                        sql_query=sql_result[0]
                        output[pattern[0]]= sql_query
                    else:
                        return output
                    text_before = text_input.split(sql_query)[0].strip("\n").strip("```sql").strip("\n")

                    if text_before is not None and len(text_before)>0:
                        output["text_before"]=text_before
                    text_after =text_input.split(sql_query)[1].strip("\n").strip("```")
                    if text_after is not None and len(text_after)>0:
                        output["text_after"]=text_after
                    return output

                if "python" in pattern[1]:
                    result = re.findall(pattern[1], text_input, re.DOTALL)
                    if len(result)>0:
                        output[pattern[0]]= result[0]
                else:

                    result = re.search(pattern[1], text_input,re.DOTALL)
                    if result:  
                        output[result.group(1)]= result.group(2)

            return output
    

class AnalyzeGPT(ChatGPT_Handler):
    
    def __init__(self,content_extractor, system_message,few_shot_examples,st,**kwargs):
        super().__init__(**kwargs)
        table_schema = "table: mavenfuzzyfactory.website_sessions, columns: website_session_id bigint, created_at timestamp, user_id bigint, is_repeat_session smallint, utm_source varchar, utm_campaign varchar, utm_content varchar, device_type varchar, http_referer varchar\n table: mavenfuzzyfactory.website_pageviews, columns: website_pageview_id bigint, created_at timestamp, website_session_id bigint, pageview_url varchar\n table: mavenfuzzyfactory.products, columns: product_id bigint, created_at timestamp, product_name varchar\n table: mavenfuzzyfactory.orders, columns: order_id bigint, created_at timestamp, website_session_id bigint, user_id bigint, primary_product_id smallint, items_purchased smallint, price_usd decimal, cogs_usd decimal\n table: mavenfuzzyfactory.order_items, columns: order_item_id bigint, created_at timestamp, order_id bigint, product_id smallint, is_primary_item smallint, price_usd decimal, cogs_usd decimal\n table: mavenfuzzyfactory.order_item_refunds, columns: order_item_refund_id bigint, created_at timestamp, order_item_id bigint, order_id bigint, refund_amount_usd decimal"
        system_message = f"""
        <<data_sources>>
        {table_schema}
        {system_message.format(sql_engine="sqlserver")}
        {few_shot_examples}
        """
        self.conversation_history =  [{"role": "system", "content": system_message}]
        self.st = st
        self.content_extractor = content_extractor
        # self.sql_query_tool = sql_query_tool
        
    def get_next_steps(self, updated_user_content, stop):
        old_user_content=""
        if len(self.conversation_history)>1:
            old_user_content= self.conversation_history.pop() #removing old history
            old_user_content=old_user_content['content']+"\n"
        self.conversation_history.append({"role": "user", "content": old_user_content+updated_user_content})
        # print("prompt input ", self.conversation_history)
        n=0
        try:
            llm_output = self._call_llm(self.conversation_history, stop)
        
            print("llm_output \n", llm_output)

        except Exception as e:
            time.sleep(8) #sleep for 8 seconds
            while n<5:
                try:
                    llm_output = self._call_llm(self.conversation_history, stop)
                except Exception as e:
                    n +=1
                    print("error calling open AI, I am retrying 5 attempts , attempt ", n)
                    time.sleep(8) #sleep for 8 seconds
                    print(e)

            llm_output = "OPENAI_ERROR"     
             
    
        # print("llm_output: ", llm_output)
        output = self.content_extractor.extract_output(llm_output)
        if len(output)==0 and llm_output != "OPENAI_ERROR": #wrong output format
            llm_output = "WRONG_OUTPUT_FORMAT"

        return llm_output,output

    def run(self, question: str, show_code,show_prompt,st) -> any:
        import numpy as np
        import plotly.express as px
        import plotly.graph_objs as go
        import pandas as pd

        st.write(f"Question: {question}")

        def execute_sql(query):
            return execute_sql_query(query)
        observation=None
        def show(data):
            if type(data) is Figure:
                st.plotly_chart(data)
            else:
                st.write(data)
            if type(data) is not Figure:
                self.st.session_state[f'observation: this was shown to user']=data
        def observe(name, data):
            try:
                data = data[:10] # limit the print out observation to 15 rows
            except:
                pass
            self.st.session_state[f'observation:{name}']=data

        max_steps = 15
        count =1

        finish = False
        new_input= f"Question: {question}"
        # if self.st.session_state['init']:
        #     new_input= f"Question: {question}"
        # else:
        #     new_input=self.st.session_state['history'] +f"\nQuestion: {question}"
        while not finish:

            llm_output,next_steps = self.get_next_steps(new_input, stop=["Observation:", f"Thought {count+1}"])
            if llm_output=='OPENAI_ERROR':
                st.write("Error Calling Azure Open AI, probably due to max service limit, please try again")
                break
            elif llm_output=='WRONG_OUTPUT_FORMAT': #just have open AI try again till the right output comes
                count +=1
                continue

            new_input += f"\n{llm_output}"
            for key, value in next_steps.items():
                new_input += f"\n{value}"
                
                if "ACTION" in key.upper():
                    if show_code:
                        st.write(key)
                        st.code(value)
                    observations =[]
                    serialized_obs=[]
                    try:
                        # if "print(" in value:
                        #     raise Exception("You must not use print() statement, instead use st.write() to write to end user or observe(name, data) to view data yourself. Please regenerate the code")
                        exec(value, locals())
                        for key in self.st.session_state.keys():
                            if "observation:" in key:
                                observation=self.st.session_state[key]
                                observations.append((key.split(":")[1],observation))
                                if type(observation) is pd:
                                    # serialized_obs.append((key.split(":")[1],observation.to_json(orient='records', date_format='iso')))
                                    serialized_obs.append((key.split(":")[1],observation.to_string()))

                                elif type(observation) is not Figure:
                                    serialized_obs.append({key.split(":")[1]:str(observation)})
                                del self.st.session_state[key]
                    except Exception as e:
                        observations.append(("Error:",str(e)))
                        serialized_obs.append({"\nEncounter following error, can you try again?\n:":str(e)+"\nAction:"})
                        
                    for observation in observations:
                        st.write(observation[0])
                        st.write(observation[1])

                    obs = f"\nObservation on the first 10 rows of data: {serialized_obs}"
                    new_input += obs
                else:
                    st.write(key)
                    st.write(value)
                if "Answer" in key:
                    print("Answer is given, finish")
                    finish= True
            if show_prompt:
                self.st.write("Prompt")
                self.st.write(self.conversation_history)

            count +=1
            if count>= max_steps:
                print("Exceeding threshold, finish")
                break

    