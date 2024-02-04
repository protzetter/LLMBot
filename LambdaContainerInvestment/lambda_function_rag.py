#lambda function for RAG base dialogue using OpenSearch vector database

import json
import os

# Defaults
DEFAULT_MODEL_ID = "anthropic.claude-v2"
AWS_REGION = "your region" #replace by your region
ENDPOINT_URL = "https://bedrock-runtime.us-west-2.amazonaws.com"
DEFAULT_MAX_TOKENS = 1000

model_kwargs = { #anthropic
            "max_tokens_to_sample": 512,
            "temperature": 0, 
            "top_k": 250, 
            "top_p": 1, 
            "stop_sequences": ["\n\nHuman:"] 
           }

# create LLM; with Bedrock
bedrock = boto3.client('bedrock-runtime' , 'us-west-2', endpoint_url=ENDPOINT_URL)
from langchain.llms.bedrock import Bedrock
llm = Bedrock(
    client=bedrock,
    model_id=DEFAULT_MODEL_ID,
    endpoint_url=ENDPOINT_URL,
    region_name=AWS_REGION,
    model_kwargs=model_kwargs
)

# import and create embeddings instance
from langchain.embeddings import BedrockEmbeddings
embeddings = BedrockEmbeddings(
    client=bedrock,
    model_id="amazon.titan-embed-text-v1",
    endpoint_url=ENDPOINT_URL,
    region_name=AWS_REGION
)

from opensearchpy import OpenSearch, RequestsHttpConnection
import boto3
from requests_aws4auth import AWS4Auth
service = "aoss"

credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, 'your region', service, session_token=credentials.token) #replace by your region where you have your open search vector store


from langchain.vectorstores import OpenSearchVectorSearch
store = OpenSearchVectorSearch(
     opensearch_url="replace_by_your_url", 
     index_name="your index",
     embedding_function=embeddings,
     http_auth=awsauth,
     use_ssl=True,
     verify_certs=True,
     connection_class=RequestsHttpConnection,
)


# define retriever instance and pass it store instance
retriever = store.as_retriever(search_kwargs={"k":1})

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Human: {question}
AI Assistant:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}


def lambda_handler(event, context):
    global llm
    global retriever
    # create langchain retrievalQA
    qa= RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs)
    intent_request = event
    input_transcript = intent_request['req']['question']
    intent_name = intent_request['req']['intentname']
    session_attributes = intent_request['req']['session']
    answer= qa({"query": input_transcript})
    ans=answer['result']
    ans=ans.replace('\n',' ')
    intent_request['res']['message'] = ans
    intent_request['res']['type'] = "plaintext"
    return intent_request


