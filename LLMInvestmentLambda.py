# Amazon lambda function to propose an investment portfolio based on a simple liszt of rrules

import boto3
import json
import os

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Defaults
DEFAULT_MODEL_ID = "anthropic.claude-v2"
AWS_REGION = 'us-west-2'#replace by your region
ENDPOINT_URL = 'https://bedrock-runtime.us-west-2.amazonaws.com'#replace by your URL
DEFAULT_MAX_TOKENS = 256



# define Bedrock parameters if needed
model_kwargs = { #anthropic
            "max_tokens_to_sample": 512,
            "temperature": 0, 
            "top_k": 250, 
            "top_p": 1, 
            "stop_sequences": ["\n\nHuman:"] 
           }


# create LLM; with Bedrock Titan
bedrock = boto3.client('bedrock-runtime' , 'us-west-2', endpoint_url='https://bedrock-runtime.us-west-2.amazonaws.com')#replace by your URL
from langchain.llms.bedrock import Bedrock
sm_llm = Bedrock(
#    client=bedrock,
    model_id=DEFAULT_MODEL_ID,
    endpoint_url=ENDPOINT_URL,
    region_name=AWS_REGION,
    model_kwargs=model_kwargs
)


prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Question: I am 50 years old and have an income of less than 100'000 Francs and I am ready to take risk
Answer: Let us think step by step.
Answer: Let us think step by step. 
Step 1:determine the age, age is 50, Age is not greater than 50
Step 2: determine income, income is less than 100000 
Step 3: determine risks, ready to take risk
The answer is Portfolio 1. 
Question: I am 45 years old and have an income of 74000 Francs and I am not ready to take risk
Answer: Let us think step by step. 
Step 1: determine age, age is 45, Age is not greater than 50
Step 2: determine income, income is 74000, income is less than 100000, 
Step 3: determine risks, not ready to take risk. 
The answer is Portfolio 2
Question: I am 45 years old and have an income of more than 100'000 Francs and I am not ready to take risk
Answer: Let us think step by step.
Step 1: determine age, age is 45, Age is not greater than 50
Step 2: determine income, income is more than 100000
Step 3: determine risks, not ready to take risk. 
The answer is Portfolio 1
Question: I am 45 years old and have an income of 150'000 Francs and I like risk
Answer: Let us think step by step.
Step 1: determine age, age is 45, Age is not greater than 50
Step 2: determine income, income is 150000, income is more than 100000 
Step 3: determine risks, ready to take risk. 
The answer is Portfolio 1
Question: I am 52 years old and have an income of less than 100'000 Francs and I am not ready to take risk
Answer: Let us think step by step.
Step 1: determine age, age is 52, age is greater than 50
Step 2: determine income, income is less than 100000
Step 3: determine risks, not ready to take risk
The answer is Portfolio 2
Question: I was born in 1985 and have an income of more than 100'000 Francs and I am ready to take risk
Answer: Let us think step by step
Step 1: determine age, he is born in 1985, we are in 2023, so age is 38, so age is less than 50
Step 2: determine income, income is more than 100000
Step 3: determine risks, ready to take risk
The Answer is Portfolio 1
Question: I was born in 1965 and have an income of more than 100'000 Francs and I am ready to take risk
Answer: Let us think step by step.
Step 1 : determine age, he is born in 1965, we are in 2023, so age is 58, so age is greater  than 50
Step 2: determine income, income is more than 100000
Step 3: determine risks, ready to take risk
The Answer is Portfolio 1
Question: I was born in 1965 and have no income and I am ready to take risk
Answer: Let us think step by step
Step 1: determine age, he is born in 1965, we are in 2023, so age is 58, so age is greater  than 50
Step 2: determine income, income is 0, income is less than 100000
Step 3: determine risks, ready to take risk
The Answer is Portfolio 2
Question: I was born in 1965 and have an income of more than 100'000 Francs and I am ready to take risk
Answer: Let us think step by step.
Step 1: determine age, he is born in 1965, we are in 2023, so age is 58, so age is greater  than 50
Step 2: determine income, income is more than 100000
Step 3: determine risks, ready to take risk
The Answer is Portfolio 1
Question: I was born in 1997 and have an income of 45'000 Francs and I do not want to take risks
Answer: Let us think step by step.
Step 1: determine age, he is born in 1997, we are in 2023, so age is 25, so age is less than 50
Step 2: determine income, income is 45000, income is less than 100000
Step 3: determine risks, not ready to take risk
The Answer is Portfolio 2
Question: {question}
Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["question"]
)


def lambda_handler(event, context):

    intent_request = event
    input_transcript = intent_request['req']['question']
    intent_name = intent_request['req']['intentname']
    session_attributes = intent_request['req']['session']
    chain = LLMChain(llm=sm_llm, prompt=PROMPT)
    ans=chain.run({'question':input_transcript})
    ans=ans.replace('\n',' ')
    print(ans)
    intent_request['res']['message'] = ans
    intent_request['res']['type'] = "plaintext"
    return intent_request
