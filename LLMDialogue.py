#lamdab function for general LLM dialogue

import boto3
import json
import os

# Defaults
DEFAULT_MODEL_ID = "anthropic.claude-v2"
AWS_REGION = 'us-west-2'
ENDPOINT_URL = "https://bedrock-runtime.us-west-2.amazonaws.com"
DEFAULT_MAX_TOKENS = 256



# define Bedrock parameters if needed
model_kwargs = {"temperature": 0, "maxTokenCount": 1000, "topP": 1}


from langchain.llms.bedrock import Bedrock



from langchain.chains import ConversationChain

# global variables - avoid creating a new client for every request
client = None


def get_client():
    print("Connecting to Bedrock Service: ", ENDPOINT_URL)
    client = boto3.client(service_name='bedrock-runtime', region_name=AWS_REGION, endpoint_url=ENDPOINT_URL)
    return client


def lambda_handler(event, context):
    global client
    intent_request = event
    input_transcript = intent_request['req']['question']
    intent_name = intent_request['req']['intentname']
    session_attributes = intent_request['req']['session']
    # create LLM
    if (client is None):
        client = get_client()
    llm = Bedrock(
#        client=client,
        model_id=DEFAULT_MODEL_ID,
        endpoint_url=ENDPOINT_URL,
        region_name=AWS_REGION
    #    model_kwargs=model_kwargs
    )

    conversation = ConversationChain(
        llm=llm, verbose=True
    )
    print('before predict')
    ans= conversation.predict(input=input_transcript)

    ans=ans.replace('\n',' ')
    print(ans)
    intent_request['res']['message'] = ans
    intent_request['res']['type'] = "plaintext"
    return intent_request
