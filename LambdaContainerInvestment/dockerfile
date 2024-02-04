FROM public.ecr.aws/lambda/python:3.11
# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

RUN yum update -y && \
  yum install -y git && \
  yum install -y curl && \
  rm -Rf /var/cache/yum

# Install the specified packages
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy function code
COPY lambda_function_rag.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "lambda_function_rag.lambda_handler" ]
