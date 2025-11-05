# Base image for Python 3.10 Lambda
FROM public.ecr.aws/lambda/python:3.10

# Copy function code
COPY app/ ${LAMBDA_TASK_ROOT}/

# Copy model
COPY model/ ${LAMBDA_TASK_ROOT}/model

# Install dependencies
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Command to run the Lambda handler
CMD ["main.lambda_handler"]
