# Use AWS Lambda Python 3.10 base image
FROM public.ecr.aws/lambda/python:3.10

# Set the working directory (Lambda uses /var/task)
WORKDIR /var/task

# Copy your application code
COPY app/ ${LAMBDA_TASK_ROOT}/

# Copy your trained model
COPY model/ ${LAMBDA_TASK_ROOT}/model

# Install Python dependencies
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Set the Lambda handler
CMD ["main.lambda_handler"]

