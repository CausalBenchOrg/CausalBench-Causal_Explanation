FROM public.ecr.aws/lambda/python:3.12

# Set working directory (optional but cleaner)
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy requirements and install early to leverage Docker layer caching
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code after installing requirements
COPY . .

# Set the Lambda function handler
CMD ["lambda_function.handler"]