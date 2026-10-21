# Lambda container image for arm64 (AWS Graviton). Build context = repo root.
FROM public.ecr.aws/lambda/python:3.12-arm64
COPY requirements.txt .
RUN pip install --no-cache-dir opencv-python-headless==5.0.0.93 numpy boto3
COPY inspect_agent ${LAMBDA_TASK_ROOT}/inspect_agent
CMD ["inspect_agent.lambda_handler.handler"]
