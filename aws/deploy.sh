#!/usr/bin/env bash
# Prereqs: AWS CLI v2 + SAM CLI + Docker (with arm64 emulation or an arm64 host), `aws configure` done.
set -euo pipefail
cd "$(dirname "$0")"
sam build --use-container -t template.yaml
sam deploy --stack-name activeinspect --resolve-s3 --resolve-image-repos \
  --capabilities CAPABILITY_IAM --region "${AWS_REGION:-us-east-1}" \
  --parameter-overrides BedrockModelId="${BEDROCK_MODEL_ID:-}" --no-confirm-changeset
aws cloudformation describe-stacks --stack-name activeinspect --query "Stacks[0].Outputs" --output table
