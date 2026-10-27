#!/usr/bin/env bash
# Deploy the API to AWS with Amazon ECS Express Mode (ECR image → Fargate + ALB + HTTPS URL).
#
# AWS App Runner stopped accepting new customers on 30 April 2026; AWS points new
# users to ECS Express Mode instead, which is what this script uses.
#
# Prerequisites (one-time, see docs/DEPLOYMENT.md §AWS):
#   * AWS CLI v2, logged in (aws sts get-caller-identity works)
#   * IAM roles: ecsTaskExecutionRole (AmazonECSTaskExecutionRolePolicy) and an
#     ECS infrastructure role for Express Mode
#
# Usage (from the repo root):
#   export AWS_REGION=us-east-1 AWS_ACCOUNT_ID=123456789012
#   export EXECUTION_ROLE_ARN=arn:aws:iam::$AWS_ACCOUNT_ID:role/ecsTaskExecutionRole
#   export INFRA_ROLE_ARN=arn:aws:iam::$AWS_ACCOUNT_ID:role/ecsInfrastructureRoleForExpressServices
#   ./deploy/aws/deploy_ecs_express.sh            # first deploy: creates the service
#   export SERVICE_ARN=arn:aws:ecs:...            # the ARN printed by the first run
#   ./deploy/aws/deploy_ecs_express.sh            # later deploys: push a new image + update
set -euo pipefail

: "${AWS_REGION:?set AWS_REGION}" "${AWS_ACCOUNT_ID:?set AWS_ACCOUNT_ID}"
: "${EXECUTION_ROLE_ARN:?set EXECUTION_ROLE_ARN}" "${INFRA_ROLE_ARN:?set INFRA_ROLE_ARN}"
SERVICE=${SERVICE:-applicant-screening-api}
REPO=${REPO:-screening-api}
TAG=${TAG:-$(git rev-parse --short HEAD)}
REGISTRY="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
IMAGE="$REGISTRY/$REPO:$TAG"

echo "==> ECR repository $REPO"
aws ecr describe-repositories --repository-names "$REPO" --region "$AWS_REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$REPO" --region "$AWS_REGION" >/dev/null

echo "==> Build and push $IMAGE"
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$REGISTRY"
# Fargate defaults to x86_64; build for it even on an Apple-silicon laptop.
docker build --platform linux/amd64 -f serving/Dockerfile --build-arg GIT_SHA="$TAG" -t "$IMAGE" .
docker push "$IMAGE"

ENV_VARS="[{name=SCREENING_ENVIRONMENT,value=aws},{name=SCREENING_LOG_PREDICTIONS_STDOUT,value=true}]"
CONTAINER="image=$IMAGE,containerPort=8000,environment=$ENV_VARS"

if [[ -n "${SERVICE_ARN:-}" ]]; then
  echo "==> Updating $SERVICE_ARN"
  aws ecs update-express-gateway-service --service-arn "$SERVICE_ARN" --region "$AWS_REGION" \
    --primary-container "$CONTAINER"
else
  echo "==> Creating $SERVICE (1 vCPU / 2 GB, 1-3 tasks)"
  aws ecs create-express-gateway-service --region "$AWS_REGION" \
    --service-name "$SERVICE" \
    --execution-role-arn "$EXECUTION_ROLE_ARN" \
    --infrastructure-role-arn "$INFRA_ROLE_ARN" \
    --health-check-path /ready \
    --primary-container "$CONTAINER" \
    --cpu 1024 --memory 2048 \
    --scaling-target "minTaskCount=1,maxTaskCount=3,autoScalingMetric=AVERAGE_CPU,autoScalingTargetValue=60"
fi
echo "Done. The service URL is shown in the ECS console (Express services) once the first task is healthy."
echo "Then run:  python serving/scripts/smoke_test.py --url https://<service-url>"
