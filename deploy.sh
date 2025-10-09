#!/bin/bash

# Deploy transcription-agent to Azure Container Apps
# Run this from Azure Cloud Shell after cloning the repository

set -e  # Exit on error

# Configuration
RESOURCE_GROUP="stokai-tk"
ENVIRONMENT="stok-ai"
APP_NAME="transcription-agent"
LOCATION="eastus2"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "Dockerfile" ]; then
    log_error "Dockerfile not found. Please run this script from the transcription-agent directory."
    exit 1
fi

log_info "Starting deployment of $APP_NAME to Azure Container Apps..."

# Prompt for secrets if not set
if [ -z "$OPENAI_API_KEY" ]; then
    read -p "Enter OpenAI API Key: " OPENAI_API_KEY
fi

# Get the URLs of other services (optional, can be updated later)
if [ -z "$MEETING_AGENT_URL" ]; then
    MEETING_AGENT_URL=$(az containerapp show --name meeting-agent --resource-group "$RESOURCE_GROUP" --query "properties.configuration.ingress.fqdn" -o tsv 2>/dev/null || echo "")
    if [ -n "$MEETING_AGENT_URL" ]; then
        MEETING_AGENT_URL="https://${MEETING_AGENT_URL}"
    fi
fi

if [ -z "$PARSING_AGENT_URL" ]; then
    PARSING_AGENT_URL=$(az containerapp show --name parsing-agent --resource-group "$RESOURCE_GROUP" --query "properties.configuration.ingress.fqdn" -o tsv 2>/dev/null || echo "")
    if [ -n "$PARSING_AGENT_URL" ]; then
        PARSING_AGENT_URL="https://${PARSING_AGENT_URL}"
    fi
fi

log_info "Deploying $APP_NAME using 'az containerapp up'..."

# Deploy using az containerapp up (handles ACR creation, build, and deployment)
az containerapp up \
    --name "$APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --environment "$ENVIRONMENT" \
    --location "$LOCATION" \
    --source . \
    --target-port 8002 \
    --ingress external \
    --cpu 2.0 \
    --memory 4.0Gi \
    --env-vars \
        "OPENAI_API_KEY=$OPENAI_API_KEY" \
        "STT_PROVIDER=local" \
        "STT_MODEL=small" \
        "STT_DEVICE=cpu" \
        "STT_COMPUTE_TYPE=int8" \
        "ENABLE_DIARIZATION=1" \
        "DIARIZATION_MODE=light" \
        "MEETING_AGENT_URL=${MEETING_AGENT_URL}" \
        "PARSING_AGENT_URL=${PARSING_AGENT_URL}" \
        "STORAGE_DIR=/app/artifacts"

# Get the URL
APP_URL=$(az containerapp show \
    --name "$APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --query "properties.configuration.ingress.fqdn" -o tsv)

log_info "Deployment complete!"
log_info "App URL: https://$APP_URL"
log_info "Health check: https://$APP_URL/health"

echo ""
echo "Test the deployment with:"
echo "  curl https://$APP_URL/health"
