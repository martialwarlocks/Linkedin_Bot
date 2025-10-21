#!/bin/bash
set -e

# Configuration
export PROJECT_ID="linkedin-bot-468005"
export REGION="us-central1"
export SERVICE_NAME="linkedin-content-creator-api"
export IMAGE_NAME="linkedin-content-creator"
export REPO="cloud-run"

echo "🚀 Deploying LinkedIn Content Creator API to Cloud Run"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Service: $SERVICE_NAME"

# Set project
echo "📋 Setting GCP project..."
gcloud config set project "$PROJECT_ID"

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com

# Create Artifact Registry repo if it doesn't exist
echo "📦 Creating Artifact Registry repository..."
gcloud artifacts repositories create "$REPO" --repository-format=docker --location="$REGION" --description="Cloud Run images" 2>/dev/null || echo "Repository already exists"

# Build and push the image
echo "🏗️ Building and pushing Docker image..."
export TAG="$(date +%Y%m%d-%H%M%S)"
export ARTIFACT_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${TAG}"

gcloud builds submit --tag "$ARTIFACT_URI" .

echo "✅ Image built and pushed: $ARTIFACT_URI"

# Create or update the secret for service account credentials
echo "🔐 Setting up service account credentials..."
gcloud secrets create gcp-sa-key --replication-policy=automatic 2>/dev/null || echo "Secret already exists"

# Check if service account key file exists
SA_KEY_FILE="/Users/yatins/Downloads/linkedin-bot-468005-key.json"
if [ ! -f "$SA_KEY_FILE" ]; then
    echo "⚠️  Warning: Service account key file not found at $SA_KEY_FILE"
    echo "Please ensure you have the correct service account key file for project $PROJECT_ID"
    echo "You can download it from: https://console.cloud.google.com/iam-admin/serviceaccounts"
    exit 1
fi

# Upload the service account key to Secret Manager
echo "📤 Uploading service account key to Secret Manager..."
gcloud secrets versions add gcp-sa-key --data-file="$SA_KEY_FILE"

# Get project number and runtime service account
export PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
export RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

echo "🔑 Runtime service account: $RUNTIME_SA"

# Grant permissions
echo "🔓 Granting permissions..."
gcloud secrets add-iam-policy-binding gcp-sa-key \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/secretmanager.secretAccessor"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/storage.admin"

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
  --image "$ARTIFACT_URI" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --service-account="${RUNTIME_SA}" \
  --set-secrets=GOOGLE_APPLICATION_CREDENTIALS_JSON=gcp-sa-key:latest \
  --set-env-vars=GCS_BUCKET_NAME=linkedin-bot-documents \
  --cpu=1 \
  --memory=1Gi \
  --max-instances=10 \
  --port=8080

# Get service URL
SERVICE_URL="$(gcloud run services describe "$SERVICE_NAME" --region "$REGION" --format='value(status.url)')"

echo ""
echo "🎉 Deployment completed successfully!"
echo "Service URL: $SERVICE_URL"
echo ""

# Test the service
echo "🧪 Testing the service..."
curl -sS "${SERVICE_URL}/" | cat

echo ""
echo "✅ Service is running and responding!"
echo "🌐 You can now access your API at: $SERVICE_URL" 