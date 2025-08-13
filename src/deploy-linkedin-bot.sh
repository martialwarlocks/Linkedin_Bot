#!/bin/bash

# 🚀 One-Click LinkedIn Bot Deployment Script
# Save this as deploy-linkedin-bot.sh and run: chmod +x deploy-linkedin-bot.sh && ./deploy-linkedin-bot.sh

set -e

echo "🤖 LinkedIn Content Creator API - One-Click Deploy"
echo "=================================================="

# 📝 Configuration - UPDATE THESE VALUES!
PROJECT_ID=""  # ⚠️ CHANGE THIS!
GROQ_API_KEY=""     # ⚠️ CHANGE THIS!
OPENAI_API_KEY="" # ⚠️ CHANGE THIS!

# Check if user updated the configuration
if [[ "$PROJECT_ID" == "your-actual-project-id" ]]; then
    echo "❌ Please update the configuration in this script first!"
    echo "📝 Edit this file and set your:"
    echo "   - PROJECT_ID (your GCP project ID)"
    echo "   - GROQ_API_KEY (your Groq API key)"
    echo "   - OPENAI_API_KEY (your OpenAI API key)"
    exit 1
fi

# Constants
REGION="asia-south1"
SERVICE_NAME="linkedin-content-creator-api"
BUCKET_NAME="linkedin-bot"
IMAGE_NAME="gcr.io/$PROJECT_ID/linkedin-bot-api"

echo "🔧 Project: $PROJECT_ID"
echo "🌍 Region: $REGION"
echo "📦 Bucket: $BUCKET_NAME"
echo ""

# Set project
echo "⚙️ Setting up Google Cloud project..."
gcloud config set project $PROJECT_ID

# Enable APIs
echo "🔌 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com run.googleapis.com secretmanager.googleapis.com storage.googleapis.com

# Create bucket if needed
echo "🪣 Setting up storage bucket..."
if ! gsutil ls gs://$BUCKET_NAME > /dev/null 2>&1; then
    gsutil mb -l $REGION gs://$BUCKET_NAME
    echo "✅ Created bucket: $BUCKET_NAME"
else
    echo "✅ Bucket already exists: $BUCKET_NAME"
fi

# Store secrets
echo "🔐 Storing API keys as secrets..."
echo "$GROQ_API_KEY" | gcloud secrets create groq-api-key --data-file=- 2>/dev/null || \
echo "$GROQ_API_KEY" | gcloud secrets versions add groq-api-key --data-file=-

echo "$OPENAI_API_KEY" | gcloud secrets create openai-api-key --data-file=- 2>/dev/null || \
echo "$OPENAI_API_KEY" | gcloud secrets versions add openai-api-key --data-file=-

# Build and deploy
echo "🔨 Building container image..."
gcloud builds submit --tag $IMAGE_NAME

echo "🚀 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --set-secrets="GROQ_API_KEY=groq-api-key:latest,OPENAI_API_KEY=openai-api-key:latest" \
    --update-env-vars="GCS_BUCKET_NAME=$BUCKET_NAME,ENVIRONMENT=production" \
    --memory 2Gi \
    --cpu 1 \
    --timeout 300s \
    --max-instances 10

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)')

echo ""
echo "🎉 Deployment Complete!"
echo "================================"
echo "🌐 Your API URL: $SERVICE_URL"
echo ""
echo "🧪 Quick Test:"
echo "curl $SERVICE_URL/"
echo ""
echo "📝 Next Steps:"
echo "1. Update your React app's API_BASE_URL to: $SERVICE_URL"
echo "2. Test the API: curl $SERVICE_URL/linkedin/stats"
echo "3. Visit: $SERVICE_URL/docs for API documentation"
echo ""
echo "🔄 To redeploy later:"
echo "gcloud builds submit --tag $IMAGE_NAME && gcloud run deploy $SERVICE_NAME --image $IMAGE_NAME --region $REGION"