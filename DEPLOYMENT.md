# LinkedIn Bot Cloud Document Management - Deployment Guide

## 🚀 Overview

This guide will help you deploy the LinkedIn Bot Document Management system using Google Cloud Storage, completely separate from SpikedAI's infrastructure.

## 📋 Prerequisites

1. **Google Cloud Account** with billing enabled
2. **Node.js** (v16 or higher)
3. **npm** or **yarn**
4. **Google Cloud CLI** (optional, for local development)

## 🔧 Setup Steps

### 1. Google Cloud Setup

#### Create a New Project
```bash
# Create a new Google Cloud project
gcloud projects create linkedin-bot-docs --name="LinkedIn Bot Documents"

# Set the project as default
gcloud config set project linkedin-bot-docs
```

#### Enable Required APIs
```bash
# Enable Cloud Storage API
gcloud services enable storage.googleapis.com

# Enable Cloud Build API (for deployment)
gcloud services enable cloudbuild.googleapis.com

# Enable Cloud Run API (for serverless deployment)
gcloud services enable run.googleapis.com
```

#### Create Service Account
```bash
# Create service account
gcloud iam service-accounts create linkedin-bot-docs-sa \
    --display-name="LinkedIn Bot Document Manager"

# Get the service account email
SA_EMAIL=$(gcloud iam service-accounts list --filter="displayName:LinkedIn Bot Document Manager" --format="value(email)")

# Grant Storage Admin role
gcloud projects add-iam-policy-binding linkedin-bot-docs \
    --member="serviceAccount:$SA_EMAIL" \
    --role="roles/storage.admin"

# Create and download key file
gcloud iam service-accounts keys create google-cloud-key.json \
    --iam-account=$SA_EMAIL
```

### 2. Backend Setup

#### Install Dependencies
```bash
cd backend
npm install
```

#### Configure Environment
```bash
# Copy environment example
cp env.example .env

# Edit .env file with your values
nano .env
```

**Required Environment Variables:**
```env
GOOGLE_CLOUD_PROJECT_ID=linkedin-bot-docs
GOOGLE_CLOUD_KEY_FILE=./google-cloud-key.json
LINKEDIN_BOT_BUCKET=linkedin-bot-documents
PORT=3001
NODE_ENV=production
```

#### Test Backend Locally
```bash
# Start the backend server
npm run dev

# Test the health endpoint
curl http://localhost:3001/health
```

### 3. Frontend Configuration

#### Update API URL
In `src/App.js`, update the `CLOUD_DOCUMENT_API` constant:

```javascript
// For local development
const CLOUD_DOCUMENT_API = 'http://localhost:3001';

// For production (after deployment)
const CLOUD_DOCUMENT_API = 'https://your-backend-url.com';
```

### 4. Deploy Backend to Google Cloud Run

#### Create Dockerfile
```dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .

EXPOSE 3001

CMD ["npm", "start"]
```

#### Deploy to Cloud Run
```bash
# Build and deploy
gcloud run deploy linkedin-bot-docs-backend \
    --source . \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --set-env-vars="GOOGLE_CLOUD_PROJECT_ID=linkedin-bot-docs,LINKEDIN_BOT_BUCKET=linkedin-bot-documents,NODE_ENV=production" \
    --set-secrets="GOOGLE_CLOUD_KEY_FILE=google-cloud-key:latest"

# Get the deployed URL
gcloud run services describe linkedin-bot-docs-backend \
    --platform managed \
    --region us-central1 \
    --format="value(status.url)"
```

#### Store Service Account Key as Secret
```bash
# Create secret
gcloud secrets create google-cloud-key --data-file=google-cloud-key.json

# Grant access to Cloud Run
gcloud secrets add-iam-policy-binding google-cloud-key \
    --member="serviceAccount:linkedin-bot-docs@linkedin-bot-docs.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

### 5. Update Frontend with Production URL

After deployment, update the frontend:

```javascript
// In src/App.js
const CLOUD_DOCUMENT_API = 'https://your-cloud-run-url.com';
```

## 🔒 Security Considerations

### 1. CORS Configuration
Update the backend to only allow your frontend domain:

```javascript
// In backend/server.js
app.use(cors({
  origin: ['http://localhost:3000', 'https://your-frontend-domain.com'],
  credentials: true
}));
```

### 2. File Size Limits
The backend is configured with a 10MB file size limit. Adjust as needed:

```javascript
// In backend/server.js
const upload = multer({
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB
  }
});
```

### 3. Rate Limiting
Consider adding rate limiting for production:

```bash
npm install express-rate-limit
```

```javascript
const rateLimit = require('express-rate-limit');

const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100 // limit each IP to 100 requests per windowMs
});

app.use('/upload', limiter);
```

## 📊 Monitoring & Logging

### 1. Enable Cloud Logging
```bash
# Enable Cloud Logging API
gcloud services enable logging.googleapis.com
```

### 2. Add Structured Logging
```javascript
// In backend/server.js
const winston = require('winston');

const logger = winston.createLogger({
  level: 'info',
  format: winston.format.json(),
  transports: [
    new winston.transports.Console()
  ]
});
```

## 🔄 CI/CD Pipeline

### GitHub Actions Example
```yaml
# .github/workflows/deploy.yml
name: Deploy Backend

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup Google Cloud CLI
        uses: google-github-actions/setup-gcloud@v0
        with:
          project_id: ${{ secrets.GCP_PROJECT_ID }}
          service_account_key: ${{ secrets.GCP_SA_KEY }}
      
      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy linkedin-bot-docs-backend \
            --source . \
            --platform managed \
            --region us-central1 \
            --allow-unauthenticated
```

## 💰 Cost Optimization

### 1. Storage Classes
Consider using different storage classes for cost optimization:

```javascript
// Use cheaper storage for older documents
const blob = bucket.file(fileName);
await blob.save(file.buffer, {
  metadata: {
    storageClass: 'NEARLINE' // Cheaper for infrequently accessed files
  }
});
```

### 2. Lifecycle Management
Set up automatic deletion of old documents:

```bash
# Create lifecycle policy
gsutil lifecycle set lifecycle.json gs://linkedin-bot-documents
```

```json
// lifecycle.json
{
  "rule": [
    {
      "action": {"type": "Delete"},
      "condition": {
        "age": 365,
        "isLive": true
      }
    }
  ]
}
```

## 🚨 Troubleshooting

### Common Issues

1. **CORS Errors**: Ensure CORS is properly configured for your frontend domain
2. **Authentication Errors**: Verify service account permissions and key file
3. **File Upload Failures**: Check file size limits and supported file types
4. **Storage Quotas**: Monitor Google Cloud Storage quotas and limits

### Debug Commands
```bash
# Check backend logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=linkedin-bot-docs-backend"

# Test bucket access
gsutil ls gs://linkedin-bot-documents

# Check service account permissions
gcloud projects get-iam-policy linkedin-bot-docs
```

## 📈 Scaling Considerations

1. **Auto-scaling**: Cloud Run automatically scales based on demand
2. **CDN**: Consider using Cloud CDN for faster document access
3. **Database**: For large-scale deployments, consider adding a database for metadata
4. **Caching**: Implement Redis for caching frequently accessed documents

## 🎯 Next Steps

1. **Deploy the backend** using the steps above
2. **Update the frontend** with the production API URL
3. **Test the complete system** with file uploads and content generation
4. **Monitor performance** and adjust as needed
5. **Set up monitoring** and alerting for production use

Your LinkedIn Bot now has a completely separate, scalable document management system! 🚀 