#!/bin/bash

echo "🚀 Starting LinkedIn Bot Backend..."

# Check if we're in the right directory
if [ ! -f "backend/server.js" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    echo "   Current directory: $(pwd)"
    echo "   Expected: linkedin-content-creator/"
    exit 1
fi

# Navigate to backend directory
cd backend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚙️ Creating .env file from template..."
    cp env.example .env
    echo "⚠️  Please edit backend/.env with your Google Cloud settings:"
    echo "   - GOOGLE_CLOUD_PROJECT_ID=your-project-id"
    echo "   - GOOGLE_CLOUD_KEY_FILE=./google-cloud-key.json"
    echo "   - LINKEDIN_BOT_BUCKET=linkedin-bot-documents"
    echo ""
    echo "Press Enter to continue anyway, or Ctrl+C to stop and configure..."
    read
fi

# Check if Google Cloud key exists
if [ ! -f "google-cloud-key.json" ]; then
    echo "⚠️  Warning: google-cloud-key.json not found"
    echo "   You need to download your Google Cloud service account key"
    echo "   and place it in backend/google-cloud-key.json"
    echo ""
    echo "Press Enter to continue anyway, or Ctrl+C to stop and configure..."
    read
fi

echo "🎯 Starting backend server on port 8000..."
echo "   Backend URL: http://localhost:8000"
echo "   Health Check: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
npm start 