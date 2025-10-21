# LinkedIn Content Creator - Deployment Guide

## Overview
This is a LinkedIn Content Creator application with AI-powered content generation, document management, and research capabilities.

## Environment Variables

### Frontend (.env)
Create a `.env` file in the root directory with the following variables:

```env
# Backend Configuration
REACT_APP_BACKEND_URL=https://your-backend-url.com

# Supabase Configuration
REACT_APP_SUPABASE_URL=https://your-project.supabase.co
REACT_APP_SUPABASE_ANON_KEY=your-supabase-anon-key
```

### Backend (.env)
Create a `.env` file in the root directory with the following variables:

```env
# API Keys
GROQ_API_KEY=your-groq-api-key
OPENAI_API_KEY=your-openai-api-key
STABILITY_API_KEY=your-stability-api-key

# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-supabase-anon-key

# Google Cloud Storage
GCS_BUCKET_NAME=your-bucket-name
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
```

## Features

### Content Generation
- AI-powered LinkedIn post generation
- Multiple creator styles (Seth Godin, Simon Sinek, etc.)
- Customizable content options (tone, format, length, etc.)
- Video script generation
- Hashtag suggestions

### Content Customization
- **Settings Button**: Located in the bottom-right corner
- **Customization Options**:
  - Content Tone (Professional, Casual, Inspirational, etc.)
  - Content Format (Bullet Points, Numbered Lists, etc.)
  - Content Style (Data-driven, Storytelling, etc.)
  - Post Length (Short, Medium, Long)
  - Include Statistics (Yes/No)
  - Call to Action (Custom text)

### Document Management
- Upload PDF, DOCX, TXT files
- Automatic text extraction and chunking
- Vector database integration
- Document-based content generation

### Research Management
- Add research items manually
- Web crawling capabilities
- Research-based content generation
- Tag-based organization

## Deployment Steps

### 1. Backend Deployment
1. Set up your environment variables
2. Deploy to your preferred platform (Google Cloud Run, Heroku, etc.)
3. Ensure all API keys are properly configured

### 2. Frontend Deployment
1. Set up your environment variables
2. Build the React app: `npm run build`
3. Deploy to your preferred platform (Vercel, Netlify, etc.)

### 3. Database Setup
1. Set up a Supabase project
2. Run the database schema from `supabase-schema.sql`
3. Configure your Supabase credentials

## Configuration

### Backend URL
The frontend connects to the backend using the `REACT_APP_BACKEND_URL` environment variable. Make sure this points to your deployed backend.

### Supabase Database
The application uses Supabase for:
- Research items storage
- Creator style profiles
- Document metadata
- User history

### Google Cloud Storage
Used for storing uploaded documents and generated images.

## Security Notes

- All API keys are now environment-based
- No hardcoded credentials in the source code
- Supabase provides secure database access
- GCS provides secure file storage

## Customization

The application is designed to be easily customizable:
- Add new creator styles in the Supabase database
- Modify content generation prompts in `main.py`
- Customize the UI in `src/App.js`
- Add new content formats and styles

## Support

For issues or questions, please check the code comments and configuration files.



