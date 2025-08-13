# LinkedIn Content Creator - Setup Guide

## System Architecture

The LinkedIn Content Creator is a full-stack AI-powered application with the following architecture:

### Frontend (React.js)
- **Technology**: React 19.1.0 with Tailwind CSS
- **Purpose**: User interface for content creation, research management, and document processing
- **Key Features**:
  - Content generation interface
  - Research database management
  - Document upload and processing
  - Web crawling interface
  - Chat interface with AI
  - Creator profiles management

### Backend (FastAPI)
- **Technology**: Python FastAPI with multiple AI/ML libraries
- **Purpose**: AI content generation, document processing, and data management
- **Key Components**:
  - **Enhanced FastAPI Backend**: Main backend with advanced features
  - **Document Processor**: Handles PDF, DOCX, PPTX, and other document formats
  - **AI Generator**: OpenAI-powered content generation
  - **Web Crawler**: Automated website content extraction
  - **Vector Database**: ChromaDB for semantic search
  - **Cloud Storage**: Google Cloud Storage integration

### Database & Storage
- **Supabase**: PostgreSQL database for user data and research
- **ChromaDB**: Vector database for semantic search
- **Google Cloud Storage**: Document storage
- **Redis**: Caching and background task management

### AI/ML Stack
- **OpenAI GPT**: Content generation
- **Sentence Transformers**: Text embeddings
- **FAISS**: Vector similarity search
- **Trafilatura**: Web content extraction
- **BeautifulSoup**: HTML parsing

## Prerequisites

### System Requirements
- **Node.js**: 18.x or higher
- **Python**: 3.9 or higher
- **Git**: Latest version
- **Docker**: (Optional, for containerized deployment)

### API Keys & Services
- **OpenAI API Key**: For content generation
- **Supabase Account**: Database and authentication
- **Google Cloud Storage**: Document storage (optional)
- **Groq API Key**: Alternative AI provider (optional)

## Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd linkedin-content-creator
```

### 2. Frontend Setup

#### Install Node.js Dependencies
```bash
npm install
```

#### Environment Configuration
Create a `.env` file in the root directory:
```env
REACT_APP_BACKEND_URL=http://localhost:8000
REACT_APP_SUPABASE_URL=your_supabase_url
REACT_APP_SUPABASE_ANON_KEY=your_supabase_anon_key
```

### 3. Backend Setup

#### Create Python Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Install Python Dependencies
```bash
# For enhanced backend
cd enhanced-fastapi-backend
pip install -r requirements.txt

# For local development
cd ..
pip install -r local-requirements.txt
```

#### Environment Configuration
Create a `.env` file in the `enhanced-fastapi-backend` directory:
```env
OPENAI_API_KEY=your_openai_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_service_key
GOOGLE_APPLICATION_CREDENTIALS=path_to_your_gcs_credentials.json
GCS_BUCKET_NAME=your_bucket_name
GCS_PROJECT_ID=your_project_id
```

### 4. Database Setup

#### Supabase Configuration
1. Create a new Supabase project
2. Run the SQL schema from `supabase-schema.sql`
3. Update the configuration in `backend-config.json`

#### Vector Database
The ChromaDB will be automatically initialized when the backend starts.

## Running the Application

### Development Mode

#### Start the Backend
```bash
# Option 1: Enhanced FastAPI Backend
cd enhanced-fastapi-backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Option 2: Local FastAPI Server
cd ..
python local-fastapi-server.py
```

#### Start the Frontend
```bash
# In a new terminal
npm start
```

The application will be available at:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### Production Mode

#### Build Frontend
```bash
npm run build
```

#### Deploy Backend
```bash
# Using Docker
cd enhanced-fastapi-backend
docker build -t linkedin-content-creator .
docker run -p 8000:8000 linkedin-content-creator

# Or using uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Core Endpoints
- `GET /` - Health check
- `POST /linkedin/generate` - Generate content
- `POST /linkedin/upload-document` - Upload documents
- `POST /linkedin/crawl` - Crawl websites
- `GET /linkedin/research` - Get research data
- `POST /linkedin/research` - Add research
- `GET /linkedin/documents` - Get uploaded documents
- `GET /linkedin/crawls` - Get web crawls
- `POST /linkedin/search` - Semantic search

### Content Generation
- `POST /linkedin/generate` - Generate LinkedIn posts, video scripts, hashtags
- `POST /linkedin/research-summary` - Generate research summaries
- `POST /linkedin/document-insights` - Extract insights from documents

## Features

### Content Generation
- **LinkedIn Posts**: Professional posts with engagement optimization
- **Video Scripts**: YouTube/TikTok style video content
- **Hashtags**: Trending and relevant hashtag suggestions
- **Engagement Tips**: Strategies to increase engagement

### Research Management
- **Document Upload**: Support for PDF, DOCX, PPTX, TXT files
- **Web Crawling**: Automated content extraction from websites
- **Semantic Search**: AI-powered research discovery
- **Research Database**: Organized storage and categorization

### AI Capabilities
- **Multi-Modal Processing**: Text, documents, and web content
- **Context-Aware Generation**: Uses research and documents as context
- **Creator Profiles**: Customizable AI personalities
- **Smart Question Answering**: Chat interface with research context

## Troubleshooting

### Common Issues

#### Backend Connection Issues
- Check if the backend is running on port 8000
- Verify CORS settings in the backend
- Check API key configurations

#### Document Upload Issues
- Ensure file size limits are appropriate
- Check Google Cloud Storage credentials
- Verify file format support

#### AI Generation Issues
- Verify OpenAI API key and quota
- Check internet connectivity
- Review error logs in the browser console

### Logs and Debugging
- Backend logs: Check terminal output
- Frontend logs: Browser developer tools
- API testing: Use http://localhost:8000/docs

## Security Considerations

- Store API keys in environment variables
- Use HTTPS in production
- Implement proper authentication
- Regular security updates
- Monitor API usage and costs

## Performance Optimization

- Use Redis for caching
- Implement background tasks with Celery
- Optimize vector database queries
- Use CDN for static assets
- Monitor memory usage

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section
- Review API documentation
- Check GitHub issues
- Contact the development team 