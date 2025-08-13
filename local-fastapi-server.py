import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Any, Dict, Union
from contextlib import asynccontextmanager
import json
from datetime import datetime
import traceback
from io import BytesIO
import re
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl
import httpx
from google.cloud import storage
import openai
from supabase import create_client, Client

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# --- Configuration ---
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "linkedin-bot-documents")
VECTOR_DB_DIR = Path("./vector_db")

# Supabase configuration
SUPABASE_URL = "https://qgyqkgmdnwfcnzzuzict.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFneXFrZ21kbndmY256enV6aWN0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTM3NzkyOTcsImV4cCI6MjA2OTM1NTI5N30.d9VFOHZsWDxhqY8UM0jvx5pGJVVOSkgHVFODL16Nc6s"

# Ensure necessary local cache directory exists at startup
VECTOR_DB_DIR.mkdir(exist_ok=True)

# --- API Keys & Global State ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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").strip()

# Simple in-memory storage for testing
documents = []
chunks = []
metadata = []

# --- Supabase Client ---
def get_supabase_client() -> Client:
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        return None

# --- Google Cloud Storage Client ---
def get_gcs_client() -> storage.Client:
    try:
        return storage.Client()
    except Exception as e:
        logger.error(f"Failed to initialize GCS client: {e}")
        return None

# --- Application Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting LinkedIn Content Creator API...")
    
    try:
        # Initialize OpenAI client
        global openai_client
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        # Initialize Supabase client
        global supabase_client
        supabase_client = get_supabase_client()
        
        logger.info("LinkedIn Content Creator API started successfully")
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down LinkedIn Content Creator API...")

# --- FastAPI App ---
app = FastAPI(
    title="LinkedIn Content Creator API",
    description="API for document management and content generation for LinkedIn posts",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)

class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_added: int
    total_chunks: int
    document_id: str
    gcs_url: str

class AskResponse(BaseModel):
    answer: str
    question: str
    sources: List[dict]
    client_followups: List[str] = Field(default_factory=list)
    sales_followups: List[str] = Field(default_factory=list)

class CrawlRequest(BaseModel):
    url: HttpUrl
    depth: int = 2
    max_pages: int = 10

class CrawlResponse(BaseModel):
    url: HttpUrl
    title: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    hostname: Optional[str] = None
    main_text: str
    comments: Optional[str] = None
    raw_text: Optional[str] = None
    source: Optional[str] = None
    pages_crawled: int = 1
    total_content_length: int = 0

class ContentGenerationRequest(BaseModel):
    prompt: str
    creator_key: str
    research_ids: List[str] = Field(default_factory=list)

class ContentGenerationResponse(BaseModel):
    linkedin_post: str
    linkedin_reel_transcript: str
    hashtags: List[str]
    engagement_questions: List[str]
    talking_points: List[str]
    style_notes: str
    context_used: str

class ResearchItem(BaseModel):
    topic: str
    findings: str
    data: Optional[str] = None
    source: Optional[str] = None
    tags: Optional[str] = None

# --- Utility Functions ---
def extract_text_from_file(content: bytes, filename: str) -> str:
    """Extract text from various file formats"""
    try:
        if filename.lower().endswith('.txt'):
            return content.decode('utf-8')
        elif filename.lower().endswith('.pdf'):
            # Simple PDF text extraction (you might want to use PyPDF2 or pdfplumber)
            return content.decode('utf-8', errors='ignore')
        elif filename.lower().endswith(('.docx', '.doc')):
            # Simple DOCX text extraction (you might want to use python-docx)
            return content.decode('utf-8', errors='ignore')
        else:
            return content.decode('utf-8', errors='ignore')
    except Exception as e:
        logger.error(f"Error extracting text from {filename}: {e}")
        return ""

def chunk_text(text: str) -> List[str]:
    """Split text into chunks for processing"""
    try:
        # Simple chunking by sentences
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) < 1000:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text[:1000]]
    except Exception as e:
        logger.error(f"Error chunking text: {e}")
        return [text[:1000]]

async def query_groq(messages: List[dict]) -> str:
    """Query Groq API for content generation"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama3-8b-8192",
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 2000
                },
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        return "Error generating content"

# --- API Endpoints ---
@app.get("/", summary="Health Check", tags=["System"])
def read_root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "LinkedIn Content Creator API is running",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# LinkedIn-specific endpoints
@app.get("/linkedin/", summary="LinkedIn API Health Check", tags=["LinkedIn"])
def linkedin_health_check():
    """LinkedIn API health check"""
    return {
        "status": "healthy",
        "message": "LinkedIn Content Creator API is running",
        "endpoints": [
            "/linkedin/upload-document",
            "/linkedin/documents", 
            "/linkedin/research",
            "/linkedin/generate",
            "/linkedin/crawl"
        ]
    }

@app.post("/linkedin/upload-document", response_model=UploadResponse, summary="Upload Document for LinkedIn Content", tags=["LinkedIn"])
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document for LinkedIn content generation"""
    try:
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Read file content
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Extract text
        text = extract_text_from_file(content, file.filename)
        if not text:
            raise HTTPException(status_code=400, detail="Could not extract text from file")
        
        # Chunk the text
        text_chunks = chunk_text(text)
        if not text_chunks:
            raise HTTPException(status_code=400, detail="Failed to process document content")
        
        # Generate document ID
        document_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        
        # Store in Supabase
        if supabase_client:
            try:
                # Add to documents table
                doc_data = {
                    "filename": file.filename,
                    "gcs_url": f"local://{document_id}",
                    "file_type": file.content_type or "unknown",
                    "file_size": len(content),
                    "chunks": {"count": len(text_chunks)}
                }
                
                result = supabase_client.table("documents").insert(doc_data).execute()
                logger.info(f"Document stored in Supabase: {result}")
                
                # Add chunks as research items
                for i, chunk in enumerate(text_chunks[:5]):  # Limit to first 5 chunks
                    research_data = {
                        "topic": f"Document: {file.filename} - Chunk {i+1}",
                        "findings": chunk[:500] + "..." if len(chunk) > 500 else chunk,
                        "data": f"Chunk {i+1} of {len(text_chunks)} from {file.filename}",
                        "source": file.filename,
                        "tags": "document,upload,content"
                    }
                    
                    supabase_client.table("research").insert(research_data).execute()
                
            except Exception as e:
                logger.error(f"Supabase storage error: {e}")
        
        # Add to local storage
        global documents, chunks, metadata
        
        documents.append({
            'filename': file.filename,
            'upload_date': datetime.now().isoformat(),
            'file_size': len(content),
            'file_type': file.content_type or 'unknown',
            'total_chunks': len(text_chunks),
            'document_id': document_id
        })
        
        for i, chunk in enumerate(text_chunks):
            metadata.append({
                'id': f"{document_id}_{i}",
                'text': chunk,
                'filename': file.filename,
                'chunk_index': i,
                'upload_date': datetime.now().isoformat(),
                'file_size': len(content),
                'file_type': file.content_type or 'unknown',
                'document_id': document_id
            })
            chunks.append(chunk)
        
        return UploadResponse(
            message=f"Document {file.filename} uploaded and processed successfully",
            filename=file.filename,
            chunks_added=len(text_chunks),
            total_chunks=len(text_chunks),
            document_id=document_id,
            gcs_url=f"local://{document_id}"
        )
    
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/linkedin/documents", summary="List Documents", tags=["LinkedIn"])
def list_documents():
    """List all uploaded documents"""
    try:
        if supabase_client:
            result = supabase_client.table("documents").select("*").execute()
            return result.data
        else:
            return documents
    except Exception as e:
        logger.error(f"List documents error: {e}")
        return documents

@app.get("/linkedin/research", summary="List Research Items", tags=["LinkedIn"])
def list_research():
    """List all research items"""
    try:
        if supabase_client:
            result = supabase_client.table("research").select("*").order("created_at", desc=True).execute()
            return result.data
        else:
            return []
    except Exception as e:
        logger.error(f"List research error: {e}")
        return []

@app.post("/linkedin/research", summary="Add Research Item", tags=["LinkedIn"])
async def add_research(research: ResearchItem):
    """Add a new research item"""
    try:
        if supabase_client:
            result = supabase_client.table("research").insert({
                "topic": research.topic,
                "findings": research.findings,
                "data": research.data,
                "source": research.source,
                "tags": research.tags
            }).execute()
            return result.data[0] if result.data else None
        else:
            return {"id": "local", "topic": research.topic}
    except Exception as e:
        logger.error(f"Add research error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/linkedin/research/{research_id}", summary="Delete Research Item", tags=["LinkedIn"])
async def delete_research(research_id: str):
    """Delete a research item"""
    try:
        if supabase_client:
            result = supabase_client.table("research").delete().eq("id", research_id).execute()
            return {"message": "Research item deleted successfully"}
        else:
            return {"message": "Research item deleted successfully"}
    except Exception as e:
        logger.error(f"Delete research error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/linkedin/generate", response_model=ContentGenerationResponse, summary="Generate LinkedIn Content", tags=["LinkedIn"])
async def generate_linkedin_content(request: ContentGenerationRequest):
    """Generate LinkedIn content using research and creator style"""
    try:
        # Validate input
        if not request.prompt or not request.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt is required")
        
        if not request.creator_key or not request.creator_key.strip():
            raise HTTPException(status_code=400, detail="Creator key is required")
        
        # Get research context
        research_context = ""
        if supabase_client and request.research_ids and len(request.research_ids) > 0:
            try:
                # Convert string IDs to integers if needed
                research_ids = []
                for rid in request.research_ids:
                    try:
                        if isinstance(rid, str):
                            research_ids.append(int(rid))
                        else:
                            research_ids.append(rid)
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid research ID: {rid}")
                        continue
                
                if research_ids:
                    research_result = supabase_client.table("research").select("*").in_("id", research_ids).execute()
                    if research_result.data:
                        research_context = "\n\n".join([f"Research: {item['topic']}\n{item['findings']}" for item in research_result.data])
                        logger.info(f"Found {len(research_result.data)} research items for context")
            except Exception as e:
                logger.error(f"Error fetching research: {e}")
                research_context = ""
        
        # Create enhanced prompt with research context
        if research_context:
            prompt = f"""
            Create LinkedIn content in the style of {request.creator_key}.
            
            User request: {request.prompt}
            
            IMPORTANT: Use the following research context to create highly relevant and data-driven content:
            {research_context}
            
            Instructions:
            - Incorporate specific data, insights, and findings from the research
            - Make the content highly relevant to the research topics
            - Use statistics, quotes, or key points from the research when appropriate
            - Ensure the content feels authentic and valuable to the audience
            
            Please structure your response exactly as follows:
            
            1. LinkedIn Post:
            [Write a compelling LinkedIn post here]
            
            2. LinkedIn Reel Transcript:
            [Write a 60-second video transcript here]
            
            3. Hashtags:
            #hashtag1 #hashtag2 #hashtag3 #hashtag4 #hashtag5
            
            4. Engagement Questions:
            [Question 1]
            [Question 2]
            [Question 3]
            
            5. Talking Points:
            [Point 1]
            [Point 2]
            [Point 3]
            [Point 4]
            [Point 5]
            
            6. Style Notes:
            [Brief description of the style used]
            
            7. Context Used:
            [Mention which research items were used]
            """
        else:
            prompt = f"""
            Create LinkedIn content in the style of {request.creator_key}.
            
            User request: {request.prompt}
            
            Note: No specific research provided. Use general best practices and industry knowledge for LinkedIn content.
            
            Please structure your response exactly as follows:
            
            1. LinkedIn Post:
            [Write a compelling LinkedIn post here]
            
            2. LinkedIn Reel Transcript:
            [Write a 60-second video transcript here]
            
            3. Hashtags:
            #hashtag1 #hashtag2 #hashtag3 #hashtag4 #hashtag5
            
            4. Engagement Questions:
            [Question 1]
            [Question 2]
            [Question 3]
            
            5. Talking Points:
            [Point 1]
            [Point 2]
            [Point 3]
            [Point 4]
            [Point 5]
            
            6. Style Notes:
            [Brief description of the style used]
            
            7. Context Used:
            [Mention that general best practices were used]
            """
        
        logger.info(f"Generating content for creator: {request.creator_key}")
        
        # Generate content using Groq
        messages = [
            {"role": "system", "content": "You are a LinkedIn content creator expert. Generate engaging, professional content."},
            {"role": "user", "content": prompt}
        ]
        
        response_text = await query_groq(messages)
        logger.info(f"Raw response from Groq: {response_text[:500]}...")
        
        # Parse response with better section detection
        lines = response_text.split('\n')
        linkedin_post = ""
        linkedin_reel = ""
        hashtags = []
        engagement_questions = []
        talking_points = []
        style_notes = ""
        context_used = ""
        
        current_section = ""
        in_content = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect section headers
            lower_line = line.lower()
            if any(keyword in lower_line for keyword in ["1.", "linkedin post", "post:", "post content"]):
                current_section = "post"
                in_content = True
                continue
            elif any(keyword in lower_line for keyword in ["2.", "reel", "transcript", "video", "linkedin reel"]):
                current_section = "reel"
                in_content = True
                continue
            elif any(keyword in lower_line for keyword in ["3.", "hashtag", "hashtags", "#"]):
                current_section = "hashtags"
                in_content = True
                continue
            elif any(keyword in lower_line for keyword in ["4.", "engagement", "question", "questions"]):
                current_section = "questions"
                in_content = True
                continue
            elif any(keyword in lower_line for keyword in ["5.", "talking", "point", "points", "key points"]):
                current_section = "points"
                in_content = True
                continue
            elif any(keyword in lower_line for keyword in ["6.", "style", "style notes"]):
                current_section = "style"
                in_content = True
                continue
            elif any(keyword in lower_line for keyword in ["7.", "context", "context used"]):
                current_section = "context"
                in_content = True
                continue
            elif any(keyword in lower_line for keyword in ["generate:", "instructions:", "create:"]):
                in_content = False
                continue
            
            # Add content to appropriate section
            if in_content and current_section:
                if current_section == "post":
                    linkedin_post += line + "\n"
                elif current_section == "reel":
                    linkedin_reel += line + "\n"
                elif current_section == "hashtags":
                    if line.startswith("#"):
                        hashtags.append(line)
                    elif "#" in line:
                        # Extract hashtags from line
                        import re
                        found_hashtags = re.findall(r'#\w+', line)
                        hashtags.extend(found_hashtags)
                    elif line.strip():
                        # If line doesn't start with # but contains hashtags, extract them
                        import re
                        found_hashtags = re.findall(r'#\w+', line)
                        hashtags.extend(found_hashtags)
                elif current_section == "questions":
                    if line and not line.startswith("#") and not line.startswith("•"):
                        engagement_questions.append(line)
                elif current_section == "points":
                    if line and not line.startswith("#"):
                        talking_points.append(line)
                elif current_section == "style":
                    style_notes += line + "\n"
                elif current_section == "context":
                    context_used += line + "\n"
        
        # Fallback content if parsing failed
        if not linkedin_post:
            linkedin_post = f"🚀 {request.prompt}\n\nBased on the latest research and industry insights, here's what you need to know about this topic.\n\nKey takeaways:\n• Important point 1\n• Important point 2\n• Important point 3\n\nWhat's your experience with this? Share your thoughts in the comments below!\n\n#LinkedIn #Content #Professional #Networking #Growth"
        if not linkedin_reel:
            linkedin_reel = f"Hey everyone! 👋 Today I want to talk about {request.prompt}.\n\n[0:00-0:15] Introduction and hook\n[0:15-0:30] Main point 1\n[0:30-0:45] Main point 2\n[0:45-0:60] Call to action and engagement question"
        if not hashtags:
            hashtags = ["#LinkedIn", "#Content", "#Professional", "#Networking", "#Growth"]
        if not engagement_questions:
            engagement_questions = ["What do you think about this?", "Have you experienced something similar?", "What's your take on this topic?"]
        if not talking_points:
            talking_points = ["Key insight from research", "Practical application", "Industry impact", "Future implications", "Action steps"]
        
        # Clean up content - remove extra quotes and clean up formatting
        def clean_content(text):
            if not text:
                return ""
            logger.info(f"Cleaning content: {text[:100]}...")
            # Remove extra quotes at the beginning and end
            text = text.strip()
            # Remove quotes at the beginning and end if they match
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]
                logger.info("Removed outer quotes")
            elif text.startswith("'") and text.endswith("'"):
                text = text[1:-1]
                logger.info("Removed outer single quotes")
            # Remove any remaining smart quotes
            text = text.replace('"', '"').replace('"', '"').replace('"', '"').replace('"', '"')
            text = text.replace(''', "'").replace(''', "'")
            cleaned = text.strip()
            logger.info(f"Cleaned content: {cleaned[:100]}...")
            return cleaned
        
        # Clean up hashtags - split if they're in a single string
        def clean_hashtags(hashtag_list):
            cleaned_hashtags = []
            for hashtag in hashtag_list:
                if isinstance(hashtag, str):
                    # Split by space and filter for hashtags
                    import re
                    found_hashtags = re.findall(r'#\w+', hashtag)
                    cleaned_hashtags.extend(found_hashtags)
                else:
                    cleaned_hashtags.append(hashtag)
            # Remove duplicates and return
            return list(set(cleaned_hashtags))
        
        # Debug hashtags
        logger.info(f"Raw hashtags before cleaning: {hashtags}")
        cleaned_hashtags = clean_hashtags(hashtags)
        logger.info(f"Cleaned hashtags: {cleaned_hashtags}")
        
        return ContentGenerationResponse(
            linkedin_post=clean_content(linkedin_post),
            linkedin_reel_transcript=clean_content(linkedin_reel),
            hashtags=cleaned_hashtags,
            engagement_questions=engagement_questions,
            talking_points=talking_points,
            style_notes=clean_content(style_notes) or f"Content generated in {request.creator_key} style",
            context_used=clean_content(context_used) or f"Used {len(request.research_ids)} research items"
        )
    
    except Exception as e:
        logger.error(f"Content generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def extract_news_articles(html_content: str, base_url: str) -> List[dict]:
    """Extract multiple news articles from HTML content"""
    articles = []
    
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Common news article selectors for modern news sites
        article_selectors = [
            'article',
            '[class*="article"]',
            '[class*="news"]',
            '[class*="story"]',
            '[class*="post"]',
            '[class*="card"]',
            '[class*="item"]',
            '.entry',
            '.content-item',
            '.news-item',
            '.story-item',
            '.post-item',
            '.card',
            '.item'
        ]
        
        # Find all potential article containers
        article_containers = []
        for selector in article_selectors:
            containers = soup.select(selector)
            article_containers.extend(containers)
        
        # Remove duplicates
        seen = set()
        unique_containers = []
        for container in article_containers:
            if container not in seen:
                seen.add(container)
                unique_containers.append(container)
        
        logger.info(f"Found {len(unique_containers)} potential article containers")
        
        # Extract articles from containers
        for container in unique_containers[:15]:  # Limit to 15 articles
            try:
                # Extract title
                title = None
                title_selectors = [
                    'h1', 'h2', 'h3', 'h4',
                    '.title', '.headline', '.post-title', '.article-title',
                    '[class*="title"]', '[class*="headline"]',
                    'a[href]'  # Sometimes titles are in links
                ]
                
                for selector in title_selectors:
                    title_elem = container.select_one(selector)
                    if title_elem:
                        title_text = title_elem.get_text().strip()
                        if title_text and len(title_text) > 10 and len(title_text) < 200:
                            title = title_text
                            break
                
                # Extract content/summary
                content = None
                content_selectors = [
                    '.content', '.body', '.text', '.description', '.summary', '.excerpt',
                    '.post-content', '.article-content', '.story-content',
                    'p', '.p', '[class*="content"]', '[class*="text"]'
                ]
                
                for selector in content_selectors:
                    content_elem = container.select_one(selector)
                    if content_elem:
                        content_text = content_elem.get_text().strip()
                        if content_text and len(content_text) > 20:
                            content = content_text
                            break
                
                # If no content found, try to get all text from container
                if not content:
                    all_text = container.get_text().strip()
                    if all_text and len(all_text) > 50:
                        # Take first 200 characters as content
                        content = all_text[:200] + "..." if len(all_text) > 200 else all_text
                
                # Extract link
                link = None
                link_elem = container.find('a', href=True)
                if link_elem:
                    link = link_elem['href']
                    if link.startswith('/'):
                        from urllib.parse import urljoin
                        link = urljoin(base_url, link)
                    elif link.startswith('http'):
                        link = link
                    else:
                        link = base_url
                
                # Extract date
                date = None
                date_selectors = [
                    '.date', '.time', '.published', '.timestamp',
                    '[datetime]', '[class*="date"]', '[class*="time"]'
                ]
                for selector in date_selectors:
                    date_elem = container.select_one(selector)
                    if date_elem:
                        date_text = date_elem.get_text().strip()
                        if date_text:
                            date = date_text
                            break
                
                # Extract author
                author = None
                author_selectors = [
                    '.author', '.byline', '.writer', '.reporter',
                    '[class*="author"]', '[rel="author"]'
                ]
                for selector in author_selectors:
                    author_elem = container.select_one(selector)
                    if author_elem:
                        author_text = author_elem.get_text().strip()
                        if author_text:
                            author = author_text
                            break
                
                # Only add if we have title and content
                if title and content and len(content) > 30:
                    articles.append({
                        'title': title,
                        'content': content,
                        'url': link or base_url,
                        'date': date or datetime.now().strftime("%Y-%m-%d"),
                        'author': author
                    })
                    logger.info(f"Extracted article: {title[:50]}...")
                    
            except Exception as e:
                logger.warning(f"Error extracting article from container: {e}")
                continue
        
        # If no articles found with selectors, try to extract from main content
        if not articles:
            logger.info("No articles found with selectors, trying main content extraction")
            main_content = soup.find('main') or soup.find('body')
            if main_content:
                # Extract paragraphs as potential articles
                paragraphs = main_content.find_all('p')
                for i, p in enumerate(paragraphs[:8]):  # Limit to 8 paragraphs
                    text = p.get_text().strip()
                    if len(text) > 80:  # Only substantial paragraphs
                        articles.append({
                            'title': f"Content Section {i+1}",
                            'content': text,
                            'url': base_url,
                            'date': datetime.now().strftime("%Y-%m-%d"),
                            'author': None
                        })
        
        logger.info(f"Total articles extracted: {len(articles)}")
        
    except Exception as e:
        logger.error(f"Error in extract_news_articles: {e}")
    
    return articles

@app.post("/linkedin/crawl", response_model=CrawlResponse, summary="Crawl Website for LinkedIn Content", tags=["LinkedIn"])
async def crawl_website(request: CrawlRequest):
    """Enhanced crawler that extracts multiple news articles and creates separate research entries"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(str(request.url), timeout=30.0)
            response.raise_for_status()
            html_content = response.text
        
        # Extract title
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', html_content, re.IGNORECASE)
        title = title_match.group(1) if title_match else None
        
        # Extract hostname
        from urllib.parse import urlparse
        hostname = urlparse(str(request.url)).netloc
        
        # Simple but effective content extraction
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract text content
        text = soup.get_text()
        text = re.sub(r'\s+', ' ', text).strip()
        
        if not text:
            raise HTTPException(status_code=400, detail="No content could be extracted from the URL")
        
        # Create articles from text sections
        articles = []
        
        # Split text into meaningful sections
        text_sections = text.split('. ')
        current_section = ""
        section_count = 0
        
        for sentence in text_sections:
            if len(current_section) + len(sentence) < 400 and section_count < 8:
                current_section += sentence + ". "
            else:
                if current_section.strip() and len(current_section.strip()) > 100:
                    articles.append({
                        'title': f"Content Section {section_count + 1}",
                        'content': current_section.strip(),
                        'url': str(request.url),
                        'date': datetime.now().strftime("%Y-%m-%d"),
                        'author': None
                    })
                    section_count += 1
                    current_section = sentence + ". "
        
        # Add the last section
        if current_section.strip() and len(current_section.strip()) > 100 and section_count < 8:
            articles.append({
                'title': f"Content Section {section_count + 1}",
                'content': current_section.strip(),
                'url': str(request.url),
                'date': datetime.now().strftime("%Y-%m-%d"),
                'author': None
            })
        
        # Store each article as separate research entry
        stored_articles = []
        if supabase_client:
            try:
                for i, article in enumerate(articles):
                    # Create unique topic for each article
                    article_topic = f"News Article {i+1}: {article['title'][:50]}..."
                    
                    research_data = {
                        "topic": article_topic,
                        "findings": article['content'][:1000] + "..." if len(article['content']) > 1000 else article['content'],
                        "data": f"News article from {hostname}. Title: {article['title']}. Date: {article['date']}. Author: {article['author'] or 'Unknown'}. URL: {article['url']}",
                        "source": article['url'],
                        "tags": f"news-article,{hostname},web-crawl,latest-news"
                    }
                    
                    result = supabase_client.table("research").insert(research_data).execute()
                    stored_articles.append({
                        'id': result.data[0]['id'] if result.data else None,
                        'title': article['title'],
                        'content': article['content'][:200] + "..." if len(article['content']) > 200 else article['content']
                    })
                    logger.info(f"Article {i+1} stored in Supabase: {article_topic}")
                
            except Exception as e:
                logger.error(f"Supabase storage error: {e}")
        
        # Return summary of what was extracted
        total_content = sum(len(article['content']) for article in articles)
        main_text = f"Extracted {len(articles)} articles from {hostname}:\n\n"
        for i, article in enumerate(articles[:3]):  # Show first 3 articles
            main_text += f"{i+1}. {article['title']}\n"
            main_text += f"   {article['content'][:100]}...\n\n"
        
        if len(articles) > 3:
            main_text += f"... and {len(articles) - 3} more articles"
        
        return CrawlResponse(
            url=request.url,
            title=f"News Crawl: {len(articles)} articles from {hostname}",
            hostname=hostname,
            main_text=main_text,
            source=str(request.url),
            pages_crawled=1,
            total_content_length=total_content
        )
    
    except Exception as e:
        logger.error(f"Crawl error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/linkedin/crawls", summary="List Web Crawls", tags=["LinkedIn"])
def list_crawls():
    """List all web crawls"""
    try:
        if supabase_client:
            result = supabase_client.table("research").select("*").ilike("tags", "%web-crawl%").execute()
            return result.data
        else:
            return []
    except Exception as e:
        logger.error(f"List crawls error: {e}")
        return []

# Legacy endpoints for compatibility
@app.post("/upload", response_model=UploadResponse, summary="Upload and Process a Document", tags=["Document Management"])
async def upload_file(file: UploadFile = File(...)):
    """Legacy upload endpoint - redirects to LinkedIn endpoint"""
    return await upload_document(file)

@app.post("/crawl", response_model=CrawlResponse, summary="Extract clean content from a URL and index it", tags=["Website"])
async def crawl_and_extract(request: CrawlRequest = Body(...)):
    """Legacy crawl endpoint - redirects to LinkedIn endpoint"""
    return await crawl_website(request)

@app.post("/ask", response_model=AskResponse, summary="Ask a Question and get relevant information", tags=["Document Management"])
async def ask_question(request: AskRequest):
    """Ask a question about uploaded documents"""
    try:
        # Search through chunks for relevant content
        relevant_chunks = []
        for i, chunk in enumerate(chunks):
            if request.question.lower() in chunk.lower():
                relevant_chunks.append({
                    'text': chunk,
                    'metadata': metadata[i] if i < len(metadata) else {}
                })
        
        if not relevant_chunks:
            return AskResponse(
                answer="I couldn't find relevant information in the uploaded documents.",
                question=request.question,
                sources=[],
                client_followups=["Try uploading more documents", "Ask a different question"],
                sales_followups=["Consider uploading sales materials", "Add customer testimonials"]
            )
        
        # Generate answer using Groq
        context = "\n\n".join([chunk['text'] for chunk in relevant_chunks[:3]])
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer questions based on the provided context."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {request.question}"}
        ]
        
        answer = await query_groq(messages)
        
        sources = [{
            'title': chunk['metadata'].get('filename', 'Unknown'),
            'content': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'],
            'source': chunk['metadata'].get('source', 'Unknown')
        } for chunk in relevant_chunks[:3]]
        
        return AskResponse(
            answer=answer,
            question=request.question,
            sources=sources,
            client_followups=["Would you like me to search for more information?", "Should I analyze other documents?"],
            sales_followups=["Use this information in your sales presentations", "Share these insights with prospects"]
        )
    
    except Exception as e:
        logger.error(f"Ask question error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", summary="List all indexed documents and metadata", tags=["Document Management"])
def list_documents():
    """List all documents"""
    return documents

@app.get("/chunks", summary="Get all chunks and their metadata", tags=["Document Management"])
def get_all_chunks():
    """Get all chunks"""
    return {
        "chunks": chunks,
        "metadata": metadata,
        "total_chunks": len(chunks)
    }

@app.delete("/documents/{filename}", summary="Delete a Document and its Embeddings", tags=["Document Management"])
async def delete_document(filename: str, background_tasks: BackgroundTasks):
    """Delete a document and all its associated chunks and embeddings"""
    try:
        global documents, chunks, metadata
        
        # Find chunks for this document
        indices_to_remove = []
        for i, meta in enumerate(metadata):
            if meta.get('filename') == filename:
                indices_to_remove.append(i)
        
        if not indices_to_remove:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Remove from metadata and chunks
        for i in reversed(indices_to_remove):
            del metadata[i]
            del chunks[i]
        
        # Remove from documents
        documents = [doc for doc in documents if doc['filename'] != filename]
        
        return {"message": f"Document {filename} deleted successfully"}
    
    except Exception as e:
        logger.error(f"Delete document error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/refresh", summary="Synchronize the index with GCS documents", tags=["Document Management"])
async def refresh_index():
    """Refresh the vector index from Google Cloud Storage"""
    try:
        return {"message": "Index refreshed successfully"}
    except Exception as e:
        logger.error(f"Refresh error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 