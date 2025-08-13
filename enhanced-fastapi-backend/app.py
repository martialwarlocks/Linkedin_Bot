import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import hashlib
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# AI and ML imports
import openai
from openai import OpenAI
import tiktoken
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Web crawling imports
import trafilatura
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import newspaper
from readability import Document
import feedparser

# Database and storage
from supabase import create_client, Client
import chromadb
from chromadb.config import Settings

# Google Cloud Storage
from google.cloud import storage
from google.cloud.storage import Blob
import google.auth
from google.auth.exceptions import DefaultCredentialsError

# Document processing
import PyPDF2
import io
from docx import Document as DocxDocument
import fitz  # PyMuPDF

# Environment and config
from dotenv import load_dotenv
import redis
from celery import Celery

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced LinkedIn Content Creator API",
    description="Advanced AI-powered LinkedIn content creation with web crawling, document processing, and vector search",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# Initialize ChromaDB for vector storage
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="linkedin_content_data",
    metadata={"hnsw:space": "cosine"}
)

# Initialize sentence transformer for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Import document processor and AI generator
from document_processor import EnhancedDocumentProcessor
from ai_generator import EnhancedAIContentGenerator

# Initialize web crawler and document processor
web_crawler = EnhancedWebCrawler()
document_processor = EnhancedDocumentProcessor()
ai_generator = EnhancedAIContentGenerator()

# Initialize Google Cloud Storage
try:
    storage_client = storage.Client()
    bucket_name = os.getenv("GCS_BUCKET_NAME", "linkedin-bot-documents")
    bucket = storage_client.bucket(bucket_name)
    logger.info(f"Google Cloud Storage initialized with bucket: {bucket_name}")
except DefaultCredentialsError:
    logger.warning("Google Cloud Storage credentials not found, using local storage")
    bucket = None
except Exception as e:
    logger.error(f"Failed to initialize Google Cloud Storage: {e}")
    bucket = None

# Pydantic models
class WebCrawlRequest(BaseModel):
    url: str
    depth: int = Field(default=2, ge=1, le=5)
    max_pages: int = Field(default=10, ge=1, le=50)
    include_links: bool = True

class ContentGenerationRequest(BaseModel):
    prompt: str
    creator_key: str = "gary-v"
    research_ids: List[int] = []
    document_ids: List[str] = []
    crawl_ids: List[int] = []
    max_tokens: int = 2000
    temperature: float = 0.7

class DocumentUploadResponse(BaseModel):
    filename: str
    content_type: str
    size: int
    extracted_text: str
    embeddings_count: int
    document_id: str

class WebCrawlResponse(BaseModel):
    crawl_id: int
    url: str
    pages_crawled: int
    total_content_length: int
    status: str
    created_at: datetime

# Enhanced Web Crawler
class EnhancedWebCrawler:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    async def crawl_website(self, url: str, depth: int = 2, max_pages: int = 10) -> Dict[str, Any]:
        """Enhanced web crawler with multiple extraction methods"""
        try:
            crawl_id = int(datetime.now().timestamp())
            crawled_pages = []
            visited_urls = set()
            
            # Start with the main URL
            urls_to_crawl = [(url, 0)]
            
            while urls_to_crawl and len(crawled_pages) < max_pages:
                current_url, current_depth = urls_to_crawl.pop(0)
                
                if current_url in visited_urls or current_depth > depth:
                    continue
                
                visited_urls.add(current_url)
                
                try:
                    # Extract content using multiple methods
                    content = await self._extract_content(current_url)
                    
                    if content and len(content['text']) > 100:  # Minimum content threshold
                        crawled_pages.append({
                            'url': current_url,
                            'title': content['title'],
                            'text': content['text'],
                            'metadata': content['metadata'],
                            'depth': current_depth
                        })
                        
                        # Store in vector database
                        await self._store_content_in_vector_db(content, current_url, crawl_id)
                        
                        # Find more links if within depth limit
                        if current_depth < depth:
                            links = content.get('links', [])
                            for link in links[:5]:  # Limit links per page
                                if link not in visited_urls:
                                    urls_to_crawl.append((link, current_depth + 1))
                
                except Exception as e:
                    logger.error(f"Error crawling {current_url}: {str(e)}")
                    continue
            
            # Store crawl metadata in Supabase
            crawl_data = {
                'crawl_id': crawl_id,
                'url': url,
                'pages_crawled': len(crawled_pages),
                'total_content_length': sum(len(page['text']) for page in crawled_pages),
                'status': 'completed',
                'created_at': datetime.now().isoformat(),
                'metadata': {
                    'depth': depth,
                    'max_pages': max_pages,
                    'extraction_methods': ['trafilatura', 'newspaper', 'readability']
                }
            }
            
            supabase.table('web_crawls').insert(crawl_data).execute()
            
            return {
                'crawl_id': crawl_id,
                'url': url,
                'pages_crawled': len(crawled_pages),
                'total_content_length': crawl_data['total_content_length'],
                'status': 'completed',
                'created_at': datetime.now(),
                'pages': crawled_pages
            }
            
        except Exception as e:
            logger.error(f"Web crawling failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Web crawling failed: {str(e)}")
    
    async def _extract_content(self, url: str) -> Dict[str, Any]:
        """Extract content using multiple methods for better results"""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            html_content = response.text
            
            content = {
                'title': '',
                'text': '',
                'metadata': {},
                'links': []
            }
            
            # Method 1: Trafilatura (best for article extraction)
            try:
                extracted = trafilatura.extract(html_content, include_formatting=True, include_links=True)
                if extracted:
                    content['text'] = extracted
                    content['title'] = trafilatura.extract_metadata(html_content).get('title', '')
            except Exception as e:
                logger.warning(f"Trafilatura extraction failed: {e}")
            
            # Method 2: Newspaper3k (good for news sites)
            if not content['text'] or len(content['text']) < 200:
                try:
                    article = newspaper.Article(url)
                    article.download(input_html=html_content)
                    article.parse()
                    if article.text and len(article.text) > len(content['text']):
                        content['text'] = article.text
                        content['title'] = article.title or content['title']
                except Exception as e:
                    logger.warning(f"Newspaper extraction failed: {e}")
            
            # Method 3: Readability (fallback)
            if not content['text'] or len(content['text']) < 100:
                try:
                    doc = Document(html_content)
                    content['text'] = doc.summary()
                    content['title'] = doc.title() or content['title']
                except Exception as e:
                    logger.warning(f"Readability extraction failed: {e}")
            
            # Extract links
            try:
                soup = BeautifulSoup(html_content, 'html.parser')
                base_url = response.url
                links = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    absolute_url = urljoin(base_url, href)
                    if urlparse(absolute_url).netloc == urlparse(base_url).netloc:
                        links.append(absolute_url)
                content['links'] = list(set(links))[:10]  # Limit to 10 unique links
            except Exception as e:
                logger.warning(f"Link extraction failed: {e}")
            
            # Add metadata
            content['metadata'] = {
                'url': url,
                'extraction_method': 'multi_method',
                'content_length': len(content['text']),
                'links_count': len(content['links'])
            }
            
            return content
            
        except Exception as e:
            logger.error(f"Content extraction failed for {url}: {e}")
            return None
    
    async def _store_content_in_vector_db(self, content: Dict[str, Any], url: str, crawl_id: int):
        """Store extracted content in vector database"""
        try:
            if not content['text']:
                return
            
            # Create embeddings
            embeddings = embedding_model.encode(content['text'])
            
            # Store in ChromaDB
            collection.add(
                embeddings=[embeddings.tolist()],
                documents=[content['text']],
                metadatas=[{
                    'url': url,
                    'title': content['title'],
                    'crawl_id': crawl_id,
                    'type': 'web_crawl',
                    'timestamp': datetime.now().isoformat()
                }],
                ids=[f"crawl_{crawl_id}_{hashlib.md5(url.encode()).hexdigest()}"]
            )
            
        except Exception as e:
            logger.error(f"Failed to store content in vector DB: {e}")

# Import additional modules
from document_processor import EnhancedDocumentProcessor
from ai_generator import EnhancedAIContentGenerator

# Initialize components
web_crawler = EnhancedWebCrawler()
document_processor = EnhancedDocumentProcessor(embedding_model, collection)
ai_generator = EnhancedAIContentGenerator(openai_client, embedding_model, collection, supabase)

# API Endpoints
@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Enhanced LinkedIn Content Creator API is online and ready",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "features": [
            "Enhanced web crawling with multiple extraction methods",
            "Vector database for semantic search",
            "Document processing (PDF, DOCX, TXT, XLSX, PPTX)",
            "Contextual AI content generation",
            "Research and document integration",
            "Semantic search capabilities"
        ]
    }

@app.post("/linkedin/crawl")
async def crawl_website(request: WebCrawlRequest):
    """Enhanced web crawling endpoint"""
    try:
        result = await web_crawler.crawl_website(
            url=request.url,
            depth=request.depth,
            max_pages=request.max_pages
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/linkedin/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process document with Google Cloud Storage"""
    try:
        file_content = await file.read()
        
        # Generate unique document ID
        document_id = hashlib.md5(f"{file.filename}_{datetime.now().isoformat()}".encode()).hexdigest()
        
        # Store file in Google Cloud Storage
        gcs_url = None
        if bucket:
            blob_name = f"documents/{document_id}/{file.filename}"
            blob = bucket.blob(blob_name)
        blob.upload_from_string(file_content, content_type=file.content_type)
            gcs_url = f"gs://{bucket.name}/{blob_name}"
            logger.info(f"File uploaded to GCS: {gcs_url}")
        else:
            # Fallback to local storage
            local_path = f"./uploads/{document_id}_{file.filename}"
            os.makedirs("./uploads", exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(file_content)
            gcs_url = f"file://{local_path}"
            logger.info(f"File stored locally: {local_path}")
        
        # Process document content
        result = await document_processor.process_document(file_content, file.filename, file.content_type)
        
        # Store document metadata in Supabase
        doc_data = {
            'document_id': document_id,
            'filename': file.filename,
            'content_type': file.content_type,
            'size': len(file_content),
            'gcs_url': gcs_url,
            'extracted_text': result.get('extracted_text', ''),
            'embeddings_count': result.get('embeddings_count', 0),
            'created_at': datetime.now().isoformat(),
            'status': 'processed'
        }
        
        supabase.table('documents').insert(doc_data).execute()
        
        return {
            'document_id': document_id,
            'filename': file.filename,
            'content_type': file.content_type,
            'size': len(file_content),
            'gcs_url': gcs_url,
            'extracted_text': result.get('extracted_text', ''),
            'embeddings_count': result.get('embeddings_count', 0),
            'status': 'success'
        }
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/linkedin/generate")
async def generate_content(request: ContentGenerationRequest):
    """Generate contextual LinkedIn content"""
    try:
        request_dict = {
            'prompt': request.prompt,
            'creator_key': request.creator_key,
            'research_ids': request.research_ids,
            'document_ids': request.document_ids,
            'crawl_ids': request.crawl_ids,
            'max_tokens': request.max_tokens,
            'temperature': request.temperature
        }
        
        result = await ai_generator.generate_contextual_content(request_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/linkedin/research")
async def get_research():
    """Get all research data"""
    try:
        result = supabase.table('research').select('*').order('created_at', desc=True).execute()
        return result.data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/linkedin/research")
async def add_research(request: dict):
    """Add new research data"""
    try:
        research_data = {
            'topic': request.get('topic', ''),
            'findings': request.get('findings', ''),
            'source': request.get('source', ''),
            'data': request.get('data', ''),
            'tags': request.get('tags', ''),
            'created_at': datetime.now().isoformat()
        }
        
        result = supabase.table('research').insert(research_data).execute()
        return result.data[0] if result.data else research_data
    except Exception as e:
        logger.error(f"Error adding research: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/linkedin/documents")
async def get_documents():
    """Get all processed documents"""
    try:
        result = supabase.table('documents').select('*').order('created_at', desc=True).execute()
        return result.data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/linkedin/crawls")
async def get_crawls():
    """Get all web crawls"""
    try:
        result = supabase.table('web_crawls').select('*').order('created_at', desc=True).execute()
        return result.data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/linkedin/creators")
async def get_creators():
    """Get all creator styles"""
    try:
        result = supabase.table('creators').select('*').execute()
        return result.data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/linkedin/search")
async def semantic_search(query: str = Form(...), limit: int = Form(5)):
    """Semantic search in vector database"""
    try:
        results = await document_processor.search_documents(query, limit)
        return {"query": query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/linkedin/research-summary")
async def generate_research_summary(research_ids: List[int] = Form(...)):
    """Generate summary of research findings"""
    try:
        result = await ai_generator.generate_research_summary(research_ids)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/linkedin/document-insights")
async def generate_document_insights(document_ids: List[str] = Form(...)):
    """Generate insights from documents"""
    try:
        result = await ai_generator.generate_document_insights(document_ids)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 