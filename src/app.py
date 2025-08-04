import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Any, Dict
from contextlib import asynccontextmanager
import json
from datetime import datetime, timedelta
import traceback
import tempfile
import re
from urllib.parse import quote, urlparse
import uuid
from fastapi import FastAPI, HTTPException, BackgroundTasks, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import httpx
from google.cloud import storage
from google.api_core.exceptions import GoogleAPIError, NotFound

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# --- Configuration ---
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "spikedai-bucket-one")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
BASE_URL = "https://linkedin-content-creator-api.run.app"

# --- API Keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY environment variable not set. Content generation will fail.")
logger.info(f"GROQ_API_KEY configured: {'✓' if GROQ_API_KEY else '✗'}")

# Global GCS client
gcs_client = None

# --- GCS Client Singleton ---
def get_gcs_client() -> storage.Client:
    """Initializes and returns a thread-safe GCS client."""
    global gcs_client
    if gcs_client is None:
        try:
            logger.info("Initializing Google Cloud Storage client.")
            gcs_client = storage.Client()
        except Exception as e:
            logger.error(f"FATAL: Failed to initialize GCS client: {e}", exc_info=True)
            raise
    return gcs_client

# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application startup and shutdown events."""
    logger.info("LinkedIn Bot API: Startup sequence initiated.")
    try:
        # Initialize GCS client
        get_gcs_client()
        logger.info("LinkedIn Bot API: Startup complete. Ready to serve requests.")
        yield
    except Exception as e:
        logger.critical(f"LinkedIn Bot API: FATAL STARTUP ERROR: {e}", exc_info=True)
        raise
    finally:
        logger.info("LinkedIn Bot API: Shutdown sequence initiated.")
        logger.info("LinkedIn Bot API: Shutdown complete.")

app = FastAPI(
    title="LinkedIn Content Creator API",
    version="1.0.0",
    description="FastAPI backend for LinkedIn Content Creator Bot. AI-powered content generation with persistent storage.",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class ResearchItem(BaseModel):
    id: Optional[int] = None
    topic: str = Field(..., min_length=1, max_length=200)
    findings: str = Field(..., min_length=1, max_length=2000)
    source: str = Field(..., min_length=1, max_length=300)
    data: str = Field(..., min_length=1, max_length=2000)
    tags: str = Field(..., min_length=1, max_length=500)  # Comma-separated string
    created_at: Optional[str] = None

class CreatorProfile(BaseModel):
    id: Optional[int] = None
    name: str = Field(..., min_length=1, max_length=100)
    key: str = Field(default="", max_length=100)
    tone: str = Field(..., min_length=1, max_length=500)
    structure: str = Field(..., min_length=1, max_length=500)
    language: str = Field(..., min_length=1, max_length=500)
    length: str = Field(..., min_length=1, max_length=200)
    hooks: str = Field(..., min_length=1, max_length=1000)  # Comma-separated string
    endings: str = Field(..., min_length=1, max_length=1000)  # Comma-separated string
    characteristics: str = Field(..., min_length=1, max_length=1000)
    created_at: Optional[str] = None

class ContentGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    creator_key: str = Field(..., min_length=1)
    research_ids: Optional[List[int]] = []

class ContentGenerationResponse(BaseModel):
    linkedin_posts: List[str]
    video_scripts: List[str]
    hashtags: List[str]
    engagement_tips: List[str]
    creator_name: str
    research_used: List[str]

class PromptHistoryItem(BaseModel):
    id: Optional[int] = None
    prompt: str
    creator_name: str
    research_topics: List[str]
    response_data: dict
    created_at: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: str
    version: str

# --- Helper Functions ---
async def query_groq(messages: List[dict], max_tokens: int = 3000, temperature: float = 0.8) -> str:
    """Send request to Groq API for content generation"""
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                GROQ_API_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    "model": GROQ_MODEL, 
                    "messages": messages, 
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            )
            logger.info(f"Groq API response status: {response.status_code}")
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
    except httpx.TimeoutException:
        logger.error("Groq API timeout")
        raise HTTPException(status_code=504, detail="Content generation timed out")
    except httpx.HTTPStatusError as e:
        logger.error(f"Groq API HTTP error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=502, detail="AI service temporarily unavailable")
    except Exception as e:
        logger.error(f"Groq query failed: {e}")
        raise HTTPException(status_code=502, detail="Failed to generate content")

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    filename = os.path.basename(filename)
    filename = re.sub(r'[^a-zA-Z0-9_\-.]', '_', filename)
    return filename

def generate_creator_key(name: str) -> str:
    """Generate a safe key from creator name"""
    return re.sub(r'[^a-zA-Z0-9]', '-', name.lower()).strip('-')

# --- API Endpoints ---

@app.get("/", response_model=HealthResponse, summary="Health Check", tags=["System"])
def read_root():
    """Health check endpoint to confirm API is running"""
    return HealthResponse(
        status="healthy",
        message="LinkedIn Content Creator API is online and ready",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

# Research Database Endpoints
@app.get("/linkedin/research", summary="Get all research items", tags=["Research Management"])
async def get_research_items():
    """Get all research items sorted by newest first"""
    try:
        client = get_gcs_client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob("linkedin/research.json")
        
        if not blob.exists():
            logger.info("Research database not found, returning empty list")
            return []
        
        data = json.loads(blob.download_as_text())
        # Sort by created_at descending (newest first)
        sorted_data = sorted(data, key=lambda x: x.get('created_at', ''), reverse=True)
        logger.info(f"Retrieved {len(sorted_data)} research items")
        return sorted_data
    except Exception as e:
        logger.error(f"Error getting research items: {e}")
        return []

@app.post("/linkedin/research", response_model=ResearchItem, summary="Add new research item", tags=["Research Management"])
async def add_research_item(research: ResearchItem):
    """Add a new research item to the database"""
    try:
        client = get_gcs_client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob("linkedin/research.json")
        
        # Get existing data
        existing_data = []
        if blob.exists():
            existing_data = json.loads(blob.download_as_text())
        
        # Create new item with auto-generated ID
        new_item = research.dict()
        new_item['id'] = max([item.get('id', 0) for item in existing_data], default=0) + 1
        new_item['created_at'] = datetime.now().isoformat()
        existing_data.append(new_item)
        
        # Save back to GCS
        blob.upload_from_string(
            json.dumps(existing_data, indent=2),
            content_type="application/json"
        )
        
        logger.info(f"Added research item: {new_item['topic']}")
        return new_item
    except Exception as e:
        logger.error(f"Error adding research item: {e}")
        raise HTTPException(status_code=500, detail="Failed to add research item")

@app.put("/linkedin/research/{research_id}", response_model=ResearchItem, summary="Update research item", tags=["Research Management"])
async def update_research_item(research_id: int, research: ResearchItem):
    """Update an existing research item"""
    try:
        client = get_gcs_client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob("linkedin/research.json")
        
        if not blob.exists():
            raise HTTPException(status_code=404, detail="Research database not found")
        
        data = json.loads(blob.download_as_text())
        
        # Find and update the item
        for i, item in enumerate(data):
            if item['id'] == research_id:
                updated_item = research.dict()
                updated_item['id'] = research_id
                updated_item['created_at'] = item.get('created_at', datetime.now().isoformat())
                data[i] = updated_item
                
                # Save back to GCS
                blob.upload_from_string(
                    json.dumps(data, indent=2),
                    content_type="application/json"
                )
                
                logger.info(f"Updated research item: {updated_item['topic']}")
                return updated_item
        
        raise HTTPException(status_code=404, detail="Research item not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating research item: {e}")
        raise HTTPException(status_code=500, detail="Failed to update research item")

@app.delete("/linkedin/research/{research_id}", summary="Delete research item", tags=["Research Management"])
async def delete_research_item(research_id: int):
    """Delete a research item from the database"""
    try:
        client = get_gcs_client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob("linkedin/research.json")
        
        if not blob.exists():
            raise HTTPException(status_code=404, detail="Research database not found")
        
        data = json.loads(blob.download_as_text())
        original_length = len(data)
        data = [item for item in data if item['id'] != research_id]
        
        if len(data) == original_length:
            raise HTTPException(status_code=404, detail="Research item not found")
        
        # Save back to GCS
        blob.upload_from_string(
            json.dumps(data, indent=2),
            content_type="application/json"
        )
        
        logger.info(f"Deleted research item with ID: {research_id}")
        return {"message": "Research item deleted successfully", "deleted_id": research_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting research item: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete research item")

# Creator Profile Endpoints
@app.get("/linkedin/creators", summary="Get all creator profiles", tags=["Creator Management"])
async def get_creator_profiles():
    """Get all creator profiles with default creators if none exist"""
    try:
        client = get_gcs_client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob("linkedin/creators.json")
        
        if not blob.exists():
            logger.info("Creator database not found, creating default creators")
            # Create default creators
            default_creators = [
                {
                    "id": 1,
                    "name": "Gary Vaynerchuk",
                    "key": "gary-v",
                    "tone": "Direct, passionate, no-nonsense",
                    "structure": "Hook → Personal story → Business insight → Call to action",
                    "language": "Casual, uses \"you guys\", lots of energy",
                    "length": "Medium to long posts",
                    "hooks": "Look..., Here's the thing..., Real talk...",
                    "endings": "What do you think?, Let me know in the comments, DM me your thoughts",
                    "characteristics": "High energy, direct approach, business-focused, motivational",
                    "created_at": datetime.now().isoformat()
                },
                {
                    "id": 2,
                    "name": "Simon Sinek",
                    "key": "simon-sinek",
                    "tone": "Inspirational, thoughtful, leader-focused",
                    "structure": "Question → Story/Example → Leadership lesson → Reflection",
                    "language": "Professional but warm, thought-provoking",
                    "length": "Medium posts with clear paragraphs",
                    "hooks": "Why is it that..., The best leaders..., I once worked with...",
                    "endings": "What would you do?, Leadership is a choice, The choice is yours",
                    "characteristics": "Leadership-focused, inspirational, storytelling, thoughtful questions",
                    "created_at": datetime.now().isoformat()
                },
                {
                    "id": 3,
                    "name": "Seth Godin",
                    "key": "seth-godin",
                    "tone": "Wise, concise, marketing-focused",
                    "structure": "Insight → Brief explanation → Broader implication",
                    "language": "Concise, profound, marketing terminology",
                    "length": "Short, punchy posts",
                    "hooks": "The thing is..., Here's what I learned..., Marketing is...",
                    "endings": "Worth considering., Just saying., Think about it.",
                    "characteristics": "Concise wisdom, marketing insights, thought-provoking, minimal but impactful",
                    "created_at": datetime.now().isoformat()
                }
            ]
            
            # Save default creators to GCS
            blob.upload_from_string(
                json.dumps(default_creators, indent=2),
                content_type="application/json"
            )
            return default_creators
        
        data = json.loads(blob.download_as_text())
        logger.info(f"Retrieved {len(data)} creator profiles")
        return data
    except Exception as e:
        logger.error(f"Error getting creator profiles: {e}")
        # Return basic fallback creators
        return [
            {
                "id": 1,
                "name": "Gary Vaynerchuk",
                "key": "gary-v",
                "tone": "Direct, passionate, no-nonsense",
                "structure": "Hook → Personal story → Business insight → Call to action",
                "language": "Casual, uses \"you guys\", lots of energy",
                "length": "Medium to long posts",
                "hooks": "Look..., Here's the thing..., Real talk...",
                "endings": "What do you think?, Let me know in the comments, DM me your thoughts",
                "characteristics": "High energy, direct approach, business-focused, motivational",
                "created_at": datetime.now().isoformat()
            }
        ]

@app.post("/linkedin/creators", response_model=CreatorProfile, summary="Add new creator profile", tags=["Creator Management"])
async def add_creator_profile(creator: CreatorProfile):
    """Add a new creator profile to the database"""
    try:
        client = get_gcs_client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob("linkedin/creators.json")
        
        # Get existing data
        existing_data = []
        if blob.exists():
            existing_data = json.loads(blob.download_as_text())
        
        # Create new creator with auto-generated ID and key
        new_creator = creator.dict()
        new_creator['id'] = max([item.get('id', 0) for item in existing_data], default=0) + 1
        new_creator['key'] = generate_creator_key(creator.name)
        new_creator['created_at'] = datetime.now().isoformat()
        existing_data.append(new_creator)
        
        # Save back to GCS
        blob.upload_from_string(
            json.dumps(existing_data, indent=2),
            content_type="application/json"
        )
        
        logger.info(f"Added creator profile: {new_creator['name']}")
        return new_creator
    except Exception as e:
        logger.error(f"Error adding creator profile: {e}")
        raise HTTPException(status_code=500, detail="Failed to add creator profile")

@app.put("/linkedin/creators/{creator_id}", response_model=CreatorProfile, summary="Update creator profile", tags=["Creator Management"])
async def update_creator_profile(creator_id: int, creator: CreatorProfile):
    """Update an existing creator profile"""
    try:
        client = get_gcs_client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob("linkedin/creators.json")
        
        if not blob.exists():
            raise HTTPException(status_code=404, detail="Creator database not found")
        
        data = json.loads(blob.download_as_text())
        
        # Find and update the creator
        for i, item in enumerate(data):
            if item['id'] == creator_id:
                updated_creator = creator.dict()
                updated_creator['id'] = creator_id
                updated_creator['key'] = generate_creator_key(creator.name)
                updated_creator['created_at'] = item.get('created_at', datetime.now().isoformat())
                data[i] = updated_creator
                
                # Save back to GCS
                blob.upload_from_string(
                    json.dumps(data, indent=2),
                    content_type="application/json"
                )
                
                logger.info(f"Updated creator profile: {updated_creator['name']}")
                return updated_creator
        
        raise HTTPException(status_code=404, detail="Creator not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating creator profile: {e}")
        raise HTTPException(status_code=500, detail="Failed to update creator profile")

@app.delete("/linkedin/creators/{creator_id}", summary="Delete creator profile", tags=["Creator Management"])
async def delete_creator_profile(creator_id: int):
    """Delete a creator profile (prevents deletion of default creators)"""
    try:
        # Protect default creators (IDs 1, 2, 3)
        if creator_id <= 3:
            raise HTTPException(status_code=400, detail="Cannot delete default creator profiles")
        
        client = get_gcs_client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob("linkedin/creators.json")
        
        if not blob.exists():
            raise HTTPException(status_code=404, detail="Creator database not found")
        
        data = json.loads(blob.download_as_text())
        original_length = len(data)
        data = [item for item in data if item['id'] != creator_id]
        
        if len(data) == original_length:
            raise HTTPException(status_code=404, detail="Creator not found")
        
        # Save back to GCS
        blob.upload_from_string(
            json.dumps(data, indent=2),
            content_type="application/json"
        )
        
        logger.info(f"Deleted creator profile with ID: {creator_id}")
        return {"message": "Creator profile deleted successfully", "deleted_id": creator_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting creator profile: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete creator profile")

# Content Generation Endpoint
@app.post("/linkedin/generate", response_model=ContentGenerationResponse, summary="Generate LinkedIn content", tags=["Content Generation"])
async def generate_linkedin_content(request: ContentGenerationRequest):
    """Generate LinkedIn content using AI based on creator style and research data"""
    try:
        logger.info(f"Generating content for creator: {request.creator_key}")
        
        # Get creator profile
        client = get_gcs_client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        
        creators_blob = bucket.blob("linkedin/creators.json")
        if not creators_blob.exists():
            raise HTTPException(status_code=404, detail="Creator profiles not found")
        
        creators_data = json.loads(creators_blob.download_as_text())
        creator = next((c for c in creators_data if c['key'] == request.creator_key), None)
        
        if not creator:
            available_keys = [c['key'] for c in creators_data]
            raise HTTPException(
                status_code=404, 
                detail=f"Creator '{request.creator_key}' not found. Available: {available_keys}"
            )
        
        # Get research data
        research_data = []
        if request.research_ids:
            research_blob = bucket.blob("linkedin/research.json")
            if research_blob.exists():
                all_research = json.loads(research_blob.download_as_text())
                research_data = [r for r in all_research if r['id'] in request.research_ids]
        
        # If no specific research requested, get the most recent 5 items
        if not research_data:
            research_blob = bucket.blob("linkedin/research.json")
            if research_blob.exists():
                all_research = json.loads(research_blob.download_as_text())
                research_data = sorted(all_research, key=lambda x: x.get('created_at', ''), reverse=True)[:5]
        
        logger.info(f"Using {len(research_data)} research items for content generation")
        
        # Build comprehensive prompt for Groq
        research_context = ""
        if research_data:
            research_context = "\n".join([
                f"Research {i+1}:\n"
                f"Topic: {r['topic']}\n"
                f"Findings: {r['findings']}\n"
                f"Data: {r['data']}\n"
                f"Source: {r['source']}\n"
                for i, r in enumerate(research_data)
            ])
        else:
            research_context = "No specific research data provided. Use general business and marketing insights."
        
        system_prompt = f"""You are a LinkedIn content creator that perfectly mimics the style of {creator['name']}. 

CREATOR STYLE PROFILE:
- Tone: {creator['tone']}
- Structure: {creator['structure']}
- Language: {creator['language']}
- Typical Length: {creator['length']}
- Common Hooks: {creator['hooks']}
- Common Endings: {creator['endings']}
- Key Characteristics: {creator['characteristics']}

RESEARCH DATA TO INCORPORATE:
{research_context}

CRITICAL INSTRUCTIONS:
1. Create content that sounds EXACTLY like {creator['name']} would write it
2. If research data is provided, incorporate insights from ALL research items
3. Make each variation distinct while maintaining the creator's authentic voice
4. Include specific data points and findings from the research
5. Create engaging, actionable content that drives LinkedIn engagement

Create TWO different variations each for LinkedIn posts and video scripts.

Return ONLY a valid JSON response with this exact structure:
{{
  "linkedinPosts": ["first post variation", "second post variation"],
  "videoScripts": ["first video script variation", "second video script variation"],
  "hashtags": ["relevant", "hashtags", "for", "content"],
  "engagement_tips": ["tip1", "tip2", "tip3"]
}}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Create {creator['name']}-style content about: {request.prompt}"}
        ]
        
        # Generate content using Groq
        ai_response = await query_groq(messages, max_tokens=3000, temperature=0.8)
        
        # Parse JSON response
        try:
            content_data = json.loads(ai_response)
            
            # Validate required fields
            required_fields = ["linkedinPosts", "videoScripts", "hashtags", "engagement_tips"]
            for field in required_fields:
                if field not in content_data:
                    content_data[field] = []
            
            # Ensure we have at least 2 variations
            if len(content_data["linkedinPosts"]) < 2:
                content_data["linkedinPosts"] = content_data["linkedinPosts"] + content_data["linkedinPosts"]
            if len(content_data["videoScripts"]) < 2:
                content_data["videoScripts"] = content_data["videoScripts"] + content_data["videoScripts"]
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from AI response: {e}")
            # Fallback content
            content_data = {
                "linkedinPosts": [
                    f"Content generated in {creator['name']}'s style: {ai_response[:500]}...",
                    f"Alternative version: {ai_response[:500]}..."
                ],
                "videoScripts": [
                    f"Video script for {creator['name']}: {ai_response[:300]}...",
                    f"Alternative video script: {ai_response[:300]}..."
                ],
                "hashtags": ["#LinkedIn", "#Content", "#Business", "#AI"],
                "engagement_tips": [
                    "Post during peak hours (8-10 AM or 12-1 PM)",
                    "Engage with comments within first 2 hours",
                    "Ask questions to encourage responses"
                ]
            }
        
        # Save to prompt history
        try:
            history_blob = bucket.blob("linkedin/prompt_history.json")
            history_data = []
            if history_blob.exists():
                history_data = json.loads(history_blob.download_as_text())
            
            history_item = {
                "id": len(history_data) + 1,
                "prompt": request.prompt,
                "creator_name": creator['name'],
                "research_topics": [r['topic'] for r in research_data],
                "response_data": content_data,
                "created_at": datetime.now().isoformat()
            }
            history_data.insert(0, history_item)  # Add to beginning
            
            # Keep only last 100 items
            history_data = history_data[:100]
            
            history_blob.upload_from_string(
                json.dumps(history_data, indent=2),
                content_type="application/json"
            )
            logger.info("Saved generation to prompt history")
        except Exception as e:
            logger.warning(f"Failed to save prompt history: {e}")
        
        response = ContentGenerationResponse(
            linkedin_posts=content_data.get("linkedinPosts", [])[:2],
            video_scripts=content_data.get("videoScripts", [])[:2],
            hashtags=content_data.get("hashtags", []),
            engagement_tips=content_data.get("engagement_tips", []),
            creator_name=creator['name'],
            research_used=[r['topic'] for r in research_data]
        )
        
        logger.info(f"Successfully generated content for {creator['name']}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating content: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate content: {str(e)}")

# Prompt History Endpoint
@app.get("/linkedin/history", summary="Get prompt generation history", tags=["Content Generation"])
async def get_prompt_history():
    """Get the history of all content generations"""
    try:
        client = get_gcs_client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob("linkedin/prompt_history.json")
        
        if not blob.exists():
            logger.info("Prompt history not found, returning empty list")
            return []
        
        data = json.loads(blob.download_as_text())
        logger.info(f"Retrieved {len(data)} prompt history items")
        return data
    except Exception as e:
        logger.error(f"Error getting prompt history: {e}")
        return []

# Analytics and Stats Endpoints
@app.get("/linkedin/stats", summary="Get usage statistics", tags=["Analytics"])
async def get_linkedin_stats():
    """Get statistics about research items, creators, and content generations"""
    try:
        client = get_gcs_client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        
        # Count research items
        research_count = 0
        research_blob = bucket.blob("linkedin/research.json")
        if research_blob.exists():
            research_data = json.loads(research_blob.download_as_text())
            research_count = len(research_data)
        
        # Count creators
        creator_count = 0
        creators_blob = bucket.blob("linkedin/creators.json")
        if creators_blob.exists():
            creators_data = json.loads(creators_blob.download_as_text())
            creator_count = len(creators_data)
        
        # Count generations
        generation_count = 0
        most_used_creator = "None"
        history_blob = bucket.blob("linkedin/prompt_history.json")
        if history_blob.exists():
            history_data = json.loads(history_blob.download_as_text())
            generation_count = len(history_data)
            
            # Find most used creator
            if history_data:
                creator_usage = {}
                for item in history_data:
                    creator_name = item.get('creator_name', 'Unknown')
                    creator_usage[creator_name] = creator_usage.get(creator_name, 0) + 1
                most_used_creator = max(creator_usage, key=creator_usage.get)
        
        return {
            "research_items": research_count,
            "creator_profiles": creator_count,
            "content_generations": generation_count,
            "most_used_creator": most_used_creator,
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {
            "research_items": 0,
            "creator_profiles": 0,
            "content_generations": 0,
            "most_used_creator": "None",
            "last_updated": datetime.now().isoformat()
        }

# Bulk Operations
@app.post("/linkedin/research/bulk", summary="Add multiple research items", tags=["Research Management"])
async def add_bulk_research(research_items: List[ResearchItem]):
    """Add multiple research items at once"""
    try:
        client = get_gcs_client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob("linkedin/research.json")
        
        # Get existing data
        existing_data = []
        if blob.exists():
            existing_data = json.loads(blob.download_as_text())
        
        added_items = []
        current_max_id = max([item.get('id', 0) for item in existing_data], default=0)
        
        for i, research in enumerate(research_items):
            new_item = research.dict()
            new_item['id'] = current_max_id + i + 1
            new_item['created_at'] = datetime.now().isoformat()
            existing_data.append(new_item)
            added_items.append(new_item)
        
        # Save back to GCS
        blob.upload_from_string(
            json.dumps(existing_data, indent=2),
            content_type="application/json"
        )
        
        logger.info(f"Added {len(added_items)} research items in bulk")
        return {"message": f"Added {len(added_items)} research items", "items": added_items}
    except Exception as e:
        logger.error(f"Error adding bulk research: {e}")
        raise HTTPException(status_code=500, detail="Failed to add research items")

@app.delete("/linkedin/research/bulk", summary="Delete multiple research items", tags=["Research Management"])
async def delete_bulk_research(research_ids: List[int] = Body(...)):
    """Delete multiple research items at once"""
    try:
        client = get_gcs_client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob("linkedin/research.json")
        
        if not blob.exists():
            raise HTTPException(status_code=404, detail="Research database not found")
        
        data = json.loads(blob.download_as_text())
        original_length = len(data)
        data = [item for item in data if item['id'] not in research_ids]
        deleted_count = original_length - len(data)
        
        # Save back to GCS
        blob.upload_from_string(
            json.dumps(data, indent=2),
            content_type="application/json"
        )
        
        logger.info(f"Deleted {deleted_count} research items in bulk")
        return {"message": f"Deleted {deleted_count} research items", "deleted_ids": research_ids}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting bulk research: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete research items")

# Search and Filter Endpoints
@app.get("/linkedin/research/search", summary="Search research items", tags=["Research Management"])
async def search_research(q: str = "", tags: str = "", limit: int = 50):
    """Search research items by query and tags"""
    try:
        client = get_gcs_client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob("linkedin/research.json")
        
        if not blob.exists():
            return []
        
        data = json.loads(blob.download_as_text())
        
        # Filter by query
        if q:
            q_lower = q.lower()
            data = [
                item for item in data
                if (q_lower in item.get('topic', '').lower() or
                    q_lower in item.get('findings', '').lower() or
                    q_lower in item.get('data', '').lower() or
                    q_lower in item.get('source', '').lower())
            ]
        
        # Filter by tags
        if tags:
            tag_list = [t.strip().lower() for t in tags.split(',')]
            data = [
                item for item in data
                if any(tag in item.get('tags', '').lower() for tag in tag_list)
            ]
        
        # Sort by relevance and date
        data = sorted(data, key=lambda x: x.get('created_at', ''), reverse=True)
        
        # Apply limit
        data = data[:limit]
        
        logger.info(f"Search returned {len(data)} results for query: '{q}', tags: '{tags}'")
        return data
    except Exception as e:
        logger.error(f"Error searching research: {e}")
        return []

@app.get("/linkedin/creators/search", summary="Search creator profiles", tags=["Creator Management"])
async def search_creators(q: str = ""):
    """Search creator profiles by name or characteristics"""
    try:
        client = get_gcs_client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob("linkedin/creators.json")
        
        if not blob.exists():
            return []
        
        data = json.loads(blob.download_as_text())
        
        if q:
            q_lower = q.lower()
            data = [
                creator for creator in data
                if (q_lower in creator.get('name', '').lower() or
                    q_lower in creator.get('tone', '').lower() or
                    q_lower in creator.get('characteristics', '').lower())
            ]
        
        logger.info(f"Creator search returned {len(data)} results for query: '{q}'")
        return data
    except Exception as e:
        logger.error(f"Error searching creators: {e}")
        return []

# Content Templates and Suggestions
@app.get("/linkedin/templates", summary="Get content templates", tags=["Content Generation"])
async def get_content_templates():
    """Get predefined content templates for different use cases"""
    templates = {
        "business_insight": {
            "prompt": "Share a business insight from recent market research",
            "description": "Create content about business trends and insights",
            "suggested_creators": ["gary-v", "seth-godin"]
        },
        "leadership_lesson": {
            "prompt": "Create a leadership lesson based on recent findings",
            "description": "Inspirational content about leadership and team building",
            "suggested_creators": ["simon-sinek"]
        },
        "industry_trend": {
            "prompt": "Discuss current industry trends and their implications",
            "description": "Analysis of industry trends and future predictions",
            "suggested_creators": ["gary-v", "seth-godin"]
        },
        "personal_story": {
            "prompt": "Share a personal story that relates to professional growth",
            "description": "Personal narrative with business lessons",
            "suggested_creators": ["gary-v", "simon-sinek"]
        },
        "how_to_guide": {
            "prompt": "Create a how-to guide based on research data",
            "description": "Educational content with actionable steps",
            "suggested_creators": ["seth-godin"]
        }
    }
    
    return templates

@app.get("/linkedin/suggestions", summary="Get content suggestions", tags=["Content Generation"])
async def get_content_suggestions():
    """Get AI-powered content suggestions based on recent research"""
    try:
        client = get_gcs_client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        research_blob = bucket.blob("linkedin/research.json")
        
        if not research_blob.exists():
            return {"suggestions": [], "message": "Add research data to get personalized suggestions"}
        
        research_data = json.loads(research_blob.download_as_text())
        recent_research = sorted(research_data, key=lambda x: x.get('created_at', ''), reverse=True)[:5]
        
        if not recent_research:
            return {"suggestions": [], "message": "No recent research found"}
        
        # Generate suggestions based on recent research
        research_topics = [r['topic'] for r in recent_research]
        
        suggestions = [
            f"Create a post about the latest trends in {research_topics[0] if research_topics else 'your industry'}",
            f"Share insights from your recent research on {research_topics[1] if len(research_topics) > 1 else 'market developments'}",
            f"Write a leadership piece inspired by {research_topics[2] if len(research_topics) > 2 else 'current challenges'}",
            "Create a how-to guide based on your latest findings",
            "Share a personal story that relates to your recent research"
        ]
        
        return {
            "suggestions": suggestions[:3],
            "based_on_research": research_topics,
            "message": f"Suggestions based on {len(recent_research)} recent research items"
        }
    except Exception as e:
        logger.error(f"Error getting suggestions: {e}")
        return {"suggestions": [], "message": "Unable to generate suggestions at this time"}

# Export and Import Endpoints
@app.get("/linkedin/export", summary="Export all data", tags=["Data Management"])
async def export_all_data():
    """Export all research, creators, and history data"""
    try:
        client = get_gcs_client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        
        export_data = {
            "research": [],
            "creators": [],
            "history": [],
            "exported_at": datetime.now().isoformat(),
            "version": "1.0"
        }
        
        # Get research data
        research_blob = bucket.blob("linkedin/research.json")
        if research_blob.exists():
            export_data["research"] = json.loads(research_blob.download_as_text())
        
        # Get creators data
        creators_blob = bucket.blob("linkedin/creators.json")
        if creators_blob.exists():
            export_data["creators"] = json.loads(creators_blob.download_as_text())
        
        # Get history data
        history_blob = bucket.blob("linkedin/prompt_history.json")
        if history_blob.exists():
            export_data["history"] = json.loads(history_blob.download_as_text())
        
        logger.info(f"Exported data: {len(export_data['research'])} research, {len(export_data['creators'])} creators, {len(export_data['history'])} history")
        return export_data
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        raise HTTPException(status_code=500, detail="Failed to export data")

@app.post("/linkedin/import", summary="Import data", tags=["Data Management"])
async def import_data(data: dict = Body(...)):
    """Import research, creators, and history data (replaces existing data)"""
    try:
        client = get_gcs_client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        
        imported_counts = {"research": 0, "creators": 0, "history": 0}
        
        # Import research data
        if "research" in data and isinstance(data["research"], list):
            research_blob = bucket.blob("linkedin/research.json")
            research_blob.upload_from_string(
                json.dumps(data["research"], indent=2),
                content_type="application/json"
            )
            imported_counts["research"] = len(data["research"])
        
        # Import creators data
        if "creators" in data and isinstance(data["creators"], list):
            creators_blob = bucket.blob("linkedin/creators.json")
            creators_blob.upload_from_string(
                json.dumps(data["creators"], indent=2),
                content_type="application/json"
            )
            imported_counts["creators"] = len(data["creators"])
        
        # Import history data
        if "history" in data and isinstance(data["history"], list):
            history_blob = bucket.blob("linkedin/prompt_history.json")
            history_blob.upload_from_string(
                json.dumps(data["history"], indent=2),
                content_type="application/json"
            )
            imported_counts["history"] = len(data["history"])
        
        logger.info(f"Imported data: {imported_counts}")
        return {
            "message": "Data imported successfully", 
            "imported": imported_counts,
            "imported_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error importing data: {e}")
        raise HTTPException(status_code=500, detail="Failed to import data")

# Admin and Maintenance Endpoints
@app.post("/linkedin/reset", summary="Reset all LinkedIn data", tags=["Data Management"])
async def reset_linkedin_data():
    """Delete all LinkedIn research, creators (except defaults), and history"""
    try:
        client = get_gcs_client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        
        deleted_items = []
        
        # Reset research
        research_blob = bucket.blob("linkedin/research.json")
        if research_blob.exists():
            research_blob.delete()
            deleted_items.append("research.json")
        
        # Reset history
        history_blob = bucket.blob("linkedin/prompt_history.json")
        if history_blob.exists():
            history_blob.delete()
            deleted_items.append("prompt_history.json")
        
        # Reset creators to defaults only
        default_creators = [
            {
                "id": 1,
                "name": "Gary Vaynerchuk",
                "key": "gary-v",
                "tone": "Direct, passionate, no-nonsense",
                "structure": "Hook → Personal story → Business insight → Call to action",
                "language": "Casual, uses \"you guys\", lots of energy",
                "length": "Medium to long posts",
                "hooks": "Look..., Here's the thing..., Real talk...",
                "endings": "What do you think?, Let me know in the comments, DM me your thoughts",
                "characteristics": "High energy, direct approach, business-focused, motivational",
                "created_at": datetime.now().isoformat()
            },
            {
                "id": 2,
                "name": "Simon Sinek",
                "key": "simon-sinek",
                "tone": "Inspirational, thoughtful, leader-focused",
                "structure": "Question → Story/Example → Leadership lesson → Reflection",
                "language": "Professional but warm, thought-provoking",
                "length": "Medium posts with clear paragraphs",
                "hooks": "Why is it that..., The best leaders..., I once worked with...",
                "endings": "What would you do?, Leadership is a choice, The choice is yours",
                "characteristics": "Leadership-focused, inspirational, storytelling, thoughtful questions",
                "created_at": datetime.now().isoformat()
            },
            {
                "id": 3,
                "name": "Seth Godin",
                "key": "seth-godin",
                "tone": "Wise, concise, marketing-focused",
                "structure": "Insight → Brief explanation → Broader implication",
                "language": "Concise, profound, marketing terminology",
                "length": "Short, punchy posts",
                "hooks": "The thing is..., Here's what I learned..., Marketing is...",
                "endings": "Worth considering., Just saying., Think about it.",
                "characteristics": "Concise wisdom, marketing insights, thought-provoking, minimal but impactful",
                "created_at": datetime.now().isoformat()
            }
        ]
        
        creators_blob = bucket.blob("linkedin/creators.json")
        creators_blob.upload_from_string(
            json.dumps(default_creators, indent=2),
            content_type="application/json"
        )
        deleted_items.append("creators.json (reset to defaults)")
        
        logger.info(f"Reset LinkedIn data: {deleted_items}")
        return {
            "message": "LinkedIn data reset successfully",
            "reset_items": deleted_items,
            "reset_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error resetting LinkedIn data: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset data")

# Health and Status Endpoints
@app.get("/linkedin/health", summary="Check LinkedIn API health", tags=["System"])
async def check_linkedin_health():
    """Comprehensive health check for LinkedIn API components"""
    health_status = {
        "overall": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {}
    }
    
    try:
        # Check GCS connectivity
        client = get_gcs_client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        bucket.reload()  # Test connection
        health_status["components"]["gcs"] = {"status": "healthy", "message": "Connected"}
    except Exception as e:
        health_status["components"]["gcs"] = {"status": "unhealthy", "message": str(e)}
        health_status["overall"] = "degraded"
    
    try:
        # Check Groq API
        if GROQ_API_KEY:
            # Simple test message
            test_messages = [{"role": "user", "content": "Hello"}]
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    GROQ_API_URL,
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json={"model": GROQ_MODEL, "messages": test_messages, "max_tokens": 10}
                )
                if response.status_code == 200:
                    health_status["components"]["groq"] = {"status": "healthy", "message": "API responding"}
                else:
                    health_status["components"]["groq"] = {"status": "unhealthy", "message": f"API error: {response.status_code}"}
                    health_status["overall"] = "degraded"
        else:
            health_status["components"]["groq"] = {"status": "unhealthy", "message": "API key not configured"}
            health_status["overall"] = "degraded"
    except Exception as e:
        health_status["components"]["groq"] = {"status": "unhealthy", "message": str(e)}
        health_status["overall"] = "degraded"
    
    # Check data stores
    try:
        bucket = client.bucket(GCS_BUCKET_NAME)
        data_status = {}
        
        # Check research data
        research_blob = bucket.blob("linkedin/research.json")
        data_status["research"] = "exists" if research_blob.exists() else "empty"
        
        # Check creators data
        creators_blob = bucket.blob("linkedin/creators.json")
        data_status["creators"] = "exists" if creators_blob.exists() else "empty"
        
        # Check history data
        history_blob = bucket.blob("linkedin/prompt_history.json")
        data_status["history"] = "exists" if history_blob.exists() else "empty"
        
        health_status["components"]["data_stores"] = {"status": "healthy", "stores": data_status}
    except Exception as e:
        health_status["components"]["data_stores"] = {"status": "unhealthy", "message": str(e)}
        health_status["overall"] = "degraded"
    
    return health_status

# Version and Info Endpoints
@app.get("/linkedin/info", summary="Get API information", tags=["System"])
async def get_api_info():
    """Get API version and configuration information"""
    return {
        "name": "LinkedIn Content Creator API",
        "version": "1.0.0",
        "description": "AI-powered LinkedIn content generation with persistent storage",
        "features": [
            "Research database management",
            "Creator profile management", 
            "AI content generation",
            "Prompt history tracking",
            "Bulk operations",
            "Search and filtering",
            "Data export/import"
        ],
        "ai_model": GROQ_MODEL,
        "storage": "Google Cloud Storage",
        "endpoints": {
            "research": 6,
            "creators": 5, 
            "content": 3,
            "admin": 4,
            "system": 3
        },
        "last_deployed": datetime.now().isoformat()
    }

# Error Handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found", 
            "message": "The requested resource was not found",
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )

# Development and Testing Endpoints (can be removed in production)
@app.get("/linkedin/test", summary="Test endpoint", tags=["Development"])
async def test_endpoint():
    """Simple test endpoint for development and debugging"""
    return {
        "message": "LinkedIn Content Creator API is working!",
        "timestamp": datetime.now().isoformat(),
        "environment": {
            "gcs_bucket": GCS_BUCKET_NAME,
            "groq_configured": bool(GROQ_API_KEY),
            "base_url": BASE_URL
        }
    }

@app.post("/linkedin/test/generate", summary="Test content generation", tags=["Development"])
async def test_content_generation():
    """Test the content generation pipeline with sample data"""
    try:
        # Use a default creator for testing
        test_request = ContentGenerationRequest(
            prompt="Create a post about the importance of AI in business",
            creator_key="gary-v",
            research_ids=[]
        )
        
        result = await generate_linkedin_content(test_request)
        return {
            "test_result": "success",
            "generated_content": result,
            "message": "Content generation test completed successfully"
        }
    except Exception as e:
        logger.error(f"Test generation failed: {e}")
        return {
            "test_result": "failed",
            "error": str(e),
            "message": "Content generation test failed"
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)