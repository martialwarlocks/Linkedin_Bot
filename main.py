import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Any, Dict
from contextlib import asynccontextmanager
import json
from datetime import datetime
import re
from io import BytesIO
import urllib.parse
import uuid
from google.cloud import storage
from google.oauth2 import service_account
import math

# --- FastAPI Imports ---
from fastapi import FastAPI, HTTPException, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx

# --- Supabase & GCS Imports ---
from supabase.client import create_client, Client
from google.api_core.exceptions import GoogleAPIError

# --- Libraries for Text Extraction ---
try:
    import pypdf
    import docx
    TEXT_EXTRACTION_ENABLED = True
except ImportError:
    TEXT_EXTRACTION_ENABLED = False

# --- Libraries for Image Generation ---
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import FancyBboxPatch
    import seaborn as sns
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import io
    import base64
    IMAGE_GENERATION_ENABLED = True
except ImportError:
    IMAGE_GENERATION_ENABLED = False

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# --- Configuration ---
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "linkedin-bot-documents")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
BASE_URL = "http://localhost:8080"

# --- API Keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY", "sk-Fu8BOqrEBB10vrRBc2sRTtZeMSBbl9NJKCoFzmqHbIvdcIma")

# --- Supabase Configuration (Hardcoded as requested) ---
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://qgyqkgmdnwfcnzzuzict.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFneXFrZ21kbndmY256enV6aWN0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTM3NzkyOTcsImV4cCI6MjA2OTM1NTI5N30.d9VFOHZsWDxhqY8UM0jvx5pGJVVOSkgHVFODL16Nc6s")
    
# --- Global Clients ---
supabase_client = None
gcs_client = None

# --- Client Singletons ---
def get_supabase_client() -> Client:
    global supabase_client
    if supabase_client is None:
        logger.info("Initializing Supabase client.")
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return supabase_client

def get_gcs_client():
    # Try to get credentials from JSON string first (for Cloud Run)
    credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if credentials_json:
        try:
            credentials_info = json.loads(credentials_json)
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
            return storage.Client(credentials=credentials)
        except Exception as e:
            logger.error(f"Failed to parse GOOGLE_APPLICATION_CREDENTIALS_JSON: {e}")
    
    # Fallback to file path (for local development)
    key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/Users/yatins/Downloads/linkedin-bot-468005-key.json")
    if os.path.exists(key_path):
        credentials = service_account.Credentials.from_service_account_file(key_path)
        return storage.Client(credentials=credentials)
    
    # Use default credentials (for local development with gcloud auth)
    return storage.Client()

# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API Startup: Initializing resources.")
    try:
        get_supabase_client()
        logger.info("Supabase client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
    
    try:
        get_gcs_client()
        logger.info("GCS client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize GCS client: {e}")
    
    yield
    logger.info("API Shutdown.")

app = FastAPI(
    title="LinkedIn Content Creator API (Supabase + GCS)",
    version="2.3.0",
    description="FastAPI backend using Supabase for metadata and GCS for file storage.",
    lifespan=lifespan
)

# Configure maximum request body size for large file uploads (100MB)
from fastapi import Request
from fastapi.responses import JSONResponse

@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    if request.method == "POST" and "/linkedin/upload-document" in str(request.url):
        # Allow up to 100MB for document uploads
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 100 * 1024 * 1024:  # 100MB
            return JSONResponse(
                status_code=413,
                content={"detail": "File too large. Maximum size is 100MB."}
            )
    response = await call_next(request)
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---

class UploadResponse(BaseModel):
    message: str
    filename: str
    gcs_url: str
    research_items_added: int

class ResearchItem(BaseModel):
    id: Optional[int] = None
    topic: str
    findings: str
    source: Optional[str] = None
    data: Optional[str] = None
    tags: Optional[str] = None
    created_at: Optional[datetime] = None

class CreatorProfile(BaseModel):
    id: Optional[int] = None
    name: str
    key: str
    tone: Optional[str] = None
    structure: Optional[str] = None
    language: Optional[str] = None
    length: Optional[str] = None
    hooks: Optional[str] = None
    endings: Optional[str] = None
    characteristics: Optional[str] = None
    created_at: Optional[datetime] = None

class ContentGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    creator_key: str = Field(..., min_length=1)
    research_ids: Optional[List[int]] = []
    # New customization options
    tone: Optional[str] = Field(None, description="Specific tone override (e.g., 'professional', 'casual', 'motivational', 'analytical')")
    content_format: Optional[str] = Field(None, description="Content format preference (e.g., 'paragraph', 'bullet_points', 'numbered_list')")
    content_style: Optional[str] = Field(None, description="Content style (e.g., 'direct', 'storytelling', 'data_driven', 'conversational')")
    include_statistics: Optional[bool] = Field(False, description="Whether to include numerical data and statistics")
    post_length: Optional[str] = Field(None, description="Preferred post length (e.g., 'short', 'medium', 'long')")
    call_to_action: Optional[str] = Field(None, description="Specific call to action preference")

class ContentGenerationResponse(BaseModel):
    linkedin_posts: List[str]
    video_scripts: List[str]
    hashtags: List[str]
    engagement_tips: List[str]
    creator_name: str
    research_used: List[str]
    contextual_image_url: Optional[str] = None
    image_source_link: Optional[str] = None

class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Description of the image to generate")
    style: Optional[str] = Field("professional", description="Style of the image (e.g., professional, artistic, technical)")
    aspect_ratio: Optional[str] = Field("16:9", description="Aspect ratio of the image")
    size: Optional[str] = Field("1024x1024", description="Size of the image")
    content_type: Optional[str] = Field("diagram", description="Type of content (diagram, graph, chart, illustration)")

class ImageGenerationResponse(BaseModel):
    image_url: str
    prompt: str
    style: str
    size: str
    content_type: str
    generated_at: datetime
    message: str

class CrawlRequest(BaseModel):
    url: str

# --- Helper Functions ---
def extract_text_from_file(content: bytes, filename: str) -> str:
    if not TEXT_EXTRACTION_ENABLED:
        raise ImportError("pypdf and python-docx are not installed.")
    file_extension = Path(filename).suffix.lower()
    text_content = ""
    try:
        if file_extension == '.pdf':
            with BytesIO(content) as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    text_content += page.extract_text() or ""
        elif file_extension == '.docx':
            with BytesIO(content) as f:
                document = docx.Document(f)
                for para in document.paragraphs:
                    text_content += para.text + "\n"
        elif file_extension == '.txt':
            text_content = content.decode('utf-8')
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        return text_content
    except Exception as e:
        raise ValueError(f"Could not extract text from file: {filename}")

def chunk_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 200) -> List[str]:
    if not text: return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

async def query_groq(messages: List[dict], max_tokens: int = 3000, temperature: float = 0.8) -> str:
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={"model": GROQ_MODEL, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

# Removed external AI image generation. We now generate deterministic analytics
# images locally with Matplotlib for maximum readability and control.

def extract_visual_data(content_data: dict, research_data: list) -> dict:
    """Extract numeric arrays and labels for charts from content/research."""
    data = {
        "content_counts": {
            "posts": len(content_data.get('linkedin_posts', [])) if content_data else 0,
            "scripts": len(content_data.get('video_scripts', [])) if content_data else 0,
            "hashtags": len(content_data.get('hashtags', [])) if content_data else 0,
            "tips": len(content_data.get('engagement_tips', [])) if content_data else 0,
        },
        "post_word_counts": [],
        "hashtags": content_data.get('hashtags', [])[:8] if content_data else [],
        "top_topics": [],
        "topic_word_counts": [],
    }

    # Post word counts
    for post in content_data.get('linkedin_posts', []) if content_data else []:
        data["post_word_counts"].append(len(post.split()))

    # Research topics and word counts
    if research_data:
        analysis = analyze_research_data(research_data)
        data["top_topics"] = analysis.get('topics', [])[:6]
        data["topic_word_counts"] = analysis.get('word_counts', [])[:6]
    return data

def build_visual_prompt_from_content(content_data: dict, research_data: list) -> str:
    """
    Build a compact, structured prompt for AI infographic generation based on actual content and research.
    """
    # Extract numeric arrays and labels
    d = extract_visual_data(content_data, research_data)
    num_posts = d["content_counts"]["posts"]
    num_scripts = d["content_counts"]["scripts"]
    num_hashtags = d["content_counts"]["hashtags"]
    num_tips = d["content_counts"]["tips"]
    hashtags_text = ", ".join([f"#{h}" for h in d["hashtags"]]) if d["hashtags"] else ""
    topics_preview = ", ".join(d["top_topics"]) if d["top_topics"] else ""
    post_counts_str = ", ".join(str(x) for x in d["post_word_counts"]) if d["post_word_counts"] else ""
    topic_counts_str = ", ".join(str(x) for x in d["topic_word_counts"]) if d["topic_word_counts"] else ""

    # Build a clear instruction for multiple charts
    prompt = (
        "You are a designer of data dashboards. Generate a crisp infographic titled 'Content Intelligence'. "
        "Strictly follow this layout (4 panels): "
        "Panel A (top-left): Pie chart — labels: Posts, Scripts, Hashtags, Tips; values: "
        f"[{num_posts}, {num_scripts}, {num_hashtags}, {num_tips}]. "
        "Panel B (top-right): Horizontal bar chart — labels (top topics): "
        f"[{topics_preview}] values: [{topic_counts_str}] (omit if empty). "
        "Panel C (bottom-left): Histogram of LinkedIn post word counts — data: "
        f"[{post_counts_str}] with readable bucket labels and counts. "
        "Panel D (bottom-right): List up to 4 engagement tips as bullets. "
        f"Hashtags to show small inline: {hashtags_text}. "
        "Style: white background, clean grid lines, large sans-serif fonts, readable axis titles, clear legend, high contrast colors: "
        "blue #2563eb, green #059669, amber #d97706, violet #8b5cf6. No photos or textures. No watermarks."
    )
    return prompt

# === Topic-based Visuals (GDP, Virus, etc.) ===
def _detect_country_code(text: str) -> str:
    """Very small country alias mapping for quick wins. Extend as needed."""
    aliases = {
        'india': 'IND', 'bharat': 'IND',
        'united states': 'USA', 'usa': 'USA', 'us': 'USA', 'america': 'USA',
        'china': 'CHN', 'prc': 'CHN',
        'united kingdom': 'GBR', 'uk': 'GBR', 'britain': 'GBR',
        'germany': 'DEU', 'france': 'FRA', 'japan': 'JPN'
    }
    t = text.lower()
    for name, code in aliases.items():
        if name in t:
            return code
    return ''

async def _fetch_worldbank_timeseries(country_code: str, indicator: str, years: int = 15) -> tuple:
    """Fetch last N years of data from World Bank. Returns (years_list, values_list)."""
    try:
        url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}?format=json&per_page=70"
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(url)
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list) or len(data) < 2:
                return [], []
            series = data[1]
            pairs = []
            for entry in series:
                year = entry.get('date')
                value = entry.get('value')
                if year and value is not None:
                    pairs.append((int(year), float(value)))
            pairs.sort()
            if years:
                pairs = pairs[-years:]
            years_list = [p[0] for p in pairs]
            values_list = [p[1] for p in pairs]
            return years_list, values_list
    except Exception as e:
        logger.warning(f"World Bank fetch failed: {e}")
        return [], []

def _save_fig_to_gcs(fig, folder: str) -> str:
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    img_buffer.seek(0)
    gcs = get_gcs_client()
    bucket = gcs.bucket(GCS_BUCKET_NAME)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{folder}/{timestamp}_{unique_id}.png"
    blob = bucket.blob(filename)
    blob.upload_from_string(img_buffer.getvalue(), content_type='image/png')
    blob.make_public()
    plt.close(fig)
    return blob.public_url

def _create_virus_visual(topic_label: str = 'Virus structure') -> str:
    if not IMAGE_GENERATION_ENABLED:
        raise ImportError("matplotlib not installed")
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
    ax.set_aspect('equal')
    ax.axis('off')
    center_x, center_y = 0.0, 0.0
    radius = 1.8
    # core
    core = plt.Circle((center_x, center_y), radius, color='#ef4444', alpha=0.2, ec='#ef4444', lw=2)
    ax.add_patch(core)
    # spikes
    for i in range(40):
        angle = (2 * math.pi / 40) * i
        x0 = center_x + math.cos(angle) * radius
        y0 = center_y + math.sin(angle) * radius
        x1 = center_x + math.cos(angle) * (radius + 0.7)
        y1 = center_y + math.sin(angle) * (radius + 0.7)
        ax.plot([x0, x1], [y0, y1], color='#b91c1c', lw=3)
        cap = plt.Circle((x1, y1), 0.12, color='#dc2626')
        ax.add_patch(cap)
    ax.text(0, -2.6, topic_label, ha='center', va='center', fontsize=18, fontweight='bold', color='#1f2937')
    return _save_fig_to_gcs(fig, 'topic_visuals')

async def try_topic_based_visual(posts_only: dict, context_text: str = "") -> tuple:
    """Attempt to create a topic-driven visual (GDP, virus, etc.) using
    the LinkedIn post and any supplemental research context."""
    posts_text = " ".join(posts_only.get('linkedin_posts', []) or [])
    tl = (posts_text + " " + (context_text or "")).lower()

    # GDP line chart for a country
    if 'gdp' in tl or 'gross domestic product' in tl:
        code = _detect_country_code(tl)
        if code:
            years, values = await _fetch_worldbank_timeseries(code, 'NY.GDP.MKTP.CD', years=15)
            if years and values and any(v for v in values):
                plt.style.use('default')
                fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
                ax.plot(years, values, marker='o', color='#2563eb', lw=3)
                ax.fill_between(years, values, alpha=0.2, color='#2563eb')
                ax.set_title(f"{code} GDP (current US$)")
                ax.set_xlabel('Year')
                ax.set_ylabel('US$')
                ax.grid(True, alpha=0.3)
                url = _save_fig_to_gcs(fig, 'topic_visuals')
                return url, f"World Bank GDP for {code}"

    # Virus schematic
    if 'virus' in tl or 'viral' in tl or 'pathogen' in tl:
        url = _create_virus_visual('Virus schematic')
        return url, 'Schematic virus visual'

    return None, None

def create_advanced_infographic(content_data: dict, research_data: list) -> str:
    """
    Create a clean analytics dashboard comprised ONLY of charts/graphs derived
    from LinkedIn posts. No tips/hashtags/boxes.
    """
    try:
        if not IMAGE_GENERATION_ENABLED:
            raise ImportError("matplotlib, seaborn, and PIL are not installed.")

        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor='white')
        ax1, ax2, ax3, ax4 = axes.flatten()

        palette_primary = '#2563eb'
        palette_secondary = '#059669'
        palette_accent = '#d97706'
        palette_alt = '#8b5cf6'

        posts = content_data.get('linkedin_posts', []) or []

        # Chart 1: Word count histogram
        word_counts = [len(p.split()) for p in posts] if posts else []
        if word_counts:
            ax1.hist(word_counts, bins=min(10, max(3, len(set(word_counts)))), color=palette_primary, alpha=0.8)
            ax1.set_title('Post Word Count Distribution')
            ax1.set_xlabel('Words per post')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No posts available', ha='center', va='center')
            ax1.axis('off')

        # Chart 2: Top keyword frequency (simple)
        from collections import Counter
        words = []
        for p in posts:
            words.extend([w.lower() for w in p.split()])
        stop = {'the','and','or','but','in','on','at','to','for','of','with','by','is','are','was','were','be','been','a','an','as','if','when','where','why','how','this','that','these','those'}
        keywords = [w.strip('.,;:!?()[]"\'') for w in words if len(w) > 3 and w not in stop]
        counts = Counter(keywords).most_common(10)
        if counts:
            labels, values = zip(*counts)
            ax2.barh(labels, values, color=palette_secondary, alpha=0.85)
            ax2.invert_yaxis()
            ax2.set_title('Top Keywords in Posts')
            ax2.set_xlabel('Frequency')
            ax2.grid(True, axis='x', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No meaningful keywords', ha='center', va='center')
            ax2.axis('off')

        # Chart 3: Sentence count per post
        import re as _re
        sentence_counts = [len([s for s in _re.split(r'[.!?]\s+', p) if s.strip()]) for p in posts]
        if sentence_counts:
            ax3.plot(range(1, len(sentence_counts)+1), sentence_counts, marker='o', color=palette_accent)
            ax3.fill_between(range(1, len(sentence_counts)+1), sentence_counts, alpha=0.2, color=palette_accent)
            ax3.set_title('Sentences per Post (sequence)')
            ax3.set_xlabel('Post #')
            ax3.set_ylabel('Sentences')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No posts available', ha='center', va='center')
            ax3.axis('off')

        # Chart 4: Post length per post
        if word_counts:
            ax4.bar(range(1, len(word_counts)+1), word_counts, color=palette_alt, alpha=0.85)
            ax4.set_title('Words per Post')
            ax4.set_xlabel('Post #')
            ax4.set_ylabel('Words')
            ax4.grid(True, axis='y', alpha=0.3)
        else:
            ax4.axis('off')

        plt.tight_layout()

        # Save to bytes
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        img_buffer.seek(0)

        # Upload to GCS
        gcs = get_gcs_client()
        bucket = gcs.bucket(GCS_BUCKET_NAME)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        filename = f"infographics/{timestamp}_{unique_id}.png"
        blob = bucket.blob(filename)
        blob.upload_from_string(img_buffer.getvalue(), content_type='image/png')
        blob.make_public()

        plt.close(fig)
        return blob.public_url
    except Exception as e:
        logger.error(f"Error creating advanced infographic: {e}")
        raise Exception(f"Failed to create advanced infographic: {str(e)}")

def analyze_research_data(research_data: list) -> dict:
    """
    Analyze research data to extract meaningful insights for visualization
    """
    analysis = {
        'topics': [],
        'word_counts': [],
        'numerical_data': [],
        'key_phrases': [],
        'content_lengths': [],
        'source_types': []
    }
    
    for item in research_data:
        topic = item.get('topic', 'Unknown')
        findings = item.get('findings', '')
        source = item.get('source', '')
        
        # Extract topics
        analysis['topics'].append(topic[:30])  # Truncate long topics
        
        # Extract word counts
        word_count = len(findings.split())
        analysis['word_counts'].append(word_count)
        
        # Extract numerical data
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?', findings)
        if numbers:
            # Use the first significant number found
            for num in numbers:
                if float(num) > 1:  # Skip small numbers like 0.5
                    analysis['numerical_data'].append(float(num))
                    break
        
        # Extract key phrases (words that appear frequently)
        words = re.findall(r'\b[A-Za-z]{4,}\b', findings.lower())
        analysis['key_phrases'].extend(words[:5])  # First 5 meaningful words
        
        # Content length
        analysis['content_lengths'].append(len(findings))
        
        # Source type
        if 'pdf' in source.lower():
            analysis['source_types'].append('PDF')
        elif 'http' in source.lower():
            analysis['source_types'].append('Web')
        else:
            analysis['source_types'].append('Other')
    
    return analysis

def create_data_driven_visualization(research_data: list) -> str:
    """
    Create dynamic visualizations based on actual research data
    """
    try:
        if not IMAGE_GENERATION_ENABLED or not research_data:
            raise Exception("Cannot create visualization")
        
        # Analyze the research data
        analysis = analyze_research_data(research_data)
        
        # Determine the best visualization type based on data
        if len(analysis['numerical_data']) >= 3:
            # Create a bar chart with numerical data
            return create_numerical_bar_chart(analysis, research_data)
        elif len(analysis['topics']) >= 3:
            # Create a topic distribution chart
            return create_topic_distribution_chart(analysis, research_data)
        elif len(analysis['word_counts']) >= 2:
            # Create a content length comparison
            return create_content_length_chart(analysis, research_data)
        else:
            # Create a simple research overview
            return create_research_overview_chart(analysis, research_data)
            
    except Exception as e:
        logger.error(f"Error creating data-driven visualization: {e}")
        raise Exception(f"Failed to create data-driven visualization: {str(e)}")

def create_data_visualization(research_data: list) -> str:
    """
    Create a sketchy data visualization chart from research data
    """
    try:
        if not IMAGE_GENERATION_ENABLED or not research_data:
            raise Exception("Cannot create visualization")
        
        # Enable sketchy style
        plt.xkcd()
        
        # Extract numerical data if available
        data_points = []
        labels = []
        
        for item in research_data[:5]:  # Limit to 5 items
            topic = item.get('topic', 'Unknown')[:20]  # Truncate long titles
            findings = item.get('findings', '')
            
            # Try to extract numbers from findings
            import re
            numbers = re.findall(r'\d+(?:\.\d+)?', findings)
            if numbers:
                # Use the first number found
                value = float(numbers[0])
                data_points.append(value)
                labels.append(topic)
        
        if not data_points:
            # Fallback: create a simple bar chart with topic lengths
            for item in research_data[:5]:
                topic = item.get('topic', 'Unknown')[:20]
                findings = item.get('findings', '')
                data_points.append(len(findings))
                labels.append(topic)
        
        # Create the sketchy chart
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='#f8f9fa')
        
        # Create sketchy bar chart
        bars = ax.bar(range(len(data_points)), data_points, 
                     color=['#ff6b6b', '#4ecdc4', '#feca57', '#ff9ff3', '#54a0ff'][:len(data_points)],
                     alpha=0.8, edgecolor='black', linewidth=2)
        
        # Customize the sketchy chart
        ax.set_xlabel('Research Topics', fontsize=14, fontweight='bold')
        ax.set_ylabel('Data Points', fontsize=14, fontweight='bold')
        ax.set_title('Research Data Overview', fontsize=18, fontweight='bold', pad=20)
        
        # Set x-axis labels
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, data_points)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(data_points)*0.01,
                   f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Style the sketchy chart
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
        
        plt.tight_layout()
        
        # Add sketchy border
        fig.patch.set_edgecolor('black')
        fig.patch.set_linewidth(3)
        
        # Save to bytes
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', 
                   facecolor='#f8f9fa', edgecolor='black')
        img_buffer.seek(0)
        
        # Upload to GCS
        gcs = get_gcs_client()
        bucket = gcs.bucket(GCS_BUCKET_NAME)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"sketchy_charts/{timestamp}_{unique_id}.png"
        
        blob = bucket.blob(filename)
        blob.upload_from_string(img_buffer.getvalue(), content_type="image/png")
        blob.make_public()
        
        plt.close(fig)
        return blob.public_url
        
    except Exception as e:
        logger.error(f"Error creating data visualization: {e}")
        raise Exception(f"Failed to create data visualization: {str(e)}")

def create_numerical_bar_chart(analysis: dict, research_data: list) -> str:
    """Deprecated: Research-source visuals are disabled. Kept for compatibility but unused."""
    raise Exception("create_numerical_bar_chart is deprecated and disabled.")

def create_topic_distribution_chart(analysis: dict, research_data: list) -> str:
    """Create a sketchy pie chart showing topic distribution"""
    
    plt.xkcd()  # Enable sketchy style
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='#f8f9fa')
    
    # Chart 1: Topic Distribution Pie Chart
    topics = analysis['topics'][:6]  # Top 6 topics
    word_counts = analysis['word_counts'][:6]
    
    colors = ['#ff6b6b', '#4ecdc4', '#feca57', '#ff9ff3', '#54a0ff', '#5f27cd']
    
    wedges, texts, autotexts = ax1.pie(word_counts, labels=topics, autopct='%1.1f%%',
                                      colors=colors[:len(topics)], startangle=90,
                                      wedgeprops=dict(edgecolor='black', linewidth=2))
    
    ax1.set_title('Research Topic Distribution', fontsize=18, fontweight='bold', pad=20)
    
    # Chart 2: Source Type Analysis
    if analysis['source_types']:
        source_counts = {}
        for source_type in analysis['source_types']:
            source_counts[source_type] = source_counts.get(source_type, 0) + 1
        
        source_types = list(source_counts.keys())
        counts = list(source_counts.values())
        
        bars = ax2.bar(source_types, counts, 
                      color=['#45b7d1', '#96ceb4', '#feca57'][:len(source_types)],
                      alpha=0.8, edgecolor='black', linewidth=2)
        
        ax2.set_title('Research Sources by Type', fontsize=18, fontweight='bold', pad=20)
        ax2.set_ylabel('Number of Sources', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=1)
        
        # Add value labels
        for bar, value in zip(bars, counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save and upload
    return save_and_upload_chart(fig, "topic_distribution")

def create_content_length_chart(analysis: dict, research_data: list) -> str:
    """Create a sketchy line chart showing content length trends"""
    
    plt.xkcd()  # Enable sketchy style
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='#f8f9fa')
    
    # Create line chart for content lengths
    topics = analysis['topics'][:8]  # Top 8 topics
    word_counts = analysis['word_counts'][:8]
    
    # Create sketchy line plot
    ax.plot(range(len(topics)), word_counts, 
           marker='o', linewidth=4, markersize=10, 
           color='#ff6b6b', alpha=0.8, markeredgecolor='black', markeredgewidth=2)
    
    # Fill area under the line
    ax.fill_between(range(len(topics)), word_counts, alpha=0.3, color='#ff6b6b')
    
    ax.set_title('Research Content Length Analysis', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Research Topics', fontsize=14, fontweight='bold')
    ax.set_ylabel('Word Count', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    
    # Set x-axis labels
    ax.set_xticks(range(len(topics)))
    ax.set_xticklabels(topics, rotation=45, ha='right')
    
    # Add value labels on points
    for i, (x, y) in enumerate(zip(range(len(topics)), word_counts)):
        ax.annotate(f'{y}', (x, y), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    # Save and upload
    return save_and_upload_chart(fig, "content_length_analysis")

def create_research_overview_chart(analysis: dict, research_data: list) -> str:
    """Create a simple sketchy overview chart when data is limited"""
    
    plt.xkcd()  # Enable sketchy style
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='#f8f9fa')
    
    # Create a simple bar chart with available data
    topics = analysis['topics'][:5]
    content_lengths = analysis['content_lengths'][:5]
    
    bars = ax.bar(topics, content_lengths, 
                 color=['#ff6b6b', '#4ecdc4', '#feca57', '#ff9ff3', '#54a0ff'][:len(topics)],
                 alpha=0.8, edgecolor='black', linewidth=2)
    
    ax.set_title('Research Content Overview', fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('Content Length (Characters)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels
    for bar, value in zip(bars, content_lengths):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(content_lengths)*0.01,
                f'{value}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save and upload
    return save_and_upload_chart(fig, "research_overview")

def save_and_upload_chart(fig, chart_type: str) -> str:
    """Save sketchy chart to bytes and upload to GCS"""
    
    # Add sketchy border to figure
    fig.patch.set_edgecolor('black')
    fig.patch.set_linewidth(3)
    
    # Save to bytes
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', 
               facecolor='#f8f9fa', edgecolor='black')
    img_buffer.seek(0)
    
    # Upload to GCS
    gcs = get_gcs_client()
    bucket = gcs.bucket(GCS_BUCKET_NAME)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"sketchy_research_charts/{timestamp}_{chart_type}_{unique_id}.png"
    
    blob = bucket.blob(filename)
    blob.upload_from_string(img_buffer.getvalue(), content_type="image/png")
    blob.make_public()
    
    plt.close(fig)
    return blob.public_url


# --- Shared helpers for infographic text overlays ---
def extract_five_ws(text: str, content_topic: Optional[str] = None) -> dict:
    """Extract concise 5W's (What/Who/When/Where/Why) from paragraph text with light heuristics.

    Falls back to topic-derived defaults when text lacks clear signals.
    """
    text = (text or "").strip()
    import re
    # Split on sentence boundaries more robustly
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    ws = {
        'what': 'Key insights and findings',
        'who': 'Target audience and stakeholders',
        'when': 'Timeline and implementation',
        'where': 'Market and application areas',
        'why': 'Business impact and benefits'
    }

    def first_phrase_with_keywords(sentences_local, keywords, window_left=2, window_right=3, min_len=6, max_len=60):
        for sentence in sentences_local:
            lower = sentence.lower()
            for keyword in keywords:
                if keyword in lower:
                    words = sentence.split()
                    for i, word in enumerate(words):
                        if keyword in word.lower():
                            start = max(0, i - window_left)
                            end = min(len(words), i + window_right)
                            phrase = ' '.join(words[start:end]).strip()
                            if len(phrase) >= min_len and len(phrase) <= max_len:
                                return phrase
        return None

    what_kw = ['technology', 'solution', 'approach', 'method', 'system', 'platform', 'tool', 'strategy', 'innovation', 'transformation', 'framework', 'model']
    who_kw = ['companies', 'businesses', 'organizations', 'customers', 'users', 'clients', 'leaders', 'teams', 'professionals', 'enterprises']
    when_kw = ['today', 'now', 'future', 'next', 'immediate', 'phase', 'stage', 'roadmap', 'timeline', 'implementation', 'deployment', 'quarter', 'q1', 'q2', 'q3', 'q4']
    where_kw = ['market', 'industry', 'sector', 'application', 'environment', 'platform', 'ecosystem', 'space', 'domain', 'field', 'region']
    why_kw = ['benefit', 'impact', 'result', 'outcome', 'advantage', 'improvement', 'growth', 'efficiency', 'productivity', 'success', 'value', 'roi', 'trust']

    # PRIORITIZE metric-bearing sentences for WHAT/WHY
    metric_sentences = [s for s in sentences if re.search(r"\b(\d+%|\d+\.\d+%|\d+\b)", s)]
    if metric_sentences:
        metric_what = first_phrase_with_keywords(metric_sentences, what_kw)
        if metric_what:
            ws['what'] = (metric_what[:70] + ('...' if len(metric_what) > 70 else ''))
        metric_why = first_phrase_with_keywords(metric_sentences, why_kw, window_left=1, window_right=5, min_len=5, max_len=70)
        if metric_why:
            ws['why'] = (metric_why[:65] + ('...' if len(metric_why) > 65 else ''))

    what_phrase = first_phrase_with_keywords(sentences, what_kw)
    who_phrase = first_phrase_with_keywords(sentences, who_kw, window_left=1, window_right=2, min_len=5, max_len=50)
    # WHEN: detect explicit dates/quarters/years and time phrases
    when_phrase = None
    date_match = None
    for s in sentences:
        m = re.search(r"\b(Q[1-4]|[JFMASOND][a-z]+\s+\d{4}|\d{4}|by\s+\d{4}|next\s+(quarter|month|year)|in\s+\d{4})\b", s, flags=re.IGNORECASE)
        if m:
            date_match = s
            break
    if date_match:
        when_phrase = date_match
    else:
        when_phrase = first_phrase_with_keywords(sentences, when_kw, window_left=1, window_right=4, min_len=5, max_len=60)

    # WHERE: prefer "in <location/industry>" patterns
    where_phrase = None
    for s in sentences:
        m = re.search(r"\bin\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})", s)
        if m and len(m.group(1).split()) <= 4:
            where_phrase = f"in {m.group(1)}"
            break
    if not where_phrase:
        where_phrase = first_phrase_with_keywords(sentences, where_kw, window_left=1, window_right=3, min_len=5, max_len=60)
    why_phrase = first_phrase_with_keywords(sentences, why_kw, window_left=1, window_right=3, min_len=5, max_len=50)

    if what_phrase:
        ws['what'] = (what_phrase[:70] + ('...' if len(what_phrase) > 70 else ''))
    if who_phrase:
        # If no explicit who, try proper-noun spans (companies, products)
        if not who_phrase:
            for s in sentences:
                m = re.findall(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3})\b", s)
                if m:
                    who_phrase = m[0]
                    break
        if who_phrase:
            ws['who'] = (who_phrase[:55] + ('...' if len(who_phrase) > 55 else ''))
    if when_phrase:
        ws['when'] = (when_phrase[:55] + ('...' if len(when_phrase) > 55 else ''))
    if where_phrase:
        ws['where'] = (where_phrase[:55] + ('...' if len(where_phrase) > 55 else ''))
    if why_phrase:
        ws['why'] = (why_phrase[:65] + ('...' if len(why_phrase) > 65 else ''))

    return ws


def create_fivew_summary_card(content_text: str, title: str = "Key Takeaways") -> str:
    """Render a clean, professional 5W-only card when there isn't enough data for charts."""
    try:
        if not IMAGE_GENERATION_ENABLED:
            raise Exception("matplotlib not available")

        import textwrap as _tw
        ws = extract_five_ws(content_text or "")

        fig = plt.figure(figsize=(12, 8), facecolor='white')
        fig.suptitle(title, fontsize=22, fontweight='bold', color='#111827', y=0.95)

        ax = fig.add_axes([0.08, 0.12, 0.84, 0.76])
        ax.axis('off')

        fivew_box = (
            f"What: {_tw.fill(ws.get('what',''), width=80)}\n\n"
            f"Who:  {_tw.fill(ws.get('who',''), width=80)}\n\n"
            f"When: {_tw.fill(ws.get('when',''), width=80)}\n\n"
            f"Where: {_tw.fill(ws.get('where',''), width=80)}\n\n"
            f"Why:  {_tw.fill(ws.get('why',''), width=80)}"
        )

        ax.text(0.0, 1.02, '5W Summary', fontsize=18, fontweight='bold', color='#111827', ha='left', va='bottom')
        ax.text(0.0, 0.98, fivew_box, fontsize=14, color='#111827', va='top', ha='left', linespacing=1.6,
                bbox=dict(boxstyle='round,pad=0.9', facecolor='#ffffff', edgecolor='#9ca3af', linewidth=2.0, alpha=0.99))

        # Border
        fig.patch.set_edgecolor('#FF6B35')
        fig.patch.set_linewidth(3)

        return save_and_upload_chart(fig, "fivew_only")
    except Exception as e:
        logger.error(f"Error creating 5W summary card: {e}")
        raise

def create_metrics_infographic(content_text: str, content_topic: str) -> Optional[str]:
    """Create a professional LinkedIn-style infographic with 5W's key points and metrics"""
    try:
        import re
        import numpy as np
        import textwrap
        
        # Find patterns like "90% ..." and grab a descriptive label after it
        matches = []
        for m in re.finditer(r"(\d{1,3})%\s+([^\.;\n]{0,140})", content_text or "", flags=re.IGNORECASE):
            pct = int(m.group(1))
            raw_label = (m.group(2) or "").strip()
            words = raw_label.split()
            if not words:
                continue
            # Keep up to 12 words for fuller labels
            if len(words) > 12:
                words = words[:12]
            label = " ".join(words)
            label_lower = label.lower()
            # Normalize common phrases to clean titles
            if "customer satisfaction" in label_lower:
                label = "Improved Customer Satisfaction"
            elif "increased revenue" in label_lower or "increase in revenue" in label_lower or "revenue increase" in label_lower:
                label = "Increased Revenue"
            elif "accuracy" in label_lower:
                label = "Prediction Accuracy"
            elif "latency" in label_lower:
                label = "Network Latency Reduction"
            elif "reliability" in label_lower:
                label = "Network Reliability"
            if len(label) < 3:
                continue
            matches.append((pct, label))

        # Deduplicate by label and keep top 3 highest percentages
        if not matches:
            return None
        unique = {}
        for pct, label in matches:
            key = label.lower()
            unique.setdefault(key, pct)
            unique[key] = max(unique[key], pct)
        metrics = sorted([(v, k) for k, v in unique.items()], reverse=True)[:3]
        if not metrics:
            return None

        # Extract 5W's from actual content
        def extract_5ws(text):
            text_lower = text.lower()
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            
            # Initialize with defaults
            ws = {
                'what': 'Key insights and findings',
                'who': 'Target audience and stakeholders', 
                'when': 'Timeline and implementation',
                'where': 'Market and application areas',
                'why': 'Business impact and benefits'
            }
            
            # Extract WHAT - look for main topics, technologies, solutions
            what_keywords = ['technology', 'solution', 'approach', 'method', 'system', 'platform', 'tool', 'strategy', 'innovation', 'transformation']
            what_phrases = []
            for sentence in sentences:
                for keyword in what_keywords:
                    if keyword in sentence.lower():
                        # Extract the phrase containing the keyword
                        words = sentence.split()
                        for i, word in enumerate(words):
                            if keyword in word.lower():
                                start = max(0, i-2)
                                end = min(len(words), i+3)
                                phrase = ' '.join(words[start:end])
                                if len(phrase) > 10 and len(phrase) < 60:
                                    what_phrases.append(phrase)
                                break
            if what_phrases:
                ws['what'] = what_phrases[0][:50] + ('...' if len(what_phrases[0]) > 50 else '')
            
            # Extract WHO - look for stakeholders, users, companies
            who_keywords = ['companies', 'businesses', 'organizations', 'customers', 'users', 'clients', 'leaders', 'teams', 'professionals', 'enterprises']
            who_phrases = []
            for sentence in sentences:
                for keyword in who_keywords:
                    if keyword in sentence.lower():
                        words = sentence.split()
                        for i, word in enumerate(words):
                            if keyword in word.lower():
                                start = max(0, i-1)
                                end = min(len(words), i+2)
                                phrase = ' '.join(words[start:end])
                                if len(phrase) > 5 and len(phrase) < 50:
                                    who_phrases.append(phrase)
                                break
            if who_phrases:
                ws['who'] = who_phrases[0][:45] + ('...' if len(who_phrases[0]) > 45 else '')
            
            # Extract WHEN - look for time references, implementation phases
            when_keywords = ['today', 'now', 'future', 'next', 'immediate', 'phase', 'stage', 'roadmap', 'timeline', 'implementation', 'deployment']
            when_phrases = []
            for sentence in sentences:
                for keyword in when_keywords:
                    if keyword in sentence.lower():
                        words = sentence.split()
                        for i, word in enumerate(words):
                            if keyword in word.lower():
                                start = max(0, i-1)
                                end = min(len(words), i+3)
                                phrase = ' '.join(words[start:end])
                                if len(phrase) > 5 and len(phrase) < 50:
                                    when_phrases.append(phrase)
                                break
            if when_phrases:
                ws['when'] = when_phrases[0][:45] + ('...' if len(when_phrases[0]) > 45 else '')
            
            # Extract WHERE - look for markets, industries, applications
            where_keywords = ['market', 'industry', 'sector', 'application', 'environment', 'platform', 'ecosystem', 'space', 'domain', 'field']
            where_phrases = []
            for sentence in sentences:
                for keyword in where_keywords:
                    if keyword in sentence.lower():
                        words = sentence.split()
                        for i, word in enumerate(words):
                            if keyword in word.lower():
                                start = max(0, i-1)
                                end = min(len(words), i+3)
                                phrase = ' '.join(words[start:end])
                                if len(phrase) > 5 and len(phrase) < 50:
                                    where_phrases.append(phrase)
                                break
            if where_phrases:
                ws['where'] = where_phrases[0][:45] + ('...' if len(where_phrases[0]) > 45 else '')
            
            # Extract WHY - look for benefits, impact, results, outcomes
            why_keywords = ['benefit', 'impact', 'result', 'outcome', 'advantage', 'improvement', 'growth', 'efficiency', 'productivity', 'success', 'value']
            why_phrases = []
            for sentence in sentences:
                for keyword in why_keywords:
                    if keyword in sentence.lower():
                        words = sentence.split()
                        for i, word in enumerate(words):
                            if keyword in word.lower():
                                start = max(0, i-1)
                                end = min(len(words), i+3)
                                phrase = ' '.join(words[start:end])
                                if len(phrase) > 5 and len(phrase) < 50:
                                    why_phrases.append(phrase)
                                break
            if why_phrases:
                ws['why'] = why_phrases[0][:45] + ('...' if len(why_phrases[0]) > 45 else '')
            
            # If no specific content found, try to extract from the main topic
            if content_topic and content_topic != 'Content Analysis':
                topic_words = content_topic.split()
                if len(topic_words) >= 2:
                    ws['what'] = f"Focus on {content_topic.lower()}"
                    if 'ai' in content_topic.lower() or 'artificial' in content_topic.lower():
                        ws['where'] = "AI and technology sectors"
                    if 'business' in content_topic.lower():
                        ws['who'] = "Business organizations and leaders"
                    if 'data' in content_topic.lower():
                        ws['why'] = "Data-driven decision making"
                
            return ws

        # Create a clean, professional infographic with a non-overlapping grid layout
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=(18, 12), facecolor='white')
        fig.suptitle(content_topic.title() if content_topic else "Key Performance Metrics",
                     fontsize=24, fontweight='bold', color='#1f2937', y=0.98)
        gs = GridSpec(nrows=2, ncols=3, figure=fig,
                      width_ratios=[2.2, 1.8, 1.6], height_ratios=[1.0, 1.0])
        fig.subplots_adjust(top=0.90, bottom=0.08, left=0.06, right=0.96, hspace=0.35, wspace=0.30)

        # Axes allocation
        ax1 = fig.add_subplot(gs[0, 0:2])      # Top-left: horizontal bars (spans 2 columns)
        ax2 = fig.add_subplot(gs[0, 2])        # Top-right: pie
        ax3 = fig.add_subplot(gs[1, 0:2])      # Bottom-left: progress bars (spans 2 columns)
        ax4 = fig.add_subplot(gs[1, 2])        # Bottom-right: 5W card

        # Chart 1: Horizontal bar chart (no axis overlap)
        percentages = [pct for pct, _ in metrics]
        labels = [label.replace('_', ' ').title() for _, label in metrics]
        colors = ['#2563eb', '#059669', '#d97706']
        
        y_pos = np.arange(len(labels))
        bars = ax1.barh(y_pos, percentages, color=colors[:len(metrics)], alpha=0.9, height=0.6)
        ax1.set_yticks(y_pos)
        # Wrap y-axis labels to avoid cutoff
        wrapped_labels = ["\n".join(textwrap.wrap(lbl, width=26)) for lbl in labels]
        ax1.set_yticklabels(wrapped_labels, fontsize=12, fontweight='bold')
        ax1.set_xlim(0, 110)  # Extra space for percentage labels
        ax1.set_xlabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Performance Metrics', fontsize=16, fontweight='bold', pad=20)
        
        for i, (bar, pct) in enumerate(zip(bars, percentages)):
            ax1.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                     f'{pct}%', va='center', fontsize=14, fontweight='bold', color=colors[i])

        # Chart 2: Pie chart (right column)
        total_pct = sum(percentages)
        pie_data = percentages[:]
        pie_labels = labels[:]
        if len(metrics) >= 2 and 0 < total_pct < 100:
            pie_data.append(100 - total_pct)
            pie_labels.append('Other')
        # Keep colors aligned
        pie_colors = colors[:len(pie_data)] + (['#e5e7eb'] if len(pie_data) > len(colors) else [])
        ax2.set_title('Distribution Overview', fontsize=16, fontweight='bold', pad=12)
        if len(pie_data) == len(pie_labels) and len(pie_data) > 0:
            wedges, texts, autotexts = ax2.pie(
                pie_data, labels=pie_labels, colors=pie_colors,
                autopct='%1.0f%%', startangle=90, textprops={'fontsize': 9}
            )
            for t in autotexts:
                t.set_fontweight('bold')
        else:
            ax2.text(0.5, 0.5, 'Single Metric\nFocus', ha='center', va='center',
                     fontsize=16, fontweight='bold', color='#6b7280')

        # Chart 3: Progress bars (bottom-left block)
        ax3.set_title('Progress Indicators', fontsize=16, fontweight='bold', pad=12)
        
        y_pos = np.arange(len(metrics))
        for i, (pct, label_key) in enumerate(metrics):
            # Background bar
            ax3.barh(i, 100, height=0.6, color='#e5e7eb', alpha=0.3)
            # Progress bar
            ax3.barh(i, pct, height=0.6, color=colors[i], alpha=0.9)
            # Label and percentage
            display_label = label_key.title()
            wrapped = "\n".join(textwrap.wrap(display_label, width=32))
            ax3.text(-12, i, wrapped,
                     va='center', ha='right', fontsize=11, fontweight='bold', linespacing=1.2)
            ax3.text(pct + 5, i, f'{pct}%', va='center', fontsize=12, fontweight='bold', color=colors[i])
        
        ax3.set_ylim(-0.5, len(metrics) - 0.5)
        # More left space for wrapped multi-line labels
        ax3.set_xlim(-70, 115)
        ax3.set_yticks([])

        # Chart 4 area is repurposed as the 5W card container; create a lightweight header above it
        ax4.axis('off')
        
        avg_pct = float(np.mean(percentages)) if percentages else 0.0
        max_pct = max(percentages) if percentages else 0
        min_pct = min(percentages) if percentages else 0
        
        stats_text = (
            f"Average Performance: {avg_pct:.1f}%\n"
            f"Highest Achievement: {max_pct}%\n"
            f"Improvement Range: {min_pct}% - {max_pct}%\n"
            f"Total Metrics: {len(metrics)}"
        )
        # 5W's overlay derived from the actual paragraph content (stacked card inside ax4)
        ws = extract_five_ws(content_text or "", content_topic)
        import textwrap as _tw
        wrap_left = 40
        wrap_right = 40
        fives_text_left = (
            f"What: {_tw.fill(ws.get('what', ''), width=wrap_left)}\n"
            f"Who:  {_tw.fill(ws.get('who', ''), width=wrap_left)}\n"
            f"When: {_tw.fill(ws.get('when', ''), width=wrap_left)}"
        )
        fives_text_right = (
            f"Where: {_tw.fill(ws.get('where', ''), width=wrap_right)}\n"
            f"Why:   {_tw.fill(ws.get('why', ''), width=wrap_right)}"
        )

        # Build stacked card within ax4 region only
        ax4.text(0.02, 0.98, 'Key Insights', fontsize=14, fontweight='bold', color='#111827', ha='left', va='top')
        ax4.text(0.02, 0.88, stats_text, fontsize=12, fontweight='bold',
                 color='#374151', va='top', ha='left',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='#f8fafc', 
                          edgecolor='#e2e8f0', linewidth=1.2, alpha=0.95))

        stacked = fives_text_left + "\n" + fives_text_right
        ax4.text(0.02, 0.44, '5W Summary', fontsize=16, fontweight='bold', color='#111827', ha='left', va='top')
        ax4.text(0.02, 0.40, stacked, fontsize=14, color='#111827', va='top', ha='left', linespacing=1.55,
                 bbox=dict(boxstyle='round,pad=0.9', facecolor='#ffffff', edgecolor='#9ca3af', linewidth=2.0, alpha=0.99))

        plt.tight_layout()

        return save_and_upload_chart(fig, "professional_linkedin_infographic")

    except Exception as e:
        logger.error(f"Professional LinkedIn infographic creation failed: {e}")
        return None

async def create_stability_ai_visual(content_topic: str, research_data: list) -> tuple:
    """
    Create engaging LinkedIn-style visuals using Stability AI based on research content
    """
    try:
        logger.info(f"Creating Stability AI visual for topic: {content_topic}")
        
        # Extract key insights from research data
        key_insights = []
        for research in research_data[:3]:  # Use top 3 research items
            findings = research.get('findings', '')
            if findings:
                # Extract key sentences (look for sentences with numbers, percentages, or key terms)
                sentences = findings.split('. ')
                for sentence in sentences[:3]:  # Take first 3 sentences
                    if any(keyword in sentence.lower() for keyword in ['%', 'percent', 'increase', 'decrease', 'growth', 'trend', 'data', 'research', 'study', 'analysis']):
                        key_insights.append(sentence.strip())
        
        # Create a compelling prompt for Stability AI
        if key_insights:
            insight_text = ' '.join(key_insights[:2])  # Use top 2 insights
            prompt = f"Professional LinkedIn post visual: {content_topic}. Key insight: {insight_text[:200]}. Style: Clean, modern business infographic with data visualization elements, professional color scheme, engaging layout suitable for LinkedIn. No text overlays, just visual elements."
        else:
            prompt = f"Professional LinkedIn post visual about {content_topic}. Style: Clean, modern business infographic, professional color scheme, engaging layout suitable for LinkedIn. Focus on the main concept visually."
        
        # Call Stability AI API
        import requests
        
        headers = {
            "Authorization": f"Bearer {STABILITY_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "text_prompts": [
                {
                    "text": prompt,
                    "weight": 1.0
                }
            ],
            "cfg_scale": 7,
            "height": 1024,
            "width": 1024,
            "samples": 1,
            "steps": 30,
            "style_preset": "photographic"
        }
        
        response = requests.post(
            "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
            headers=headers,
            json=data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Get the generated image
            if result.get('artifacts') and len(result['artifacts']) > 0:
                image_data = result['artifacts'][0]['base64']
                
                # Upload to GCS
                gcs = get_gcs_client()
                bucket = gcs.bucket(GCS_BUCKET_NAME)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = str(uuid.uuid4())[:8]
                filename = f"stability_images/linkedin_visual_{timestamp}_{unique_id}.png"
                
                blob = bucket.blob(filename)
                blob.upload_from_string(base64.b64decode(image_data), content_type="image/png")
                blob.make_public()
                
                logger.info(f"Successfully created Stability AI visual: {blob.public_url}")
                return blob.public_url, "Generated with Stability AI"
        
        logger.warning(f"Stability AI request failed: {response.status_code} - {response.text}")
        return None, None
        
    except Exception as e:
        logger.error(f"Error creating Stability AI visual: {e}")
        return None, None

async def convert_image_to_sketchy_theme(image_url: str, content_topic: str) -> tuple:
    """
    Convert an existing image to hand-drawn, sketchy, artistic theme using Stability AI
    """
    try:
        logger.info(f"Converting image to hand-drawn artistic theme: {image_url}")
        
        # Create a detailed prompt for hand-drawn artistic conversion matching your reference examples
        prompt = f"""Convert this image to a hand-drawn, sketchy, artistic style exactly like the reference examples. 
        Make it look like it was drawn by hand with colored pencils, markers, or crayons. 
        Features to include:
        - Thick, wobbly, hand-drawn borders and lines
        - Vibrant, artistic colors (coral red, teal, yellow, pink, blue)
        - Hand-drawn arrows and annotations
        - Sketchy, doodle-like appearance
        - Artistic exclamation marks and symbols
        - Hand-drawn typography and labels
        - Keep the same data/content but make it look artistic and hand-drawn
        - Style: hand-drawn infographic, sketchy, colorful, artistic, doodle-like, creative illustration
        Topic: {content_topic}"""
        
        # Download the original image
        import requests
        response = requests.get(image_url, timeout=30)
        if response.status_code != 200:
            raise Exception(f"Failed to download image: {response.status_code}")
        
        image_data = response.content
        
        # Convert to base64
        import base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Call Stability AI API for image-to-image conversion
        headers = {
            "Authorization": f"Bearer {STABILITY_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "text_prompts": [
                {
                    "text": prompt,
                    "weight": 1.0
                }
            ],
            "init_image": image_base64,
            "init_image_mode": "IMAGE_STRENGTH",
            "image_strength": 0.35,  # Keep some of the original structure
            "cfg_scale": 7,
            "samples": 1,
            "steps": 30,
            "style_preset": "comic-book"
        }
        
        response = requests.post(
            "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/image-to-image",
            headers=headers,
            json=data,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Get the converted image
            if result.get('artifacts') and len(result['artifacts']) > 0:
                converted_image_data = result['artifacts'][0]['base64']
                
                # Upload to GCS
                gcs = get_gcs_client()
                bucket = gcs.bucket(GCS_BUCKET_NAME)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = str(uuid.uuid4())[:8]
                filename = f"sketchy_converted_images/converted_{timestamp}_{unique_id}.png"
                
                blob = bucket.blob(filename)
                blob.upload_from_string(base64.b64decode(converted_image_data), content_type="image/png")
                blob.make_public()
                
                logger.info(f"Successfully converted image to sketchy theme: {blob.public_url}")
                return blob.public_url, "Converted to sketchy theme with Stability AI"
        
        logger.warning(f"Stability AI conversion failed: {response.status_code} - {response.text}")
        return None, None
        
    except Exception as e:
        logger.error(f"Error converting image to sketchy theme: {e}")
        return None, None

async def search_web_for_images(content_topic: str) -> tuple:
    """
    Find relevant images based on content topics - news articles, topic-related images
    """
    try:
        logger.info(f"Searching for relevant images for topic: {content_topic}")
        
        # Extract key terms for better image matching
        topic_lower = content_topic.lower()
        
        # Enhanced topic-to-image mapping with more creative and specific images
        topic_image_map = {
            # AI & Machine Learning - More specific and creative
            'deep learning': "https://images.unsplash.com/photo-1555949963-aa79dcee981c?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            'neural network': "https://images.unsplash.com/photo-1555949963-aa79dcee981c?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            'machine learning': "https://images.unsplash.com/photo-1555949963-aa79dcee981c?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            'ai': "https://images.unsplash.com/photo-1677442136019-21780ecad995?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            'artificial intelligence': "https://images.unsplash.com/photo-1677442136019-21780ecad995?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            'prediction': "https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            'accuracy': "https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            
            # Business Performance & Results
            'customer satisfaction': "https://images.unsplash.com/photo-1556742049-0cfed4f6a45d?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            'revenue': "https://images.unsplash.com/photo-1559526324-4b87b5e36e44?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            'profit': "https://images.unsplash.com/photo-1559526324-4b87b5e36e44?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            'performance': "https://images.unsplash.com/photo-1460925895917-afdab827c52f?ixlib=rb-4.0.3&auto=format&fit=crop&w=2015&q=80",
            'success': "https://images.unsplash.com/photo-1460925895917-afdab827c52f?ixlib=rb-4.0.3&auto=format&fit=crop&w=2015&q=80",
            'growth': "https://images.unsplash.com/photo-1460925895917-afdab827c52f?ixlib=rb-4.0.3&auto=format&fit=crop&w=2015&q=80",
            'improvement': "https://images.unsplash.com/photo-1460925895917-afdab827c52f?ixlib=rb-4.0.3&auto=format&fit=crop&w=2015&q=80",
            
            # Network & Operations
            'network traffic': "https://images.unsplash.com/photo-1558494949-ef010cbdcc31?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            'network': "https://images.unsplash.com/photo-1558494949-ef010cbdcc31?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            'traffic': "https://images.unsplash.com/photo-1558494949-ef010cbdcc31?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            'operations': "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            'efficiency': "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            'productivity': "https://images.unsplash.com/photo-1460925895917-afdab827c52f?ixlib=rb-4.0.3&auto=format&fit=crop&w=2015&q=80",
            
            # Data & Analytics
            'data': "https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            'analytics': "https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            'big data': "https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            'data science': "https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            'research': "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            
            # Technology & Innovation
            'technology': "https://images.unsplash.com/photo-1518709268805-4e9042af2176?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            'innovation': "https://images.unsplash.com/photo-1518709268805-4e9042af2176?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            'digital': "https://images.unsplash.com/photo-1518709268805-4e9042af2176?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            'transformation': "https://images.unsplash.com/photo-1518709268805-4e9042af2176?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            'automation': "https://images.unsplash.com/photo-1518709268805-4e9042af2176?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            
            # Business & Finance
            'business': "https://images.unsplash.com/photo-1551434678-e076c223a692?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            'venture capital': "https://images.unsplash.com/photo-1554224155-6726b3ff858f?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            'investment': "https://images.unsplash.com/photo-1554224155-6726b3ff858f?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            'startup': "https://images.unsplash.com/photo-1559136555-9303baea8ebd?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            'finance': "https://images.unsplash.com/photo-1554224155-6726b3ff858f?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            'funding': "https://images.unsplash.com/photo-1554224155-6726b3ff858f?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            
            # Leadership & Management
            'leadership': "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            'management': "https://images.unsplash.com/photo-1560472354-b33ff0c44a43?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            'team': "https://images.unsplash.com/photo-1522071820081-009f0129c71c?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            'collaboration': "https://images.unsplash.com/photo-1522071820081-009f0129c71c?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            
            # Marketing & Content
            'marketing': "https://images.unsplash.com/photo-1611224923853-80b023f02d71?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            'social media': "https://images.unsplash.com/photo-1611224923853-80b023f02d71?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            'content': "https://images.unsplash.com/photo-1552664730-d307ca884978?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            'strategy': "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            
            # Future & Trends
            'future': "https://images.unsplash.com/photo-1518709268805-4e9042af2176?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80",
            'trends': "https://images.unsplash.com/photo-1518709268805-4e9042af2176?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80"
        }
        
        # Find the best matching topic
        best_match = None
        best_score = 0
        
        for topic, image_url in topic_image_map.items():
            if topic in topic_lower:
                score = len(topic.split())  # Prefer longer, more specific matches
                if score > best_score:
                    best_score = score
                    best_match = (image_url, f"https://unsplash.com/search/photos/{topic.replace(' ', '-')}")
        
        if best_match:
            logger.info(f"Found relevant image for topic: {content_topic}")
            return best_match
        
        # Fallback to general business image
        fallback_image = "https://images.unsplash.com/photo-1551434678-e076c223a692?ixlib=rb-4.0.3&auto=format&fit=crop&w=2015&q=80"
        fallback_source = f"https://unsplash.com/search/photos/{content_topic.replace(' ', '-')}"
        
        logger.info(f"Using fallback image for topic: {content_topic}")
        return fallback_image, fallback_source
        
    except Exception as e:
        logger.error(f"Image search failed: {e}")
        return None, None

async def generate_contextual_image(content_data: dict, research_data: list) -> tuple:
    """
    Enhanced contextual image generation that combines visual content with data insights
    Returns (image_url, source_link)
    """
    try:
        # Extract key topics from LinkedIn posts
        content_topics = []
        all_content_text = ""
        
        # Build a posts-only view of content for deterministic visuals
        posts_only = {
            'linkedin_posts': content_data.get('linkedin_posts', []) or [],
            'video_scripts': [],
            'hashtags': [],
            'engagement_tips': []
        }
        
        # Concatenate only LinkedIn posts text
        if posts_only['linkedin_posts']:
            for post in posts_only['linkedin_posts']:
                all_content_text += post + " "
        
        # Extract meaningful words from all content
        if all_content_text:
            words = all_content_text.lower().split()
            
            # Enhanced filtering for meaningful words
            stop_words = {
                'this', 'that', 'with', 'from', 'they', 'have', 'will', 'been', 'were', 'said', 
                'each', 'which', 'their', 'time', 'would', 'there', 'could', 'other', 'after', 
                'first', 'well', 'also', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 
                'most', 'us', 'is', 'are', 'was', 'be', 'to', 'of', 'and', 'a', 'in', 'for', 'on', 
                'with', 'as', 'by', 'an', 'the', 'or', 'at', 'it', 'you', 'he', 'she', 'we', 'they', 
                'me', 'him', 'her', 'us', 'them', 'your', 'our', 'my', 'his', 'her', 'its', 'our', 
                'their', 'what', 'when', 'where', 'why', 'how', 'who', 'whom', 'whose', 'can', 
                'could', 'should', 'would', 'may', 'might', 'must', 'shall', 'will', 'do', 'does', 
                'did', 'done', 'doing', 'get', 'got', 'getting', 'make', 'made', 'making', 'take', 
                'took', 'taking', 'come', 'came', 'coming', 'go', 'went', 'going', 'see', 'saw', 
                'seeing', 'know', 'knew', 'knowing', 'think', 'thought', 'thinking', 'look', 'looked', 
                'looking', 'use', 'used', 'using', 'find', 'found', 'finding', 'work', 'worked', 
                'working', 'help', 'helped', 'helping', 'feel', 'felt', 'feeling', 'seem', 'seemed', 
                'seeming', 'try', 'tried', 'trying', 'leave', 'left', 'leaving', 'call', 'called', 
                'calling', 'ask', 'asked', 'asking', 'need', 'needed', 'needing', 'become', 'became', 
                'becoming', 'move', 'moved', 'moving', 'live', 'lived', 'living', 'believe', 'believed', 
                'believing', 'hold', 'held', 'holding', 'bring', 'brought', 'bringing', 'happen', 
                'happened', 'happening', 'write', 'wrote', 'writing', 'provide', 'provided', 'providing', 
                'sit', 'sat', 'sitting', 'stand', 'stood', 'standing', 'lose', 'lost', 'losing', 
                'pay', 'paid', 'paying', 'meet', 'met', 'meeting', 'include', 'included', 'including', 
                'continue', 'continued', 'continuing', 'set', 'setting', 'learn', 'learned', 'learning', 
                'change', 'changed', 'changing', 'lead', 'led', 'leading', 'understand', 'understood', 
                'understanding', 'watch', 'watched', 'watching', 'follow', 'followed', 'following', 
                'stop', 'stopped', 'stopping', 'create', 'created', 'creating', 'speak', 'spoke', 
                'speaking', 'read', 'reading', 'allow', 'allowed', 'allowing', 'add', 'added', 'adding', 
                'spend', 'spent', 'spending', 'grow', 'grew', 'growing', 'open', 'opened', 'opening', 
                'walk', 'walked', 'walking', 'win', 'won', 'winning', 'offer', 'offered', 'offering', 
                'remember', 'remembered', 'remembering', 'love', 'loved', 'loving', 'consider', 
                'considered', 'considering', 'appear', 'appeared', 'appearing', 'buy', 'bought', 'buying', 
                'wait', 'waited', 'waiting', 'serve', 'served', 'serving', 'die', 'died', 'dying', 
                'send', 'sent', 'sending', 'expect', 'expected', 'expecting', 'build', 'built', 'building', 
                'stay', 'stayed', 'staying', 'fall', 'fell', 'falling', 'cut', 'cutting', 'reach', 
                'reached', 'reaching', 'kill', 'killed', 'killing', 'remain', 'remained', 'remaining', 
                'suggest', 'suggested', 'suggesting', 'raise', 'raised', 'raising', 'pass', 'passed', 
                'passing', 'sell', 'sold', 'selling', 'require', 'required', 'requiring', 'report', 
                'reported', 'reporting', 'decide', 'decided', 'deciding', 'pull', 'pulled', 'pulling'
            }
            
            meaningful_words = [word for word in words if len(word) > 3 and word not in stop_words]
            content_topics.extend(meaningful_words)
        
        # Create a topic string from the most common words (limit length to avoid URL issues)
        if content_topics:
            from collections import Counter
            topic_counts = Counter(content_topics)
            
            # Prioritize business-relevant terms
            business_terms = ['business', 'ai', 'technology', 'data', 'marketing', 'leadership', 
                            'innovation', 'digital', 'strategy', 'growth', 'investment', 'startup',
                            'venture', 'capital', 'machine', 'learning', 'analytics', 'transformation']
            
            # Get business terms first, then other common terms
            prioritized_topics = []
            for term in business_terms:
                if term in [t.lower() for t in content_topics]:
                    prioritized_topics.append(term)
            
            # Add other common terms if we don't have enough
            other_topics = [topic for topic, count in topic_counts.most_common(5) 
                          if topic.lower() not in business_terms]
            prioritized_topics.extend(other_topics[:3-len(prioritized_topics)])
            
            content_topic = " ".join(prioritized_topics[:3])  # Max 3 topics
            
            # Limit topic length to prevent URL issues
            if len(content_topic) > 50:
                content_topic = content_topic[:50]
            
            logger.info(f"Generated content topics: {content_topic}")
            
            # Build research context text to guide topic detection
            research_context_text = " ".join([
                (r.get('topic') or '') + ' ' + (r.get('findings') or '') + ' ' + (r.get('data') or '')
                for r in (research_data or [])
            ])

            # Strategy 1: If post contains clear percentages, render a metrics infographic
            primary_post = (posts_only['linkedin_posts'][0] if posts_only['linkedin_posts'] else "")
            metrics_url = create_metrics_infographic(primary_post, content_topic)
            if metrics_url:
                logger.info("Created metrics-driven infographic from post claims")
                return metrics_url, "Infographic from post claims"

            # Strategy 2: Create analytical infographic from post content only (no research sources)
            analytical_url = create_content_analytical_visualization(posts_only, content_topic)
            if analytical_url:
                logger.info("Created analytical infographic from post content")
                return analytical_url, "Analytical visualization from post content"

            # No research source infographics - only post content infographics
            logger.info("No post content infographic could be generated - returning None")
            return None, None
        
        # No fallback to random images - only infographics
        logger.info("No infographic could be generated - returning None")
        return None, None
        
    except Exception as e:
        logger.error(f"Contextual image generation failed: {e}")
        return None, None

async def create_hybrid_visual_content(content_data: dict, research_data: list, content_topic: str) -> tuple:
    """Keep only professional visuals for hybrid requests (no playful styles)."""
    logger.info("Hybrid visuals constrained to professional style")
    # Reuse professional content analytical visualization from the content only
    url = create_content_analytical_visualization(
        {
            'linkedin_posts': content_data.get('linkedin_posts', []),
            'video_scripts': [], 'hashtags': [], 'engagement_tips': []
        },
        content_topic or ""
    )
    return url, "Professional visualization"

def create_sketchy_analytical_visualization(content_data: dict, content_text: str) -> str:
    """
    Create truly hand-drawn, sketchy, artistic visualization matching the reference examples
    """
    try:
        if not IMAGE_GENERATION_ENABLED:
            logger.error("Image generation libraries not available")
            raise Exception("matplotlib, seaborn, and PIL are not installed.")
        
        logger.info("Creating hand-drawn artistic visualization")
        
        # Force xkcd style for hand-drawn look
        with plt.xkcd():
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), facecolor='#f8f9fa')
        
        # Hand-drawn, artistic color palette matching your reference examples
        colors = {
            'primary': '#FF6B6B',      # Coral red (hand-drawn feel)
            'secondary': '#4ECDC4',    # Teal (artistic)
            'accent': '#45B7D1',       # Sky blue (vibrant)
            'success': '#96CEB4',      # Mint green (soft)
            'warning': '#FECA57',      # Yellow (bright)
            'purple': '#FF9FF3',       # Pink (artistic)
            'green': '#54A0FF',        # Light blue (vibrant)
            'red': '#FF6B6B',          # Coral red
            'orange': '#FF9F43',       # Orange (warm)
            'light_gray': '#F8F9FA',   # Very light gray
            'medium_gray': '#E9ECEF',  # Medium gray
            'dark': '#2D3436',         # Dark gray
            'text': '#2D3436'          # Text color
        }
        
        # Chart 1: Content Type Distribution
        content_types = ['LinkedIn Posts', 'Video Scripts', 'Hashtags', 'Engagement Tips']
        content_counts = [
            len(content_data.get('linkedin_posts', [])),
            len(content_data.get('video_scripts', [])),
            len(content_data.get('hashtags', [])),
            len(content_data.get('engagement_tips', []))
        ]
        
        # Create hand-drawn style bars with thick, wobbly borders
        bars1 = ax1.bar(content_types, content_counts, 
                       color=[colors['primary'], colors['secondary'], colors['warning'], colors['purple']],
                       alpha=0.9, edgecolor='black', linewidth=4)
        
        # Hand-drawn style title with artistic feel
        ax1.set_title('CONTENT BREAKDOWN', fontsize=20, fontweight='bold', pad=30, color=colors['dark'])
        ax1.set_ylabel('Count', fontsize=16, fontweight='bold', color=colors['text'])
        
        # Remove grid for cleaner hand-drawn look
        ax1.grid(False)
        ax1.set_facecolor('#F8F9FA')
        
        # Add hand-drawn style value labels
        for bar, value in zip(bars1, content_counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value}', ha='center', va='bottom', fontweight='bold', 
                    fontsize=16, color=colors['dark'])
        
        # Chart 2: Content Length Analysis
        post_lengths = []
        if content_data.get('linkedin_posts'):
            for post in content_data['linkedin_posts']:
                post_lengths.append(len(post.split()))
        
        if post_lengths:
            # Hand-drawn style histogram
            ax2.hist(post_lengths, bins=min(10, len(post_lengths)), 
                    color=colors['accent'], alpha=0.9, edgecolor='black', linewidth=4)
            ax2.set_title('POST LENGTH TRENDS', fontsize=20, fontweight='bold', pad=30, color=colors['dark'])
            ax2.set_xlabel('Word Count', fontsize=16, fontweight='bold', color=colors['text'])
            ax2.set_ylabel('Frequency', fontsize=16, fontweight='bold', color=colors['text'])
            ax2.grid(False)  # Remove grid for hand-drawn look
            ax2.set_facecolor('#F8F9FA')
        else:
            ax2.text(0.5, 0.5, 'No posts to analyze', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('LinkedIn Post Length Distribution', fontsize=14, fontweight='bold', pad=20)
        
        # Chart 3: Keyword Frequency (Top 10)
        words = content_text.lower().split()
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'a', 'an', 'as', 'if', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'}
        meaningful_words = [word for word in words if len(word) > 3 and word not in stop_words]
        
        if meaningful_words:
            from collections import Counter
            word_counts = Counter(meaningful_words)
            top_words = word_counts.most_common(10)
            
            if top_words:
                words_list, counts_list = zip(*top_words)
                bars3 = ax3.barh(range(len(words_list)), counts_list, 
                                color=colors['accent'], alpha=0.9, edgecolor='white', linewidth=3)
                ax3.set_yticks(range(len(words_list)))
                ax3.set_yticklabels(words_list, fontsize=12, color=colors['text'])
                ax3.set_title('TOP KEYWORDS ANALYSIS', fontsize=18, fontweight='bold', pad=25, color=colors['dark'])
                ax3.set_xlabel('Frequency', fontsize=14, fontweight='bold', color=colors['text'])
                ax3.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
                ax3.set_facecolor('#FAFAFA')
                
                # Add value labels
                for i, (bar, value) in enumerate(zip(bars3, counts_list)):
                    ax3.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                            f'{value}', ha='left', va='center', fontweight='bold')
            else:
                ax3.text(0.5, 0.5, 'No meaningful keywords found', ha='center', va='center', 
                        transform=ax3.transAxes, fontsize=12)
                ax3.set_title('Top Keywords in Generated Content', fontsize=14, fontweight='bold', pad=20)
        else:
            ax3.text(0.5, 0.5, 'No meaningful keywords found', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Top Keywords in Generated Content', fontsize=14, fontweight='bold', pad=20)
        
        # Chart 4: Content Quality Metrics
        metrics = ['Posts Generated', 'Scripts Generated', 'Hashtags Provided', 'Tips Generated']
        values = [
            len(content_data.get('linkedin_posts', [])),
            len(content_data.get('video_scripts', [])),
            len(content_data.get('hashtags', [])),
            len(content_data.get('engagement_tips', []))
        ]
        
        # Create a radar-like visualization
        angles = [n / float(len(metrics)) * 2 * 3.14159 for n in range(len(metrics))]
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        ax4 = plt.subplot(2, 2, 4, projection='polar')
        ax4.plot(angles, values, 'o-', linewidth=4, color=colors['red'], markersize=10)
        ax4.fill(angles, values, alpha=0.4, color=colors['red'])
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(metrics, fontsize=12, fontweight='bold', color=colors['text'])
        ax4.set_title('CONTENT METRICS RADAR', fontsize=18, fontweight='bold', pad=25, color=colors['dark'])
        ax4.grid(True, linestyle='-', alpha=0.3)
        ax4.set_facecolor('#FAFAFA')
        
        # Add value labels
        for angle, value, metric in zip(angles[:-1], values[:-1], metrics):
            ax4.text(angle, value + 0.5, str(value), ha='center', va='center', fontweight='bold')
        
        # Add overall title to the figure
        fig.suptitle('LINKEDIN CONTENT ANALYTICS DASHBOARD', fontsize=24, fontweight='bold', 
                    color=colors['dark'], y=0.98)
        
        # 5W panel on the right side of the grid (within figure coordinates)
        ws = extract_five_ws(content_text or "")
        import textwrap as _tw
        fivew_box = (
            f"What: {_tw.fill(ws.get('what',''), width=36)}\n"
            f"Who:  {_tw.fill(ws.get('who',''), width=36)}\n"
            f"When: {_tw.fill(ws.get('when',''), width=36)}\n"
            f"Where: {_tw.fill(ws.get('where',''), width=36)}\n"
            f"Why:  {_tw.fill(ws.get('why',''), width=36)}"
        )

        # Use an invisible axes overlay for the 5W's to avoid disturbing chart layout
        overlay_ax = fig.add_axes([0.76, 0.08, 0.22, 0.36])  # [left, bottom, width, height]
        overlay_ax.axis('off')
        overlay_ax.text(0.0, 1.05, '5W Summary', fontsize=13, fontweight='bold', color=colors['dark'], ha='left', va='bottom')
        overlay_ax.text(0.0, 0.98, fivew_box, fontsize=12, color=colors['text'], va='top', ha='left', linespacing=1.3,
                        bbox=dict(boxstyle="round,pad=0.6", facecolor='#FFFFFF', edgecolor='#D1D5DB', linewidth=1.5, alpha=0.98))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Add a professional border around the entire figure
        fig.patch.set_edgecolor(colors['primary'])
        fig.patch.set_linewidth(4)
        
        # Save to bytes with high quality
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor=colors['primary'])
        img_buffer.seek(0)
        
        # Upload to GCS
        gcs = get_gcs_client()
        bucket = gcs.bucket(GCS_BUCKET_NAME)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"sketchy_content_analysis/{timestamp}_{unique_id}.png"
        
        blob = bucket.blob(filename)
        blob.upload_from_string(img_buffer.getvalue(), content_type="image/png")
        blob.make_public()
        
        plt.close(fig)
        return blob.public_url
        
    except Exception as e:
        logger.error(f"Error creating sketchy content analytical visualization: {e}")
        raise Exception(f"Failed to create sketchy content analytical visualization: {str(e)}")

def create_hand_drawn_artistic_visualization(content_data: dict, content_text: str) -> str:
    """
    Create a truly hand-drawn, artistic visualization using PIL to draw sketchy elements
    """
    try:
        if not IMAGE_GENERATION_ENABLED:
            raise Exception("PIL not available")
        
        logger.info("Creating hand-drawn artistic visualization with PIL")
        
        # Create a large canvas
        width, height = 1200, 800
        img = Image.new('RGB', (width, height), '#F8F9FA')
        draw = ImageDraw.Draw(img)
        
        # Try to load a hand-drawn style font, fallback to default
        try:
            # Use a bold font for hand-drawn feel
            title_font = ImageFont.truetype("/System/Library/Fonts/Arial Bold.ttf", 36)
            label_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
            small_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
        except:
            # Fallback fonts
            title_font = ImageFont.load_default()
            label_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Hand-drawn color palette
        colors = {
            'primary': '#FF6B6B',    # Coral red
            'secondary': '#4ECDC4',  # Teal
            'accent': '#45B7D1',     # Sky blue
            'warning': '#FECA57',    # Yellow
            'purple': '#FF9FF3',     # Pink
            'green': '#96CEB4',      # Mint green
            'dark': '#2D3436',       # Dark gray
            'orange': '#FF9F43'      # Orange
        }
        
        # Draw hand-drawn title with wobbly effect
        title = "LINKEDIN CONTENT ANALYTICS"
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (width - title_width) // 2
        
        # Draw title with slight offset for hand-drawn effect
        draw.text((title_x + 2, 30), title, fill=colors['dark'], font=title_font)
        draw.text((title_x, 28), title, fill=colors['primary'], font=title_font)
        
        # Get content data
        content_types = ['LinkedIn Posts', 'Video Scripts', 'Hashtags', 'Tips']
        content_counts = [
            len(content_data.get('linkedin_posts', [])),
            len(content_data.get('video_scripts', [])),
            len(content_data.get('hashtags', [])),
            len(content_data.get('engagement_tips', []))
        ]
        
        # Draw hand-drawn bar chart
        chart_x, chart_y = 100, 150
        chart_width, chart_height = 1000, 400
        bar_width = chart_width // len(content_types) - 20
        max_count = max(content_counts) if content_counts else 1
        
        # Draw chart background with hand-drawn border
        draw.rectangle([chart_x-10, chart_y-10, chart_x+chart_width+10, chart_y+chart_height+10], 
                      outline=colors['dark'], width=6)
        
        # Draw bars with hand-drawn style
        bar_colors = [colors['primary'], colors['secondary'], colors['warning'], colors['purple']]
        for i, (content_type, count, color) in enumerate(zip(content_types, content_counts, bar_colors)):
            bar_x = chart_x + i * (bar_width + 20) + 10
            bar_height = (count / max_count) * (chart_height - 100)
            bar_y = chart_y + chart_height - bar_height - 50
            
            # Draw bar with thick, wobbly border
            draw.rectangle([bar_x, bar_y, bar_x + bar_width, bar_y + bar_height], 
                          fill=color, outline=colors['dark'], width=4)
            
            # Draw value label
            value_text = str(count)
            value_bbox = draw.textbbox((0, 0), value_text, font=label_font)
            value_width = value_bbox[2] - value_bbox[0]
            draw.text((bar_x + bar_width//2 - value_width//2, bar_y - 30), 
                     value_text, fill=colors['dark'], font=label_font)
            
            # Draw category label (rotated)
            label_bbox = draw.textbbox((0, 0), content_type, font=small_font)
            label_width = label_bbox[2] - label_bbox[0]
            draw.text((bar_x + bar_width//2 - label_width//2, chart_y + chart_height - 20), 
                     content_type, fill=colors['dark'], font=small_font)
        
        # Draw hand-drawn arrows and annotations
        # Arrow pointing to the chart
        arrow_start = (50, 200)
        arrow_end = (chart_x, chart_y + chart_height//2)
        draw.line([arrow_start, arrow_end], fill=colors['orange'], width=6)
        
        # Arrow head
        draw.polygon([arrow_end, (arrow_end[0]-15, arrow_end[1]-10), (arrow_end[0]-15, arrow_end[1]+10)], 
                    fill=colors['orange'])
        
        # Add hand-drawn exclamation marks
        draw.text((chart_x + chart_width + 20, chart_y + 50), "!", fill=colors['primary'], font=title_font)
        draw.text((chart_x + chart_width + 20, chart_y + 100), "!", fill=colors['secondary'], font=title_font)
        
        # Add some hand-drawn doodles
        # Draw a small star
        star_points = [
            (width - 100, 100), (width - 90, 80), (width - 80, 100),
            (width - 100, 90), (width - 80, 90)
        ]
        draw.polygon(star_points, fill=colors['warning'], outline=colors['dark'], width=3)
        
        # Draw a small heart
        heart_x, heart_y = width - 150, 200
        draw.ellipse([heart_x, heart_y, heart_x+20, heart_y+20], fill=colors['purple'], outline=colors['dark'], width=3)
        draw.ellipse([heart_x+15, heart_y, heart_x+35, heart_y+20], fill=colors['purple'], outline=colors['dark'], width=3)
        draw.polygon([heart_x+10, heart_y+20, heart_x+25, heart_y+35, heart_x+40, heart_y+20], 
                    fill=colors['purple'], outline=colors['dark'], width=3)
        
        # 5W panel (right column)
        ws = extract_five_ws(content_text or "")
        fivew_text = (
            f"What: {ws.get('what','')}\n"
            f"Who:  {ws.get('who','')}\n"
            f"When: {ws.get('when','')}\n"
            f"Where: {ws.get('where','')}\n"
            f"Why:  {ws.get('why','')}"
        )
        box_x1, box_y1 = width - 460, 240
        box_x2, box_y2 = width - 40, 560
        draw.rectangle([box_x1, box_y1, box_x2, box_y2], outline=colors['dark'], width=4, fill='#FFFFFF')
        header = "5W Summary"
        draw.text((box_x1 + 18, box_y1 + 16), header, fill=colors['dark'], font=label_font)
        # Wrap lines for better readability
        import textwrap as _tw
        wrapped_lines = []
        for line in fivew_text.split('\n'):
            wrapped_lines.extend(_tw.wrap(line, width=44) or [''])
        draw.multiline_text((box_x1 + 18, box_y1 + 54), '\n'.join(wrapped_lines), fill=colors['dark'], font=label_font, spacing=8)

        # Save to bytes
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG', quality=95)
        img_buffer.seek(0)
        
        # Upload to GCS
        gcs = get_gcs_client()
        bucket = gcs.bucket(GCS_BUCKET_NAME)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"hand_drawn_artistic/{timestamp}_{unique_id}.png"
        
        blob = bucket.blob(filename)
        blob.upload_from_string(img_buffer.getvalue(), content_type="image/png")
        blob.make_public()
        
        logger.info(f"Successfully created hand-drawn artistic visualization: {blob.public_url}")
        return blob.public_url
        
    except Exception as e:
        logger.error(f"Error creating hand-drawn artistic visualization: {e}")
        raise Exception(f"Failed to create hand-drawn artistic visualization: {str(e)}")

def create_simple_infographic_fallback(content_data: dict, content_text: str) -> str:
    """
    Create a simple infographic-style visualization as fallback
    """
    try:
        if not IMAGE_GENERATION_ENABLED:
            raise Exception("matplotlib not available")
        
        logger.info("Creating simple infographic fallback")
        
        # Create a simple, clean visualization
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
        
        # Professional colors
        colors = ['#FF6B35', '#004E89', '#1A659E', '#00A8CC']
        
        # Simple data from content
        content_types = ['LinkedIn Posts', 'Video Scripts', 'Hashtags', 'Tips']
        content_counts = [
            len(content_data.get('linkedin_posts', [])),
            len(content_data.get('video_scripts', [])),
            len(content_data.get('hashtags', [])),
            len(content_data.get('engagement_tips', []))
        ]
        
        # Create bar chart
        bars = ax.bar(content_types, content_counts, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        
        # Style the chart
        ax.set_title('CONTENT GENERATION SUMMARY', fontsize=20, fontweight='bold', pad=30, color='#2C3E50')
        ax.set_ylabel('Count', fontsize=14, fontweight='bold', color='#2C3E50')
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        ax.set_facecolor('#FAFAFA')
        
        # Add value labels
        for bar, value in zip(bars, content_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{value}', ha='center', va='bottom', fontweight='bold', fontsize=14, color='#2C3E50')
        
        # Add border
        fig.patch.set_edgecolor('#FF6B35')
        fig.patch.set_linewidth(3)
        
        # Prominent 5W panel (professional style) with fixed overlay region to avoid overlap
        ws = extract_five_ws(content_text or "")
        import textwrap as _tw
        fivew_box = (
            f"What: {_tw.fill(ws.get('what',''), width=36)}\n"
            f"Who:  {_tw.fill(ws.get('who',''), width=36)}\n"
            f"When: {_tw.fill(ws.get('when',''), width=36)}\n"
            f"Where: {_tw.fill(ws.get('where',''), width=36)}\n"
            f"Why:  {_tw.fill(ws.get('why',''), width=36)}"
        )
        overlay_ax = fig.add_axes([0.62, 0.12, 0.35, 0.36])  # right bottom quadrant
        overlay_ax.axis('off')
        overlay_ax.text(0.0, 1.06, '5W Summary', fontsize=15, fontweight='bold', color='#111827', ha='left', va='bottom')
        overlay_ax.text(0.0, 1.0, fivew_box, fontsize=13, color='#111827', va='top', ha='left', linespacing=1.35,
                        bbox=dict(boxstyle="round,pad=0.7", facecolor='#FFFFFF', edgecolor='#9ca3af', linewidth=2.0, alpha=0.99))

        plt.tight_layout()
        
        # Save and upload
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='#FF6B35')
        img_buffer.seek(0)
        
        # Upload to GCS
        gcs = get_gcs_client()
        bucket = gcs.bucket(GCS_BUCKET_NAME)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"simple_infographics/{timestamp}_{unique_id}.png"
        
        blob = bucket.blob(filename)
        blob.upload_from_string(img_buffer.getvalue(), content_type="image/png")
        blob.make_public()
        
        plt.close(fig)
        return blob.public_url
        
    except Exception as e:
        logger.error(f"Error creating simple infographic: {e}")
        raise Exception(f"Failed to create simple infographic: {str(e)}")

def create_content_analytical_visualization(content_data: dict, content_text: str) -> str:
    """
    Create professional, business-style visualization only (no playful/hand-drawn styles).

    Strategy:
    1) Try metrics-driven professional infographic if the paragraph contains percentages/claims.
    2) Else, create a clean, simple professional infographic.
    """
    try:
        # Prefer professional metrics infographic when claims exist
        metrics_url = create_metrics_infographic(content_text or "", content_text or "Content Analysis")
        if metrics_url:
            return metrics_url

        # If there isn't enough quantitative structure for charts, render a 5W-only card
        has_counts = any([
            len(content_data.get('linkedin_posts', [])),
            len(content_data.get('video_scripts', [])),
            len(content_data.get('hashtags', [])),
            len(content_data.get('engagement_tips', []))
        ])
        if not has_counts or (content_text and len((content_text or '').split()) < 25):
            return create_fivew_summary_card(content_text or "", title="Content Summary")

        # Otherwise, clean professional fallback
        return create_simple_infographic_fallback(content_data, content_text)
    except Exception as e:
        logger.warning(f"Professional visualization failed: {e}, using simple fallback")
        return create_simple_infographic_fallback(content_data, content_text)

def generate_creator_key(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]', '-', name.lower()).strip('-')

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "healthy", "database": "Supabase", "file_storage": "GCS"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/test-customization")
async def test_customization(request: ContentGenerationRequest):
    """Test endpoint to verify customization parameters are received correctly"""
    return {
        "message": "Customization parameters received successfully",
        "received_parameters": {
            "tone": request.tone,
            "content_format": request.content_format,
            "content_style": request.content_style,
            "include_statistics": request.include_statistics,
            "post_length": request.post_length,
            "call_to_action": request.call_to_action
        },
        "all_parameters": request.model_dump()
    }

# --- Document Management Endpoints ---
@app.post("/linkedin/upload-document", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    try:
        # For large files, we need to handle them in chunks to avoid memory issues
        # First, upload to GCS using streaming upload
        gcs = get_gcs_client()
        bucket = gcs.bucket(GCS_BUCKET_NAME)
        unique_filename = f"uploads/{uuid.uuid4()}-{filename}"
        blob = bucket.blob(unique_filename)
        
        # Stream the file content directly to GCS to avoid loading entire file into memory
        content_type = file.content_type or "application/octet-stream"
        
        # Reset file pointer to beginning
        await file.seek(0)
        
        # Upload file to GCS using streaming
        blob.upload_from_file(file.file, content_type=content_type)
        blob.make_public()
        public_gcs_url = blob.public_url
        
        # Now read the file content for text extraction
        # Reset file pointer again for text extraction
        await file.seek(0)
        content = await file.read()
        
        # Extract text and chunk it
        text = extract_text_from_file(content, filename)
        
        # For very large documents, use larger chunks to reduce database load
        chunk_size = 2000 if len(text) > 100000 else 1500  # Larger chunks for big docs
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=200)
        
        logger.info(f"Processing document '{filename}' with {len(chunks)} chunks (total text length: {len(text)})")
        
        # Insert chunks in batches to avoid overwhelming the database
        supabase = get_supabase_client()
        batch_size = 50  # Insert 50 chunks at a time
        total_items_added = 0
        
        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            
            items_to_insert = []
            for i, chunk in enumerate(batch_chunks):
                chunk_index = batch_start + i + 1
                items_to_insert.append({
                    "topic": f"From Doc: {filename} (Part {chunk_index})", 
                    "findings": chunk,
                    "source": public_gcs_url, 
                    "data": f"Chunk {chunk_index}/{len(chunks)} from: {filename}",
                    "tags": "document,upload,ingested",
                })
            
            logger.info(f"Inserting batch {batch_start//batch_size + 1}: chunks {batch_start + 1}-{batch_end}")
            response = supabase.table("research").insert(items_to_insert).execute()
            
            if response.data:
                total_items_added += len(response.data)
            else:
                logger.warning(f"Batch {batch_start//batch_size + 1} failed to insert")
        
        logger.info(f"Total items added: {total_items_added}")
        
        if total_items_added == 0:
            error_detail = f"Failed to insert any data into Supabase for document '{filename}'"
            logger.error(error_detail)
            raise HTTPException(status_code=500, detail=error_detail)

        return UploadResponse(
            message=f"Document '{filename}' uploaded successfully with {total_items_added} chunks.",
            filename=filename, gcs_url=public_gcs_url,
            research_items_added=total_items_added
        )
    except Exception as e:
        logger.error(f"Error processing upload for {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/linkedin/documents/{filename}")
async def delete_document_by_filename(filename: str):
    try:
        decoded_filename = urllib.parse.unquote(filename)
        supabase = get_supabase_client()
        
        # Find ALL research items that match this document filename
        # We need to search by both source URL pattern and data field pattern
        like_pattern = f"%{decoded_filename}"
        
        # Search by source URL containing the filename
        source_lookup = supabase.table("research").select("id, source").like("source", f"%{like_pattern}").execute()
        
        # Also search by data field containing the filename (for chunk identification)
        data_lookup = supabase.table("research").select("id, source").like("data", f"%{like_pattern}").execute()
        
        # Combine and deduplicate results
        all_items = {}
        for item in source_lookup.data + data_lookup.data:
            all_items[item['id']] = item
        
        lookup_res_data = list(all_items.values())
        
        if not lookup_res_data:
            raise HTTPException(status_code=404, detail=f"No document found with filename: {decoded_filename}")

        # Get the GCS URL from the first item (they should all have the same source)
        gcs_url_to_delete = lookup_res_data[0]['source']
        source_prefix = f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/"
        
        # Delete the file from GCS
        if gcs_url_to_delete.startswith(source_prefix):
            gcs = get_gcs_client()
            bucket = gcs.bucket(GCS_BUCKET_NAME)
            encoded_blob_name = gcs_url_to_delete.replace(source_prefix, '')
            
            # Decode the filename before accessing GCS
            blob_name = urllib.parse.unquote(encoded_blob_name)

            blob = bucket.blob(blob_name)
            if blob.exists():
                blob.delete()
                logger.info(f"Deleted GCS blob: {blob_name}")

        # Delete ALL research items (chunks) associated with this document
        ids_to_delete = [item['id'] for item in lookup_res_data]
        delete_res = supabase.table("research").delete().in_("id", ids_to_delete).execute()
        
        logger.info(f"Deleted {len(delete_res.data)} research items for document '{decoded_filename}'")
        
        return {"message": f"Document '{decoded_filename}' and its {len(delete_res.data)} chunks deleted successfully."}

    except Exception as e:
        logger.error(f"Error deleting document {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
# --- Web Crawling Endpoints ---
@app.post("/linkedin/crawl")
async def crawl_website(request: CrawlRequest):
    """
    "Crawls" a website by creating a research item summarizing the action.
    A full crawl is complex, so this simulates the result for app functionality.
    """
    try:
        from urllib.parse import urlparse
        
        parsed_url = urlparse(request.url)
        hostname = parsed_url.netloc
        if not hostname:
            raise HTTPException(status_code=400, detail="Invalid URL provided.")

        # Simulate a crawl by creating a research entry
        crawl_item = {
            "topic": f"Web Crawl: {hostname}",
            "findings": f"Content was extracted from {request.url}. This data can be used for generating insights.",
            "source": request.url,
            "data": f"Crawled on: {datetime.now().isoformat()}",
            "tags": "web-crawl,ingested",
        }

        supabase = get_supabase_client()
        response = supabase.table("research").insert(crawl_item).execute()

        if not response.data:
            raise HTTPException(status_code=500, detail="Failed to save crawl data to Supabase.")

        return response.data[0]

    except Exception as e:
        logger.error(f"Error crawling website {request.url}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/linkedin/crawls", response_model=List[ResearchItem])
async def get_crawls():
    """
    Retrieves all research items that are tagged as 'web-crawl'.
    """
    try:
        supabase = get_supabase_client()
        # Use 'ilike' for a case-insensitive search for the tag
        response = supabase.table("research").select("*").ilike("tags", "%web-crawl%").order("created_at", desc=True).execute()
        return response.data
    except Exception as e:
        logger.error(f"Error getting crawls: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch crawl data.")

# --- Research Management Endpoints ---
@app.get("/linkedin/research", response_model=List[ResearchItem])
async def get_research_items():
    supabase = get_supabase_client()
    response = supabase.table("research").select("*").order("created_at", desc=True).execute()
    return response.data

@app.post("/linkedin/research", response_model=ResearchItem)
async def add_research_item(item: ResearchItem):
    supabase = get_supabase_client()
    item_data = item.model_dump(exclude_unset=True, exclude={'id', 'created_at'})
    response = supabase.table("research").insert(item_data).execute()
    return response.data[0]

@app.put("/linkedin/research/{research_id}", response_model=ResearchItem)
async def update_research_item(research_id: int, item: ResearchItem):
    supabase = get_supabase_client()
    item_data = item.model_dump(exclude_unset=True, exclude={'id', 'created_at'})
    response = supabase.table("research").update(item_data).eq("id", research_id).execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="Research item not found.")
    return response.data[0]

@app.delete("/linkedin/research/{research_id}")
async def delete_research_item(research_id: int):
    supabase = get_supabase_client()
    response = supabase.table("research").delete().eq("id", research_id).execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="Research item not found.")
    return {"message": "Research item deleted."}

# --- Creator Management Endpoints ---
@app.get("/linkedin/creators", response_model=List[CreatorProfile])
async def get_creator_profiles():
    supabase = get_supabase_client()
    response = supabase.table("creators").select("*").order("name").execute()
    return response.data

@app.post("/linkedin/creators", response_model=CreatorProfile)
async def add_creator_profile(creator: CreatorProfile):
    supabase = get_supabase_client()
    # Only generate key if not provided
    if not creator.key:
        creator.key = generate_creator_key(creator.name)
    creator_data = creator.model_dump(exclude_unset=True, exclude={'id', 'created_at'})
    response = supabase.table("creators").insert(creator_data).execute()
    return response.data[0]

@app.put("/linkedin/creators/{creator_id}", response_model=CreatorProfile)
async def update_creator_profile(creator_id: int, creator: CreatorProfile):
    supabase = get_supabase_client()
    # Only generate key if not provided
    if not creator.key:
        creator.key = generate_creator_key(creator.name)
    creator_data = creator.model_dump(exclude_unset=True, exclude={'id', 'created_at'})
    response = supabase.table("creators").update(creator_data).eq("id", creator_id).execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="Creator not found.")
    return response.data[0]

@app.delete("/linkedin/creators/{creator_id}")
async def delete_creator_profile(creator_id: int):
    supabase = get_supabase_client()
    response = supabase.table("creators").delete().eq("id", creator_id).execute()
    if not response.data:
        raise HTTPException(status_code=404, detail="Creator not found.")
    return {"message": "Creator profile deleted."}

# --- Content Generation Endpoint ---
@app.post("/linkedin/generate", response_model=ContentGenerationResponse)
async def generate_linkedin_content(request: ContentGenerationRequest):
    try:
        # Log the incoming request for debugging
        logger.info(f"=== CONTENT GENERATION REQUEST ===")
        logger.info(f"Prompt: {request.prompt}")
        logger.info(f"Creator Key: {request.creator_key}")
        logger.info(f"Research IDs: {request.research_ids}")
        logger.info(f"Tone: {request.tone}")
        logger.info(f"Content Format: {request.content_format}")
        logger.info(f"Content Style: {request.content_style}")
        logger.info(f"Include Statistics: {request.include_statistics}")
        logger.info(f"Post Length: {request.post_length}")
        logger.info(f"Call to Action: {request.call_to_action}")
        
        # Log the full request object
        logger.info(f"Full request object: {request.model_dump()}")
        
        supabase = get_supabase_client()
        
        creator_res = supabase.table("creators").select("*").eq("key", request.creator_key).limit(1).execute()
        
        # If no creator found in database, use a smart fallback based on requested key
        if not creator_res.data:
            key_lower = (request.creator_key or "").lower().strip()
            if key_lower in {"jessica-sibley", "jessica", "sibley"}:
                creator = {
                    "name": "Jessica Sibley",
                    "tone": "Executive, data-driven, concise, audience-first",
                    "structure": "Hook → Insight with metrics → Implications for operators → Clear CTA",
                    "characteristics": "Media-operator clarity, focus on revenue levers, partnerships, customer impact"
                }
                logger.warning("Creator 'jessica-sibley' not found in DB; using built-in profile")
            else:
                creator = {
                    "name": "Default Creator",
                    "tone": "Professional and engaging",
                    "structure": "Hook → Main content → Call to action",
                    "characteristics": "Clear, concise, and professional"
                }
                logger.warning(f"Creator '{request.creator_key}' not found in database, using default creator")
        else:
            creator = creator_res.data[0]

        research_data = []
        if request.research_ids:
            research_res = supabase.table("research").select("*").in_("id", request.research_ids).execute()
            research_data = research_res.data
        else:
            # Smart default: pull the latest 5 research items to ensure contextual, data-driven visuals
            try:
                recent_res = supabase.table("research").select("*").order("created_at", desc=True).limit(5).execute()
                research_data = recent_res.data or []
            except Exception as e:
                logger.warning(f"Failed to load recent research for context: {e}")
        
        research_context = "\n".join([
            f"Research Topic: {r.get('topic')}\nKey Findings: {(r.get('findings') or '')[:800]}"
            for r in research_data
        ]) or "No specific research was provided."
        
        # Build customization instructions
        customization_instructions = []
        
        # Tone customization
        if request.tone:
            customization_instructions.append(f"TONE: Use a {request.tone} tone throughout the content.")
        else:
            customization_instructions.append(f"TONE: {creator.get('tone', 'informative')}")
        
        # Content format customization
        if request.content_format:
            if request.content_format == 'bullet_points':
                customization_instructions.append("FORMAT: MANDATORY - Use bullet points (•) for ALL main points. Structure the entire post as a list with bullet points. Each bullet point should be a complete thought. Each bullet point should be on a NEW LINE. Example format:\n• Point 1\n• Point 2\n• Point 3")
            elif request.content_format == 'numbered_list':
                customization_instructions.append("FORMAT: MANDATORY - Use numbered lists (1., 2., 3.) for ALL main points. Structure the entire post as a numbered list. You MUST create AT LEAST 3-5 numbered points. Each number should be a complete thought. Each numbered point should be on a NEW LINE. Break down your content into multiple distinct points. Example format:\n1. Point 1\n2. Point 2\n3. Point 3\n4. Point 4\n5. Point 5")
            elif request.content_format == 'paragraph':
                customization_instructions.append("FORMAT: Write in flowing paragraph format with clear transitions between ideas.")
        
        # Content style customization
        if request.content_style:
            if request.content_style == 'direct':
                customization_instructions.append("STYLE: Be direct and to the point, avoid unnecessary fluff.")
            elif request.content_style == 'storytelling':
                customization_instructions.append("STYLE: Use storytelling techniques with examples and narratives.")
            elif request.content_style == 'data_driven':
                customization_instructions.append("STYLE: Focus heavily on data, statistics, and analytical insights.")
            elif request.content_style == 'conversational':
                customization_instructions.append("STYLE: Write in a conversational, engaging manner as if talking to a friend.")
        
        # Statistics inclusion
        if request.include_statistics:
            customization_instructions.append("STATISTICS: Include specific numbers, percentages, and data points from the research.")
        
        # Post length customization
        if request.post_length:
            if request.post_length == 'short':
                customization_instructions.append("LENGTH: Keep posts concise (100-200 words) and video scripts short (30-60 seconds).")
            elif request.post_length == 'medium':
                customization_instructions.append("LENGTH: Write medium-length posts (200-300 words) and video scripts medium length (60-90 seconds).")
            elif request.post_length == 'long':
                customization_instructions.append("LENGTH: Create detailed, longer posts (300-500 words) and video scripts longer (90-120 seconds).")
        else:
            customization_instructions.append("LENGTH: Aim for 200-400 words per post and 60-90 seconds for video scripts.")
        
        # Call to action customization
        if request.call_to_action:
            customization_instructions.append(f"CALL TO ACTION: End with this specific call to action: '{request.call_to_action}'")
        
        customization_text = "\n".join(customization_instructions)
        
        # Log the customization instructions being sent to AI
        logger.info(f"=== CUSTOMIZATION INSTRUCTIONS ===")
        logger.info(f"Customization text: {customization_text}")
        logger.info(f"Number of instructions: {len(customization_instructions)}")

        system_prompt = f"""You are a LinkedIn content creator that perfectly mimics the style of {creator.get('name', 'a professional')}.

        CREATOR STYLE PROFILE:
        - Base Tone: {creator.get('tone', 'informative')}
        - Structure: {creator.get('structure', 'standard post')}
        - Key Characteristics: {creator.get('characteristics', 'clear and concise')}

        ═══════════════════════════════════════════════════════════════════════════════
        🎯 CUSTOMIZATION REQUIREMENTS (MANDATORY - OVERRIDE CREATOR STYLE):
        ═══════════════════════════════════════════════════════════════════════════════
        {customization_text}
        ═══════════════════════════════════════════════════════════════════════════════
        
        EXAMPLE FORMATS:
        • If bullet points are requested, structure like this:
        • Main point 1 with supporting details
        
        • Main point 2 with supporting details  
        
        • Main point 3 with supporting details
        
        • If numbered lists are requested, structure like this:
        1. First main point with supporting details
        
        2. Second main point with supporting details
        
        3. Third main point with supporting details
        
        ⚠️ IMPORTANT: These format examples show how the ACTUAL LinkedIn post content should be structured. The posts in the "linkedin_posts" array should follow this exact format.

        RESEARCH CONTEXT TO USE:
        {research_context}

        CRITICAL INSTRUCTIONS:
        1. 🚨 PRIORITY #1: Follow ALL customization requirements above. These override the creator's default style.
        2. Create content that sounds like the specified creator while following the customization requirements above.
        3. Incorporate insights from the provided research context.
        4. Include references to data visualizations, charts, or infographics that will be generated alongside the content.
        5. Mention specific metrics, statistics, or insights that would be visualized in charts.
        6. Use phrases like "As shown in the data visualization above", "The chart reveals", "Looking at the metrics", "The infographic demonstrates", etc.
        7. Make the content engaging and data-driven.
        8. Return ONLY a valid JSON response with these exact keys: "linkedin_posts", "video_scripts", "hashtags", "engagement_tips".
        9. The "linkedin_posts" and "video_scripts" values must be lists of strings.
        10. Each LinkedIn post should be substantial and reference visual elements that will be generated.
        🚨 JSON FORMATTING: Ensure all newlines within strings are properly escaped as \\n. Each post should be a single JSON string with \\n for line breaks.
        11. Video scripts should follow the same length requirements as specified in the LENGTH customization above.
        12. ⚠️ CRITICAL: If customization requirements are specified, they MUST be followed exactly. Do not ignore them.
        13. 📝 FORMAT COMPLIANCE: If bullet points are requested, EVERY LinkedIn post must use bullet points (•) for structure. If numbered lists are requested, EVERY post must use numbers (1., 2., 3.) for structure. Each bullet point or numbered item must be on a NEW LINE.
        14. 🎯 CONTENT STRUCTURE: The format requirements apply to the actual LinkedIn post content, not just the overall structure. Each post should be formatted exactly as requested.
        15. 📋 LINKEDIN POST FORMAT: The "linkedin_posts" array should contain posts that are already formatted with bullet points or numbered lists as requested. Do not just mention the format - actually implement it in the post content.
        16. 🔢 NUMBERED LIST REQUIREMENT: If numbered lists are requested, you MUST create AT LEAST 3-5 numbered points per post. Break down your content into multiple distinct, substantial points. Do not create just one numbered point.
        17. 🎯 FINAL CHECK: Before returning the JSON, verify that each LinkedIn post follows the requested format exactly. If bullet points are requested, each post should start with • and have multiple bullet points. If numbered lists are requested, each post should start with 1. and have multiple numbered items (at least 3-5 points).
        18. 🚨 CRITICAL: The format requirements are MANDATORY and must be implemented in the actual post content. Do not just describe the format - actually write the posts using the requested format.
        19. 📝 EXAMPLE: If bullet points are requested, a LinkedIn post should look like this: "• Key insight 1 with details\n• Key insight 2 with details\n• Key insight 3 with details" - NOT like this: "Here are the key insights in bullet point format: [then regular text]"
        """
        
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": request.prompt}]
        
        ai_response = await query_groq(messages, max_tokens=5000, temperature=0.8)

        content_data = {}
        try:
            # Remove markdown code blocks if present
            ai_response_clean = ai_response.replace('```json', '').replace('```', '')
            
            json_start = ai_response_clean.find('{')
            json_end = ai_response_clean.rfind('}') + 1

            if json_start != -1 and json_end != 0:
                json_string = ai_response_clean[json_start:json_end]
                
                # Try to fix common JSON formatting issues
                # Replace unescaped newlines within strings
                import re
                # This regex finds strings and replaces unescaped newlines within them
                json_string = re.sub(r'(?<!\\)\n(?=\s*[^"}])', '\\n', json_string)
                
                content_data = json.loads(json_string)
            else:
                raise ValueError("No valid JSON object found in the AI response.")

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse JSON from AI. Response was:\n{ai_response}")
            logger.error(f"JSON parsing error: {str(e)}")
            raise HTTPException(status_code=502, detail="The AI service returned an invalid or empty response. Please try regenerating.")

        # --- START: NEW VALIDATION LOGIC ---
        # Check if the most critical field, linkedin_posts, is missing or empty.
        if not content_data.get("linkedin_posts"):
            logger.error(f"AI returned a valid JSON but with no content. Response: {content_data}")
            raise HTTPException(status_code=502, detail="The AI service failed to generate content for the given prompt. Please try again with a more specific request.")
        # --- END: NEW VALIDATION LOGIC ---

        # Generate contextual image based on generated content
        contextual_image_url = None
        image_source_link = None
        try:
            logger.info(f"Generating contextual image for generated content")
            contextual_image_url, image_source_link = await generate_contextual_image(content_data, research_data)
            if contextual_image_url:
                logger.info(f"Successfully generated contextual image: {contextual_image_url}")
                logger.info(f"Image source link: {image_source_link}")
            else:
                logger.warning("Contextual image generation returned None")
        except Exception as e:
            logger.warning(f"Failed to generate contextual image: {e}")
            contextual_image_url = None
            image_source_link = None

        return ContentGenerationResponse(
            linkedin_posts=content_data.get("linkedin_posts", []),
            video_scripts=content_data.get("video_scripts", []),
            hashtags=content_data.get("hashtags", []),
            engagement_tips=content_data.get("engagement_tips", []),
            creator_name=creator.get('name', 'Unknown'),
            research_used=[r.get('topic') for r in research_data],
            contextual_image_url=contextual_image_url,
            image_source_link=image_source_link
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"An unexpected error occurred for creator '{request.creator_key}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected internal error occurred: {str(e)}")

# --- Enhanced Visual Content Endpoint ---
@app.post("/linkedin/generate-visual-content")
async def generate_enhanced_visual_content(
    content_topic: str = Body(..., description="The main topic or theme of the content"),
    content_type: str = Body("mixed", description="Type of visual: 'contextual', 'data', 'hybrid', 'mixed'"),
    research_ids: Optional[List[int]] = Body([], description="Research IDs to include in the visual")
):
    """
    Generate enhanced visual content that combines contextual images with data insights
    """
    try:
        logger.info(f"Generating enhanced visual content for topic: {content_topic}")
        
        # Get research data if IDs provided
        research_data = []
        if research_ids:
            supabase = get_supabase_client()
            research_res = supabase.table("research").select("*").in_("id", research_ids).execute()
            research_data = research_res.data or []
        
        # Create dummy content data for visualization
        dummy_content = {
            'linkedin_posts': [f"Content about {content_topic}"],
            'video_scripts': [],
            'hashtags': [],
            'engagement_tips': []
        }
        
        result = None
        source = None
        
        # Strategy 1: If the caller passed a topic that likely includes explicit metrics, attempt infographic
        metrics_try = create_metrics_infographic(f"Content about {content_topic}", content_topic)
        if metrics_try:
            result, source = metrics_try, "Infographic from post claims"
        else:
            result, source = None, None
        
        if not result:
            # Strategy 2: Create analytical infographic from content only (no research sources)
            result = create_content_analytical_visualization(dummy_content, content_topic)
            source = "Generated analytical visualization from content"
        
        if not result:
            # Final fallback - create simple visualization from content only
            result = create_content_analytical_visualization(dummy_content, content_topic)
            source = "Generated fallback visualization from content"
        
        return {
            "image_url": result,
            "source_link": source,
            "content_topic": content_topic,
            "content_type": content_type,
            "research_used": len(research_data),
            "generated_at": datetime.now().isoformat(),
            "message": f"Enhanced visual content generated for '{content_topic}'"
        }
        
    except Exception as e:
        logger.error(f"Error generating enhanced visual content: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate visual content: {str(e)}")

# --- Sketchy Theme Conversion Endpoint ---
@app.post("/linkedin/convert-to-sketchy")
async def convert_image_to_sketchy(
    image_url: str = Body(..., description="URL of the image to convert"),
    content_topic: str = Body("", description="Topic/theme of the content for better conversion")
):
    """
    Convert an existing image to sketchy/arty theme using AI
    """
    try:
        logger.info(f"Converting image to sketchy theme: {image_url}")
        
        # Try AI conversion first if Stability AI is available
        if STABILITY_API_KEY:
            converted_url, source = await convert_image_to_sketchy_theme(image_url, content_topic)
            if converted_url:
                return {
                    "original_image_url": image_url,
                    "sketchy_image_url": converted_url,
                    "conversion_method": "AI-powered with Stability AI",
                    "source": source,
                    "message": "Successfully converted to sketchy theme using AI"
                }
        
        # Fallback: Return the original image with a note
        return {
            "original_image_url": image_url,
            "sketchy_image_url": image_url,
            "conversion_method": "Fallback - original image returned",
            "source": "Original image (AI conversion not available)",
            "message": "AI conversion not available, returning original image. All new visualizations use sketchy theme by default."
        }
        
    except Exception as e:
        logger.error(f"Error converting image to sketchy theme: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to convert image: {str(e)}")

# --- Image Generation Endpoint ---
@app.post("/linkedin/generate-image", response_model=ImageGenerationResponse)
async def generate_image(request: ImageGenerationRequest):
    """
    Generate deterministic analytics visuals (no external AI). Sources:
    - If recent content exists in memory of the request, use it.
    - Else, use latest research items.
    """
    try:
        logger.info(f"Generating deterministic image with prompt: {request.prompt}")

        # Attempt to read latest research to seed visuals
        supabase = get_supabase_client()
        try:
            recent_res = supabase.table("research").select("*").order("created_at", desc=True).limit(5).execute()
            research_data = recent_res.data or []
        except Exception:
            research_data = []

        # Only create content-based visualizations, no research source infographics
            dummy_content = {"linkedin_posts": [request.prompt], "video_scripts": [], "hashtags": [], "engagement_tips": []}
            image_url = create_content_analytical_visualization(dummy_content, request.prompt)

        return ImageGenerationResponse(
            image_url=image_url,
            prompt=request.prompt,
            style=request.style,
            size=request.size,
            content_type=request.content_type,
            generated_at=datetime.now(),
            message=f"Generated {request.content_type} visualization (deterministic)"
        )
    except Exception as e:
        logger.error(f"Error generating deterministic image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate image: {str(e)}")

# --- Image Management Endpoints ---
@app.get("/linkedin/images")
async def list_generated_images():
    """
    List all generated images stored in GCS
    """
    try:
        gcs_client = get_gcs_client()
        bucket = gcs_client.bucket(GCS_BUCKET_NAME)
        
        # List all images in the stability_images folder
        blobs = bucket.list_blobs(prefix="stability_images/")
        
        images = []
        for blob in blobs:
            if blob.name.endswith(('.png', '.jpg', '.jpeg')):
                images.append({
                    "filename": blob.name,
                    "url": blob.public_url,
                    "size": blob.size,
                    "created": blob.time_created.isoformat() if blob.time_created else None
                })
        
        return {"images": images, "count": len(images)}
        
    except Exception as e:
        logger.error(f"Error listing images: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list images: {str(e)}")

@app.delete("/linkedin/images/{filename}")
async def delete_generated_image(filename: str):
    """
    Delete a generated image from GCS
    """
    try:
        gcs_client = get_gcs_client()
        bucket = gcs_client.bucket(GCS_BUCKET_NAME)
        
        # Ensure the filename is in the stability_images folder
        if not filename.startswith("stability_images/"):
            filename = f"stability_images/{filename}"
        
        blob = bucket.blob(filename)
        
        if not blob.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        
        blob.delete()
        
        return {"message": f"Image {filename} deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete image: {str(e)}")
    
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)