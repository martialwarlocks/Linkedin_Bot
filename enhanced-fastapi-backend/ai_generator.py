import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb

logger = logging.getLogger(__name__)

class EnhancedAIContentGenerator:
    def __init__(self, openai_client: OpenAI, embedding_model: SentenceTransformer, collection, supabase_client):
        self.openai_client = openai_client
        self.embedding_model = embedding_model
        self.collection = collection
        self.supabase = supabase_client
        
        self.creator_styles = {
            "gary-v": {
                "name": "Gary Vaynerchuk",
                "tone": "Direct, passionate, no-nonsense",
                "structure": "Hook → Personal story → Business insight → Call to action",
                "language": "Casual, uses 'you guys', lots of energy",
                "hooks": "Look..., Here's the thing..., Real talk...",
                "endings": "What do you think?, Let me know in the comments, DM me your thoughts",
                "characteristics": "High energy, direct approach, business-focused, motivational"
            },
            "simon-sinek": {
                "name": "Simon Sinek",
                "tone": "Inspirational, thoughtful, leader-focused",
                "structure": "Question → Story/Example → Leadership lesson → Reflection",
                "language": "Professional but warm, thought-provoking",
                "hooks": "Why is it that..., The best leaders..., I once worked with...",
                "endings": "What would you do?, Leadership is a choice, The choice is yours",
                "characteristics": "Leadership-focused, inspirational, storytelling, thoughtful questions"
            },
            "seth-godin": {
                "name": "Seth Godin",
                "tone": "Wise, concise, marketing-focused",
                "structure": "Insight → Brief explanation → Broader implication",
                "language": "Concise, profound, marketing terminology",
                "hooks": "The thing is..., Here's what I learned..., Marketing is...",
                "endings": "Worth considering., Just saying., Think about it.",
                "characteristics": "Concise wisdom, marketing insights, thought-provoking, minimal but impactful"
            },
            "brene-brown": {
                "name": "Brené Brown",
                "tone": "Vulnerable, authentic, research-based",
                "structure": "Personal story → Research insight → Vulnerability lesson → Connection",
                "language": "Warm, academic but accessible, uses 'we' and 'us'",
                "hooks": "I was recently..., Research shows..., Here's what I learned...",
                "endings": "What's your story?, How do you show up?, Let's connect",
                "characteristics": "Vulnerability, authenticity, research-backed, connection-focused"
            },
            "adam-grant": {
                "name": "Adam Grant",
                "tone": "Evidence-based, curious, contrarian",
                "structure": "Contrary finding → Research evidence → Practical insight → Question",
                "language": "Academic but engaging, uses data, thought-provoking",
                "hooks": "Contrary to popular belief..., New research shows..., Here's what surprised me...",
                "endings": "What do you think?, What's your experience?, Let's discuss",
                "characteristics": "Evidence-based, contrarian thinking, research-driven, curiosity"
            }
        }
    
    async def generate_contextual_content(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate contextual LinkedIn content using AI with research and document context"""
        try:
            # Gather context from multiple sources
            context_data = await self._gather_context(request)
            
            # Create enhanced prompt
            enhanced_prompt = self._create_enhanced_prompt(request['prompt'], request['creator_key'], context_data)
            
            # Generate content using OpenAI
            response = await self._generate_with_openai(enhanced_prompt, request.get('max_tokens', 2000), request.get('temperature', 0.7))
            
            # Parse and structure the response
            structured_content = self._parse_ai_response(response)
            
            # Store generation metadata
            await self._store_generation_metadata(request, context_data, structured_content)
            
            return structured_content
            
        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            raise
    
    async def _gather_context(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Gather context from research, documents, and crawled content"""
        context_data = {
            'research': [],
            'documents': [],
            'crawled_content': [],
            'vector_results': [],
            'total_context_length': 0
        }
        
        try:
            # Get research data
            if request.get('research_ids'):
                research_data = self.supabase.table('research').select('*').in_('id', request['research_ids']).execute()
                context_data['research'] = research_data.data or []
            
            # Get document data
            if request.get('document_ids'):
                doc_data = self.supabase.table('documents').select('*').in_('document_id', request['document_ids']).execute()
                context_data['documents'] = doc_data.data or []
            
            # Get crawled content
            if request.get('crawl_ids'):
                crawl_data = self.supabase.table('web_crawls').select('*').in_('crawl_id', request['crawl_ids']).execute()
                context_data['crawled_content'] = crawl_data.data or []
            
            # Get relevant content from vector database
            vector_results = await self._search_vector_db(request['prompt'])
            context_data['vector_results'] = vector_results
            
            # Calculate total context length
            total_length = 0
            for research in context_data['research']:
                total_length += len(str(research.get('findings', '')))
            for doc in context_data['documents']:
                total_length += len(str(doc.get('extracted_text', '')))
            for crawl in context_data['crawled_content']:
                total_length += len(str(crawl.get('content', '')))
            
            context_data['total_context_length'] = total_length
            
            return context_data
            
        except Exception as e:
            logger.error(f"Context gathering failed: {e}")
            return context_data
    
    async def _search_vector_db(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search vector database for relevant content"""
        try:
            # Create query embedding
            query_embedding = self.embedding_model.encode(query)
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=limit,
                include=['documents', 'metadatas', 'distances']
            )
            
            return [
                {
                    'content': doc,
                    'metadata': meta,
                    'relevance_score': 1 - distance
                }
                for doc, meta, distance in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )
            ]
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _create_enhanced_prompt(self, prompt: str, creator_key: str, context_data: Dict[str, Any]) -> str:
        """Create an enhanced prompt with context and creator style"""
        creator_style = self.creator_styles.get(creator_key, self.creator_styles["gary-v"])
        
        # Build context string
        context_parts = []
        
        if context_data['research']:
            research_context = "Research Context:\n" + "\n".join([
                f"- {r.get('topic', 'Unknown')}: {r.get('findings', '')}"
                for r in context_data['research']
            ])
            context_parts.append(research_context)
        
        if context_data['vector_results']:
            vector_context = "Relevant Content:\n" + "\n".join([
                f"- {result['content'][:200]}..."
                for result in context_data['vector_results'][:3]
            ])
            context_parts.append(vector_context)
        
        if context_data['documents']:
            doc_context = "Document Context:\n" + "\n".join([
                f"- {doc.get('filename', 'Unknown')}: {doc.get('extracted_text', '')[:200]}..."
                for doc in context_data['documents']
            ])
            context_parts.append(doc_context)
        
        context_string = "\n\n".join(context_parts)
        
        # Create the enhanced prompt
        enhanced_prompt = f"""
You are an expert LinkedIn content creator specializing in {creator_style['name']}'s style.

Creator Style:
- Tone: {creator_style['tone']}
- Structure: {creator_style['structure']}
- Language: {creator_style['language']}
- Hooks: {creator_style['hooks']}
- Endings: {creator_style['endings']}
- Characteristics: {creator_style['characteristics']}

Context Information:
{context_string}

User Request: {prompt}

Please create:
1. A LinkedIn post in {creator_style['name']}'s style (2-3 paragraphs)
2. A LinkedIn reel transcript (30-60 seconds)
3. Key talking points for the content
4. Relevant hashtags
5. Engagement questions

Format your response as JSON:
{{
    "linkedin_post": "the post content",
    "linkedin_reel_transcript": "the transcript",
    "talking_points": ["point1", "point2", "point3"],
    "hashtags": ["#hashtag1", "#hashtag2", "#hashtag3"],
    "engagement_questions": ["question1", "question2"],
    "style_notes": "notes about how the style was applied",
    "context_used": "summary of what context was incorporated"
}}
"""
        
        return enhanced_prompt
    
    async def _generate_with_openai(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate content using OpenAI API"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert LinkedIn content creator and social media strategist. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
    
    def _parse_ai_response(self, response: str) -> Dict[str, Any]:
        """Parse AI response and extract structured content"""
        try:
            # Try to parse as JSON
            if response.strip().startswith('{'):
                parsed = json.loads(response)
                # Ensure all required fields exist
                return {
                    'linkedin_post': parsed.get('linkedin_post', ''),
                    'linkedin_reel_transcript': parsed.get('linkedin_reel_transcript', ''),
                    'talking_points': parsed.get('talking_points', []),
                    'hashtags': parsed.get('hashtags', []),
                    'engagement_questions': parsed.get('engagement_questions', []),
                    'style_notes': parsed.get('style_notes', ''),
                    'context_used': parsed.get('context_used', '')
                }
            
            # Fallback parsing
            lines = response.split('\n')
            content = {
                'linkedin_post': '',
                'linkedin_reel_transcript': '',
                'talking_points': [],
                'hashtags': [],
                'engagement_questions': [],
                'style_notes': '',
                'context_used': ''
            }
            
            current_section = None
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if 'linkedin post' in line.lower():
                    current_section = 'linkedin_post'
                elif 'reel transcript' in line.lower():
                    current_section = 'linkedin_reel_transcript'
                elif 'talking points' in line.lower():
                    current_section = 'talking_points'
                elif 'hashtags' in line.lower():
                    current_section = 'hashtags'
                elif 'engagement' in line.lower():
                    current_section = 'engagement_questions'
                elif current_section:
                    if current_section in ['linkedin_post', 'linkedin_reel_transcript']:
                        content[current_section] += line + '\n'
                    elif current_section == 'talking_points':
                        if line.startswith('-') or line.startswith('•'):
                            content['talking_points'].append(line[1:].strip())
                    elif current_section == 'hashtags':
                        if '#' in line:
                            hashtags = [tag.strip() for tag in line.split() if tag.startswith('#')]
                            content['hashtags'].extend(hashtags)
                    elif current_section == 'engagement_questions':
                        if line.startswith('-') or line.startswith('•') or line.endswith('?'):
                            content['engagement_questions'].append(line.strip())
            
            return content
            
        except Exception as e:
            logger.error(f"Failed to parse AI response: {e}")
            return {
                'linkedin_post': response,
                'linkedin_reel_transcript': 'Transcript generation failed',
                'talking_points': [],
                'hashtags': [],
                'engagement_questions': [],
                'style_notes': 'Response parsing failed',
                'context_used': 'No context used due to parsing error'
            }
    
    async def _store_generation_metadata(self, request: Dict[str, Any], context_data: Dict[str, Any], content: Dict[str, Any]):
        """Store generation metadata for analytics"""
        try:
            metadata = {
                'prompt': request['prompt'],
                'creator_key': request['creator_key'],
                'research_ids': request.get('research_ids', []),
                'document_ids': request.get('document_ids', []),
                'crawl_ids': request.get('crawl_ids', []),
                'context_length': context_data['total_context_length'],
                'generated_at': datetime.now().isoformat(),
                'content_length': len(str(content)),
                'style_used': request['creator_key']
            }
            
            self.supabase.table('content_generations').insert(metadata).execute()
            
        except Exception as e:
            logger.error(f"Failed to store generation metadata: {e}")
    
    async def generate_research_summary(self, research_ids: List[int]) -> Dict[str, Any]:
        """Generate a summary of research findings"""
        try:
            if not research_ids:
                return {"summary": "No research provided", "key_insights": []}
            
            research_data = self.supabase.table('research').select('*').in_('id', research_ids).execute()
            research_items = research_data.data or []
            
            if not research_items:
                return {"summary": "No research found", "key_insights": []}
            
            # Create research summary prompt
            research_text = "\n\n".join([
                f"Topic: {item.get('topic', 'Unknown')}\nFindings: {item.get('findings', '')}\nData: {item.get('data', '')}"
                for item in research_items
            ])
            
            prompt = f"""
Based on the following research, create a comprehensive summary and extract key insights:

{research_text}

Please provide:
1. A concise summary of the main findings
2. 3-5 key insights that could be used for content creation
3. Potential content angles

Format as JSON:
{{
    "summary": "main summary",
    "key_insights": ["insight1", "insight2", "insight3"],
    "content_angles": ["angle1", "angle2", "angle3"]
}}
"""
            
            response = await self._generate_with_openai(prompt, 1000, 0.7)
            return self._parse_ai_response(response)
            
        except Exception as e:
            logger.error(f"Research summary generation failed: {e}")
            return {"summary": "Failed to generate summary", "key_insights": []}
    
    async def generate_document_insights(self, document_ids: List[str]) -> Dict[str, Any]:
        """Generate insights from uploaded documents"""
        try:
            if not document_ids:
                return {"insights": "No documents provided", "key_points": []}
            
            doc_data = self.supabase.table('documents').select('*').in_('document_id', document_ids).execute()
            documents = doc_data.data or []
            
            if not documents:
                return {"insights": "No documents found", "key_points": []}
            
            # Create document analysis prompt
            doc_text = "\n\n".join([
                f"Document: {doc.get('filename', 'Unknown')}\nContent: {doc.get('extracted_text', '')[:1000]}..."
                for doc in documents
            ])
            
            prompt = f"""
Analyze the following documents and extract key insights for LinkedIn content creation:

{doc_text}

Please provide:
1. Main themes and topics
2. Key insights that could be shared
3. Potential content ideas
4. Professional implications

Format as JSON:
{{
    "themes": ["theme1", "theme2"],
    "key_insights": ["insight1", "insight2"],
    "content_ideas": ["idea1", "idea2"],
    "professional_implications": ["implication1", "implication2"]
}}
"""
            
            response = await self._generate_with_openai(prompt, 1000, 0.7)
            return self._parse_ai_response(response)
            
        except Exception as e:
            logger.error(f"Document insights generation failed: {e}")
            return {"insights": "Failed to generate insights", "key_points": []} 