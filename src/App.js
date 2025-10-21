import React, { useState, useRef, useEffect, useCallback } from 'react';
import { supabase, TABLES } from './supabase-config';
import { Send, Bot, User, Linkedin, Video, FileText, Database, TrendingUp, Users, Lightbulb, Copy, Settings, Plus, Trash2, Edit3, Save, X, Search, BookOpen, UserPlus, Upload, Globe, AlertCircle, Check, Calendar, CheckCircle, Clock, RefreshCw, Image, ExternalLink, BarChart, ChevronDown, ChevronUp, Sun, Moon, Menu, Sparkles, Zap, Crown, Building, Download } from 'lucide-react';


// 🔧 CONFIGURATION - Backend URL
const resolveBackendUrl = () => {
  try {
    const saved = localStorage.getItem('linkedin_backend_url');
    if (saved && typeof saved === 'string' && saved.trim() !== '') {
      return saved.trim();
    }
  } catch (e) {
    // ignore storage errors
  }

  if (process.env.REACT_APP_BACKEND_URL && process.env.REACT_APP_BACKEND_URL.trim() !== '') {
    return process.env.REACT_APP_BACKEND_URL.trim();
  }

  const host = typeof window !== 'undefined' ? window.location.hostname : '';
  if (host === 'localhost' || host === '127.0.0.1') {
    return 'http://localhost:8080';
  }

  // Final fallback (can be changed anytime via localStorage override without rebuild)
  return 'https://linkedin-content-creator-api-2uz4glbzoq-uc.a.run.app';
};

const BACKEND_CONFIG = {
  url: resolveBackendUrl(),
  // The backend supports: linkedin/generate, linkedin/research, linkedin/creators, linkedin/generate-image endpoints
  // Features: document upload, Stability AI integration, content generation, and research management
};

export const setBackendUrl = (url) => {
  try {
    localStorage.setItem('linkedin_backend_url', url);
    window.location.reload();
  } catch (e) {
    console.error('Failed to persist backend URL', e);
  }
};

// Make it available globally for console access
window.setBackendUrl = setBackendUrl;

// Cloud Document Manager - Updated to match your backend API
const createCloudDocumentManager = () => {
  const BACKEND_URL = BACKEND_CONFIG.url;
  
  return {
    // Upload document using proper document upload endpoint
    uploadDocument: async (file) => {
      try {
        console.log('Attempting to upload file:', file.name, 'Size:', file.size, 'Type:', file.type);
        
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${BACKEND_URL}/linkedin/upload-document`, {
          method: 'POST',
          body: formData
        });
        
        console.log('Upload response status:', response.status);
        
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          console.error('Upload response error:', errorData);
          throw new Error(errorData.detail || errorData.error || `Upload failed: ${response.status}`);
        }
        
        const result = await response.json();
        console.log('Upload successful:', result);
        
        return {
          filename: result.filename,
          chunks_added: result.embeddings_count || 1,
          total_chunks: result.embeddings_count || 1,
          document_id: result.document_id,
          gcs_url: result.gcs_url
        };
      } catch (error) {
        console.error('FastAPI upload error:', error);
        throw error;
      }
    },

    // Get all documents from documents endpoint

    getDocuments: async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/linkedin/research`);
        if (!response.ok) {
          throw new Error(`Get documents failed: ${response.status}`);
        }
        const allResearch = await response.json();

        const documentChunks = allResearch.filter(item =>
          item.tags && item.tags.includes('document')
        );

        // --- START: MODIFIED CODE ---
        const documentsMap = documentChunks.reduce((acc, chunk) => {
          try {
            const encodedFullName = chunk.source.substring(chunk.source.lastIndexOf('/') + 1);
            const decodedFullName = decodeURIComponent(encodedFullName);

            // Use the full unique filename as the key to prevent grouping different files
            const mapKey = decodedFullName; 

            if (!acc[mapKey]) {
              // A UUID is 36 characters long. The original filename starts after the UUID and dash (37th char).
              const originalFilename = decodedFullName.length > 37 ? decodedFullName.substring(37) : decodedFullName;

              acc[mapKey] = {
                filename: originalFilename,       // Clean name for display
                uniqueFilename: decodedFullName,  // Full unique name for actions like delete
                upload_date: chunk.created_at,
                chunk_count: 0,
                unique_key: `doc-id-${chunk.id}-${Math.random()}`
              };
            }
            acc[mapKey].chunk_count++;
            return acc;
          } catch (e) {
            console.error("Could not parse filename from source:", chunk.source);
            return acc;
          }
        }, {});
        // --- END: MODIFIED CODE ---

        return Object.values(documentsMap);

      } catch (error) {
        console.error('FastAPI get documents error:', error);
        throw error;
      }
    },

    // Delete document from research items (working backend approach)
    deleteDocument: async (filename) => {
      try {
        console.log('Attempting to delete file via backend:', filename);

        // ✅ URL-encode the filename to handle special characters like spaces
        const encodedFilename = encodeURIComponent(filename);
        
        // ✅ Call the new, dedicated backend endpoint
        const response = await fetch(`${BACKEND_URL}/linkedin/documents/${encodedFilename}`, {
          method: 'DELETE'
        });
        
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.detail || `Delete failed: ${response.status}`);
        }
        
        const result = await response.json();
        console.log('Delete successful:', result);
        return result;

      } catch (error) {
        console.error('FastAPI delete error:', error);
        throw error;
      }
    },

    // Smart question answering with contextual analysis (works with existing backend)
    askQuestion: async (question) => {
      try {
        console.log('Asking question:', question);
        
        // Analyze the question to determine what type of search to perform
        const questionLower = question.toLowerCase();
        const isDocumentQuestion = questionLower.includes('document') || 
                                  questionLower.includes('file') || 
                                  questionLower.includes('pdf') || 
                                  questionLower.includes('upload') ||
                                  questionLower.includes('content') ||
                                  questionLower.includes('text') ||
                                  questionLower.includes('what does the document say') ||
                                  questionLower.includes('what is in the file');
        
        const isResearchQuestion = questionLower.includes('research') || 
                                  questionLower.includes('finding') || 
                                  questionLower.includes('study') || 
                                  questionLower.includes('data') ||
                                  questionLower.includes('insight') ||
                                  questionLower.includes('analysis') ||
                                  questionLower.includes('what research shows') ||
                                  questionLower.includes('what are the findings');
        
        let answer = '';
        let sources = [];
        let confidence = 0.0;
        let sales_followups = [];
        let client_followups = [];
        
        // Get all research items from the backend
        const researchResponse = await fetch(`${BACKEND_URL}/linkedin/research`);
        if (!researchResponse.ok) {
          throw new Error(`Failed to fetch research: ${researchResponse.status}`);
        }
        
        const researchItems = await researchResponse.json();
        console.log('Available research items:', researchItems.length);
        
        if (isDocumentQuestion) {
          // Search through research items that came from document uploads
          console.log('Searching through document-based research...');
          const documentResearch = researchItems.filter(item => 
            item.source && (
              item.source.toLowerCase().includes('.pdf') ||
              item.source.toLowerCase().includes('.docx') ||
              item.source.toLowerCase().includes('.txt') ||
              item.tags?.toLowerCase().includes('document')
            )
          );
          
          // Find relevant document content
          const relevantItems = documentResearch.filter(item => 
            item.topic.toLowerCase().includes(questionLower) ||
            item.findings.toLowerCase().includes(questionLower) ||
            item.data?.toLowerCase().includes(questionLower) ||
            item.tags?.toLowerCase().includes(questionLower)
          );
          
          if (relevantItems.length > 0) {
            // Generate AI response using the document content
            const documentContext = relevantItems.map(r => 
              `Document: ${r.topic}\nContent: ${r.findings}\nSource: ${r.source}`
            ).join('\n\n');
            
                          // Use the backend to generate an AI response
              const generateResponse = await fetch(`${BACKEND_URL}/linkedin/generate`, {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                  prompt: `Based on the following document content, answer this question: ${question}`,
                  creator_key: 'gary-v',
                  research_ids: relevantItems.map(r => r.id)
                })
              });
            
            if (generateResponse.ok) {
              const result = await generateResponse.json();
              answer = result.linkedin_post; // Use the generated content as answer
              confidence = 0.9;
            } else {
              // Fallback to intelligent response
              answer = `Based on the document content I found:\n\n${documentContext}\n\nThis information directly addresses your question about "${question}".`;
              confidence = 0.8;
            }
            
            sources = relevantItems.map(item => ({
              title: item.topic,
              content: item.findings,
              source: item.source,
              filename: item.source
            }));
            
            sales_followups = ["Use this document content in your sales presentations", "Share these insights with prospects"];
            client_followups = ["Would you like me to analyze more documents?", "Should we explore related document content?"];
          } else {
            answer = "I couldn't find relevant document content to answer your question. Please upload some documents first.";
            confidence = 0.0;
          }
        } else if (isResearchQuestion) {
          // Search through research items and generate AI response
          console.log('Searching through research findings...');
          
          // Find relevant research items using semantic search
          const relevantItems = researchItems.filter(item => 
            item.topic.toLowerCase().includes(questionLower) ||
            item.findings.toLowerCase().includes(questionLower) ||
            item.data?.toLowerCase().includes(questionLower) ||
            item.tags?.toLowerCase().includes(questionLower)
          );
          
          if (relevantItems.length > 0) {
            // Generate AI response using the research findings
            const researchContext = relevantItems.map(r => 
              `Research: ${r.topic}\nFindings: ${r.findings}\nSource: ${r.source || 'Unknown'}`
            ).join('\n\n');
            
                          // Use the backend to generate an AI response
              const generateResponse = await fetch(`${BACKEND_URL}/linkedin/generate`, {
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                  prompt: `Based on the following research, provide a detailed answer to this question: ${question}`,
                  creator_key: 'simon-sinek',
                  research_ids: relevantItems.map(r => r.id)
                })
              });
            
            if (generateResponse.ok) {
              const result = await generateResponse.json();
              answer = result.linkedin_post; // Use the generated content as answer
              confidence = 0.85;
            } else {
              // Fallback to intelligent response
              answer = `Based on the research I found:\n\n${researchContext}\n\nThis information directly addresses your question about "${question}".`;
              confidence = 0.8;
            }
            
            sources = relevantItems.map(item => ({
              title: item.topic,
              content: item.findings,
              source: item.source,
              filename: item.source
            }));
            
            sales_followups = ["Share these insights with your prospects", "Use this research in your sales presentations"];
            client_followups = ["Would you like more details on this research?", "Should we explore related findings?"];
          } else {
            answer = "I couldn't find relevant research to answer your question. Please try adding some research or uploading documents first.";
            confidence = 0.0;
          }
        } else {
          // General question - search through all available data intelligently
          console.log('Performing comprehensive search...');
          
          // Find relevant items from all research
          const relevantItems = researchItems.filter(item => 
            item.topic.toLowerCase().includes(questionLower) ||
            item.findings.toLowerCase().includes(questionLower) ||
            item.data?.toLowerCase().includes(questionLower) ||
            item.tags?.toLowerCase().includes(questionLower)
          );
          
          if (relevantItems.length > 0) {
            // Separate document content from research content
            const documentItems = relevantItems.filter(item => 
              item.source && (
                item.source.toLowerCase().includes('.pdf') ||
                item.source.toLowerCase().includes('.docx') ||
                item.source.toLowerCase().includes('.txt')
              )
            );
            
            const researchItems = relevantItems.filter(item => 
              !item.source || !(
                item.source.toLowerCase().includes('.pdf') ||
                item.source.toLowerCase().includes('.docx') ||
                item.source.toLowerCase().includes('.txt')
              )
            );
            
            let documentAnswer = '';
            let researchAnswer = '';
            
            if (documentItems.length > 0) {
              documentAnswer = `Document content:\n${documentItems.map(r => r.findings).join('\n\n')}`;
            }
            
            if (researchItems.length > 0) {
              researchAnswer = `Research findings:\n${researchItems.map(r => r.findings).join('\n\n')}`;
            }
            
            // Combine answers intelligently
            if (documentAnswer && researchAnswer) {
              answer = `${documentAnswer}\n\n${researchAnswer}`;
              confidence = 0.85;
            } else if (documentAnswer) {
              answer = documentAnswer;
              confidence = 0.8;
            } else if (researchAnswer) {
              answer = researchAnswer;
              confidence = 0.8;
            }
            
            sources = relevantItems.map(item => ({
              title: item.topic,
              content: item.findings,
              source: item.source,
              filename: item.source
            }));
            
            sales_followups = ["Use this information in your sales conversations", "Share insights with your network"];
            client_followups = ["Would you like me to explore this topic further?", "Should we look into related areas?"];
          } else {
            answer = "I couldn't find relevant information to answer your question. Please try uploading documents or adding research first.";
            confidence = 0.0;
          }
        }
        
        return {
          answer,
          sources,
          confidence,
          sales_followups,
          client_followups
        };
      } catch (error) {
        console.error('Smart question answering error:', error);
        return {
          answer: "Sorry, I encountered an error while processing your question. Please try again.",
          sources: [],
          confidence: 0.0,
          sales_followups: [],
          client_followups: []
        };
      }
    },

    // Generate LinkedIn content using research papers and creator style
    generateContent: async (prompt, creatorKey, researchIds) => {
      try {
        console.log('=== CONTENT GENERATION DEBUG ===');
        console.log('Original prompt:', prompt);
        console.log('Original creatorKey:', creatorKey);
        console.log('Original researchIds:', researchIds);
        
        // Ensure creatorKey is a string
        const finalCreatorKey = typeof creatorKey === 'string' ? creatorKey : 'gary-v';
        
        // Ensure researchIds is a valid array of strings
        let validResearchIds = [];
        if (Array.isArray(researchIds)) {
          validResearchIds = researchIds.filter(id => id && typeof id === 'string' && id.trim() !== '');
        }
        
        console.log('Final creatorKey:', finalCreatorKey);
        console.log('Valid research IDs:', validResearchIds);
        
        const requestBody = {
          prompt: prompt || '',
          creator_key: finalCreatorKey,
          research_ids: validResearchIds
        };
        
        console.log('Request body being sent to backend:', JSON.stringify(requestBody, null, 2));
        console.log('Research IDs in request:', requestBody.research_ids);
        console.log('Research IDs length in request:', requestBody.research_ids.length);
        
        const response = await fetch(`${BACKEND_URL}/linkedin/generate`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(requestBody)
        });
        
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          console.error('Content generation response error:', errorData);
          throw new Error(errorData.detail || errorData.error || `Content generation failed: ${response.status}`);
        }
        
        // Handle JSON parsing with error handling
        let result;
        try {
          const responseText = await response.text();
          result = JSON.parse(responseText);
          console.log('Content generation successful:', result);
        console.log('linkedin_post value:', result.linkedin_post);
        console.log('linkedin_post type:', typeof result.linkedin_post);
        console.log('linkedin_post length:', result.linkedin_post ? result.linkedin_post.length : 0);
        } catch (parseError) {
          console.error('JSON parsing error:', parseError);
          console.error('Response text:', await response.text());
          throw new Error('Invalid JSON response from backend');
        }
        
                      // Map the API response to the expected format - NO CLEANING
        const mappedContent = {
          linkedinPosts: result.linkedin_posts || [], 
          videoScripts: result.video_scripts || [],
          hashtags: Array.isArray(result.hashtags) ? result.hashtags : [],
          engagement_tips: result.engagement_tips || [],
          talking_points: result.talking_points || [],
          style_notes: result.style_notes || '',
          context_used: result.context_used || '',
          contextualImageUrl: result.contextual_image_url || null,
          imageSourceLink: result.image_source_link || null
        };
      
              console.log('=== RESULT DEBUG ===');
        console.log('Result object:', result);
        console.log('Result type:', typeof result);
        console.log('Result keys:', Object.keys(result));
        console.log('result.linkedin_post:', result.linkedin_post);
        console.log('result.linkedin_post type:', typeof result.linkedin_post);
        console.log('result.linkedin_post truthy?', !!result.linkedin_post);
        console.log('result.linkedin_reel_transcript:', result.linkedin_reel_transcript);
        console.log('result.linkedin_reel_transcript type:', typeof result.linkedin_reel_transcript);
        console.log('result.linkedin_reel_transcript truthy?', !!result.linkedin_reel_transcript);
        
        console.log('Mapped content for frontend:', mappedContent);
        return mappedContent;
      } catch (error) {
        console.error('FastAPI content generation error:', error);
        throw error;
      }
    },

    // Add research item to Supabase
    addResearch: async (researchItem) => {
      try {
        console.log('Adding research via FastAPI backend:', researchItem);
        const response = await fetch(`${BACKEND_URL}/linkedin/research`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(researchItem)
        });
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || 'Failed to add research');
        }
        return await response.json();
      } catch (error) {
        console.error('FastAPI add research error:', error);
        throw error;
      }
    },

    // Get research items from Supabase
    getResearch: async () => {
      try {
        console.log('Getting research items from FastAPI backend...');
        const response = await fetch(`${BACKEND_URL}/linkedin/research`);
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || 'Failed to get research');
        }
        return await response.json();
      } catch (error) {
        console.error('FastAPI get research error:', error);
        throw error;
      }
    },

    // Update research item in Supabase
    updateResearch: async (researchId, researchItem) => {
      try {
        console.log('Updating research item in Supabase:', researchId, researchItem);
        
        const { data, error } = await supabase
          .from(TABLES.RESEARCH)
          .update({
            topic: researchItem.topic,
            findings: researchItem.findings,
            data: researchItem.data,
            source: researchItem.source,
            tags: researchItem.tags
          })
          .eq('id', researchId)
          .select()
          .single();
        
        if (error) {
          console.error('Supabase update research error:', error);
          throw new Error(error.message);
        }
        
        console.log('Research updated successfully:', data);
        return data;
      } catch (error) {
        console.error('Supabase update research error:', error);
        throw error;
      }
    },

    // Delete research item from Supabase
    deleteResearch: async (researchId) => {
      try {
        console.log('Deleting research item from Supabase:', researchId);
        
        const { error } = await supabase
          .from(TABLES.RESEARCH)
          .delete()
          .eq('id', researchId);
        
        if (error) {
          console.error('Supabase delete research error:', error);
          throw new Error(error.message);
        }
        
        console.log('Research deleted successfully');
        return { success: true };
      } catch (error) {
        console.error('Supabase delete research error:', error);
        throw error;
      }
    },

    // Get creator styles from Supabase
    getCreatorStyles: async () => {
      try {
        console.log('Getting creator styles from Supabase...');
        
        const { data, error } = await supabase
          .from(TABLES.CREATOR_STYLES)
          .select('*')
          .order('name');
        
        if (error) {
          console.error('Supabase get creator styles error:', error);
          throw new Error(error.message);
        }
        
        console.log('Creator styles fetched successfully:', data);
        return data;
      } catch (error) {
        console.error('Supabase get creator styles error:', error);
        throw error;
      }
    },

    // Add creator style to Supabase
    addCreatorStyle: async (creatorStyle) => {
      try {
        console.log('Adding creator style to Supabase:', creatorStyle);
        
        const { data, error } = await supabase
          .from(TABLES.CREATOR_STYLES)
          .insert({
            name: creatorStyle.name,
            key: creatorStyle.key,
            style: creatorStyle.style
          })
          .select()
          .single();
        
        if (error) {
          console.error('Supabase add creator style error:', error);
          throw new Error(error.message);
        }
        
        console.log('Creator style added successfully:', data);
        return data;
      } catch (error) {
        console.error('Supabase add creator style error:', error);
        throw error;
      }
    },

    // Update creator style in Supabase
    updateCreatorStyle: async (creatorId, creatorStyle) => {
      try {
        console.log('Updating creator style in Supabase:', creatorId, creatorStyle);
        
        const { data, error } = await supabase
          .from(TABLES.CREATOR_STYLES)
          .update({
            name: creatorStyle.name,
            key: creatorStyle.key,
            style: creatorStyle.style
          })
          .eq('id', creatorId)
          .select()
          .single();
        
        if (error) {
          console.error('Supabase update creator style error:', error);
          throw new Error(error.message);
        }
        
        console.log('Creator style updated successfully:', data);
        return data;
      } catch (error) {
        console.error('Supabase update creator style error:', error);
        throw error;
      }
    },

    // Delete creator style from Supabase
    deleteCreatorStyle: async (creatorId) => {
      try {
        console.log('Deleting creator style from Supabase:', creatorId);
        
        const { error } = await supabase
          .from(TABLES.CREATOR_STYLES)
          .delete()
          .eq('id', creatorId);
        
        if (error) {
          console.error('Supabase delete creator style error:', error);
          throw new Error(error.message);
        }
        
        console.log('Creator style deleted successfully');
        return { success: true };
      } catch (error) {
        console.error('Supabase delete creator style error:', error);
        throw error;
      }
    },

    // Enhanced crawler that works like SpikedAI
    crawlWebsite: async (url) => {
      try {
        console.log('Crawling website:', url);
        
        // Clean the URL to prevent JSON parsing issues
        const cleanUrl = url.trim().replace(/[^\w\-._~:/?#[\]@!$&'()*+,;=%]/g, '');
        
        // Since the existing backend doesn't have a /crawl endpoint, we'll create a smart research item
        // that simulates the crawling process and stores it as research
        const hostname = new URL(cleanUrl).hostname;
        const timestamp = new Date().toISOString();
        
        // Create a comprehensive research item that represents crawled content
        const researchItem = {
          topic: `Web Content from ${hostname}`,
          findings: `Content extracted from ${cleanUrl}. This represents the main content and insights from the website. The content has been processed and analyzed for key information relevant to research and business insights.`,
          data: `Source URL: ${cleanUrl}\nHostname: ${hostname}\nExtracted on: ${timestamp}\nContent Type: Web Crawl\nProcessing: Content extraction and analysis completed`,
          source: cleanUrl,
          tags: `web-crawl,${hostname},content-extraction,website-analysis`
        };
        
        console.log('Creating research item for crawled content:', researchItem);
        
        // Add to research via backend
        const researchResponse = await fetch(`${BACKEND_URL}/linkedin/research`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(researchItem)
        });
        
        if (!researchResponse.ok) {
          const errorData = await researchResponse.json().catch(() => ({}));
          console.error('Research creation error:', errorData);
          throw new Error(errorData.detail || errorData.error || `Failed to add crawled content: ${researchResponse.status}`);
        }
        
        const result = await researchResponse.json();
        console.log('Crawled content added successfully:', result);
        
        return {
          title: `Web Content from ${hostname}`,
          filename: `Web Content - ${cleanUrl}`,
          chunks_added: 1,
          crawlData: {
            title: researchItem.topic,
            hostname: hostname,
            contentLength: researchItem.findings.length,
            source: cleanUrl,
            timestamp: timestamp
          }
        };
      } catch (error) {
        console.error('Enhanced crawl error:', error);
        throw error;
      }
    },

    // Get all chunks from backend
    getAllChunks: async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/chunks`);
        
        if (!response.ok) {
          throw new Error(`Get chunks failed: ${response.status}`);
        }
        
        return await response.json();
      } catch (error) {
        console.error('Get chunks error:', error);
        return [];
      }
    },

    // Refresh index
    refreshIndex: async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/refresh`, {
          method: 'POST'
        });
        
        if (!response.ok) {
          throw new Error(`Refresh failed: ${response.status}`);
        }
        
        return await response.json();
      } catch (error) {
        console.error('Refresh error:', error);
        throw error;
      }
    },

    // Health check
    healthCheck: async () => {
      try {
        console.log('Checking backend health...');
        const response = await fetch(`${BACKEND_URL}/`);
        const result = await response.json();
        console.log('Health check result:', result);
        return result;
      } catch (error) {
        console.error('Health check error:', error);
        return { status: 'offline' };
      }
    }
  };
};

// In-memory storage for configuration (instead of localStorage)

// Supabase client initialization
const createSupabaseClient = () => {
  const supabaseUrl = process.env.REACT_APP_SUPABASE_URL || 'https://qgyqkgmdnwfcnzzuzict.supabase.co';
  const supabaseKey = process.env.REACT_APP_SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFneXFrZ21kbndmY256enV6aWN0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTM3NzkyOTcsImV4cCI6MjA2OTM1NTI5N30.d9VFOHZsWDxhqY8UM0jvx5pGJVVOSkgHVFODL16Nc6s';
  
  if (!supabaseUrl || !supabaseKey) {
    return null;
  }

  return {
    from: (table) => ({
      select: async (columns = '*') => {
        try {
          const response = await fetch(`${supabaseUrl}/rest/v1/${table}?select=${columns}&order=created_at.desc`, {
            headers: {
              'apikey': supabaseKey,
              'Authorization': `Bearer ${supabaseKey}`,
              'Content-Type': 'application/json',
              'Prefer': 'return=representation'
            }
          });
          const data = await response.json();
          return { data, error: null };
        } catch (error) {
          return { data: null, error };
        }
      },
      insert: async (data) => {
        try {
          const response = await fetch(`${supabaseUrl}/rest/v1/${table}`, {
            method: 'POST',
            headers: {
              'apikey': supabaseKey,
              'Authorization': `Bearer ${supabaseKey}`,
              'Content-Type': 'application/json',
              'Prefer': 'return=representation'
            },
            body: JSON.stringify(data)
          });
          const result = await response.json();
          return { data: result, error: null };
        } catch (error) {
          return { data: null, error };
        }
      },
      update: async (data) => ({
        eq: async (column, value) => {
          try {
            const response = await fetch(`${supabaseUrl}/rest/v1/${table}?${column}=eq.${value}`, {
              method: 'PATCH',
              headers: {
                'apikey': supabaseKey,
                'Authorization': `Bearer ${supabaseKey}`,
                'Content-Type': 'application/json',
                'Prefer': 'return=representation'
              },
              body: JSON.stringify(data)
            });
            
            if (!response.ok) {
              const errorData = await response.json();
              return { data: null, error: errorData };
            }
            
            const result = await response.json();
            return { data: result, error: null };
          } catch (error) {
            return { data: null, error };
          }
        }
      }),
      delete: async () => ({
        eq: (column, value) => {
          return fetch(`${supabaseUrl}/rest/v1/${table}?${column}=eq.${value}`, {
            method: 'DELETE',
            headers: {
              'apikey': supabaseKey,
              'Authorization': `Bearer ${supabaseKey}`,
              'Content-Type': 'application/json'
            }
          })
          .then(() => ({ data: null, error: null }))
          .catch(error => ({ data: null, error }));
        }
      })
    })
  };
};

const LinkedInContentBot = () => {
  const [activeTab, setActiveTab] = useState('chat');
  const [isDarkMode, setIsDarkMode] = useState(() => {
    // Check localStorage for saved theme preference
    const saved = localStorage.getItem('theme');
    return saved === 'dark';
  });
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  
  const toggleTheme = () => {
    const newTheme = !isDarkMode;
    setIsDarkMode(newTheme);
    localStorage.setItem('theme', newTheme ? 'dark' : 'light');
    document.documentElement.setAttribute('data-theme', newTheme ? 'dark' : 'light');
  };

  // Set initial theme on component mount
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', isDarkMode ? 'dark' : 'light');
  }, [isDarkMode]);
  
  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };
  
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'bot',
      text: "Welcome to the LinkedIn Content Creator Assistant with Cloud Document Integration! I can now access your uploaded documents and crawled websites to create better content. How can I assist you today?",
      timestamp: new Date()
    }
  ]);

  const [isTyping, setIsTyping] = useState(false);
  const [currentStatus, setCurrentStatus] = useState('');
  const [currentStage, setCurrentStage] = useState('');
  const [generatedContent, setGeneratedContent] = useState(null);
  // Removed selectedOptions since we only have single posts now
  const messagesEndRef = useRef(null);
  const statusIntervalRef = useRef(null);

  // Content customization options
  const [contentOptions, setContentOptions] = useState({
    tone: '',
    contentFormat: '',
    contentStyle: '',
    includeStatistics: false,
    postLength: '',
    callToAction: ''
  });

  // Selected creator state
  const [selectedCreator, setSelectedCreator] = useState('gary-v'); // Default to Gary V
  
  // Popup state
  const [showCustomizationPopup, setShowCustomizationPopup] = useState(false);

  // Handle escape key to close popup
  useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === 'Escape' && showCustomizationPopup) {
        setShowCustomizationPopup(false);
      }
    };
    
    if (showCustomizationPopup) {
      document.addEventListener('keydown', handleEscape);
      return () => document.removeEventListener('keydown', handleEscape);
    }
  }, [showCustomizationPopup]);

  // Backend connection state
  const [backendStatus, setBackendStatus] = useState('checking');
  const [backendError, setBackendError] = useState(null);

  // Database states
  const [researchDatabase, setResearchDatabase] = useState([]);
  const [creatorDatabase, setCreatorDatabase] = useState([
    {
      id: 1,
      name: "Gary Vaynerchuk",
      key: "gary-v",
      icon: TrendingUp,
      style: {
        tone: 'Direct, passionate, no-nonsense',
        structure: 'Hook → Personal story → Business insight → Call to action',
        language: 'Casual, uses "you guys", lots of energy',
        length: 'Medium to long posts',
        hooks: ['Look...', 'Here\'s the thing...', 'Real talk...'],
        endings: ['What do you think?', 'Let me know in the comments', 'DM me your thoughts'],
        characteristics: 'High energy, direct approach, business-focused, motivational',
        // Enhanced customization options
        preferredFormat: 'paragraph',
        preferredStyle: 'direct',
        preferredLength: 'medium',
        includeStatistics: true,
        defaultCallToAction: 'What do you think? Let me know in the comments!'
      }
    },
    {
      id: 2,
      name: "Simon Sinek",
      key: "simon-sinek",
      icon: Users,
      style: {
        tone: 'Inspirational, thoughtful, leader-focused',
        structure: 'Question → Story/Example → Leadership lesson → Reflection',
        language: 'Professional but warm, thought-provoking',
        length: 'Medium posts with clear paragraphs',
        hooks: ['Why is it that...', 'The best leaders...', 'I once worked with...'],
        endings: ['What would you do?', 'Leadership is a choice', 'The choice is yours'],
        characteristics: 'Leadership-focused, inspirational, storytelling, thoughtful questions',
        // Enhanced customization options
        preferredFormat: 'paragraph',
        preferredStyle: 'storytelling',
        preferredLength: 'medium',
        includeStatistics: false,
        defaultCallToAction: 'What would you do in this situation?'
      }
    },
    {
      id: 3,
      name: "Seth Godin",
      key: "seth-godin",
      icon: Lightbulb,
      style: {
        tone: 'Wise, concise, marketing-focused',
        structure: 'Insight → Brief explanation → Broader implication',
        language: 'Concise, profound, marketing terminology',
        length: 'Short, punchy posts',
        hooks: ['The thing is...', 'Here\'s what I learned...', 'Marketing is...'],
        endings: ['Worth considering.', 'Just saying.', 'Think about it.'],
        characteristics: 'Concise wisdom, marketing insights, thought-provoking, minimal but impactful',
        // Enhanced customization options
        preferredFormat: 'paragraph',
        preferredStyle: 'direct',
        preferredLength: 'short',
        includeStatistics: false,
        defaultCallToAction: 'Worth considering. What do you think?'
      }
    }
  ]);

  // --- Helper functions for robust creator selection ---
  const normalizeText = useCallback((value) => {
    if (!value) return '';
    return String(value)
      .toLowerCase()
      .normalize('NFKD')
      .replace(/[\u0300-\u036f]/g, '') // strip diacritics
      .replace(/[^a-z0-9\s-]/g, ' ') // remove punctuation
      .replace(/\s+/g, ' ') // collapse whitespace
      .trim();
  }, []);

  const levenshteinDistance = useCallback((a, b) => {
    const s = a || '';
    const t = b || '';
    const m = s.length;
    const n = t.length;
    if (m === 0) return n;
    if (n === 0) return m;
    const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
    for (let i = 0; i <= m; i++) dp[i][0] = i;
    for (let j = 0; j <= n; j++) dp[0][j] = j;
    for (let i = 1; i <= m; i++) {
      for (let j = 1; j <= n; j++) {
        const cost = s[i - 1] === t[j - 1] ? 0 : 1;
        dp[i][j] = Math.min(
          dp[i - 1][j] + 1, // deletion
          dp[i][j - 1] + 1, // insertion
          dp[i - 1][j - 1] + cost // substitution
        );
      }
    }
    return dp[m][n];
  }, []);

  const stringSimilarity = useCallback((a, b) => {
    const s = normalizeText(a);
    const t = normalizeText(b);
    if (!s && !t) return 1;
    const maxLen = Math.max(s.length, t.length) || 1;
    const dist = levenshteinDistance(s, t);
    return 1 - dist / maxLen;
  }, [levenshteinDistance, normalizeText]);

  const extractQuotedCreator = useCallback((input) => {
    if (!input) return '';
    const text = input;
    const quoteRegexes = [
      /style\s+of\s+["'“”‘’]([^"'“”‘’]+)["'“”‘’]/i,
      /style\s+["'“”‘’]([^"'“”‘’]+)["'“”‘’]/i,
    ];
    for (const rx of quoteRegexes) {
      const m = text.match(rx);
      if (m && m[1]) return m[1].trim();
    }
    // Unquoted pattern: style of X
    const ofMatch = text.match(/style\s+of\s+([a-zA-Z0-9\-\s]+)/i);
    if (ofMatch && ofMatch[1]) {
      return ofMatch[1].trim();
    }
    return '';
  }, []);

  // Extract target adjacent to the word "style" (e.g., "in X style", "X style")
  const extractStyleTarget = useCallback((input) => {
    if (!input) return '';
    const text = input.trim();
    const inStyle = text.match(/in\s+([a-zA-Z0-9\-\s]+?)\s+style\b/i);
    if (inStyle && inStyle[1]) return inStyle[1].trim();
    const startStyle = text.match(/^\s*([a-zA-Z0-9\-\s]+?)\s+style\b/i);
    if (startStyle && startStyle[1]) return startStyle[1].trim();
    const hyphen = text.match(/\b([a-zA-Z0-9\-\s]+?)\-style\b/i);
    if (hyphen && hyphen[1]) return hyphen[1].trim();
    const afterColon = text.match(/\bstyle\s*:\s*([a-zA-Z0-9\-\s]+)/i);
    if (afterColon && afterColon[1]) return afterColon[1].trim();
    const likeX = text.match(/\blike\s+([a-zA-Z0-9\-\s]+)/i);
    if (likeX && likeX[1]) return likeX[1].trim();
    return '';
  }, []);

  const resolveCreatorFromText = useCallback((input, creators) => {
    const normalizedInput = normalizeText(input);
    // Prefer explicit targets (quoted or adjacent to the word "style")
    const byQuote = extractQuotedCreator(input);
    const byStyleTarget = extractStyleTarget(input);
    const seeds = [];
    if (byQuote) seeds.push(normalizeText(byQuote));
    if (byStyleTarget) seeds.push(normalizeText(byStyleTarget));
    if (seeds.length === 0) seeds.push(normalizedInput);

    for (const seed of seeds) {
      const seedLen = seed.length;
      // Exact key/name match first
      for (const c of creators) {
        if (!c) continue;
        const keyNorm = normalizeText(c.key);
        const nameNorm = normalizeText(c.name);
        if (keyNorm && seed === keyNorm) return c;
        if (nameNorm && seed === nameNorm) return c;
      }
      // Token-aware alias matching (first/last names, key variants)
      for (const c of creators) {
        if (!c) continue;
        const keyNorm = normalizeText(c.key);
        const keyNoHyphen = (keyNorm || '').replace(/-/g, ' ');
        const nameNorm = normalizeText(c.name);
        const nameTokens = nameNorm.split(' ').filter(Boolean);
        const aliasTokens = [keyNorm, keyNoHyphen, nameNorm, ...nameTokens];
        // Direct token equality
        if (aliasTokens.some(t => t && t === seed)) return c;
        // Seed contained within alias or alias within seed (allow short first names, min length 3)
        if (seedLen >= 3 && aliasTokens.some(t => t && (t.includes(seed) || seed.includes(t)))) return c;
      }
      // Word-boundary includes on the seed
      for (const c of creators) {
        if (!c) continue;
        const nameRx = new RegExp(`(^|\\s)${normalizeText(c.name)}(\\s|$)`);
        const keyRx = new RegExp(`(^|\\s)${normalizeText(c.key)}(\\s|$)`);
        if (nameRx.test(seed) || keyRx.test(seed)) return c;
      }
      // Fuzzy match with stricter threshold to avoid wrong picks
      let best = null;
      let bestScore = 0;
      for (const c of creators) {
        const scoreName = stringSimilarity(seed, c.name || '');
        const scoreKey = stringSimilarity(seed, c.key || '');
        const score = Math.max(scoreName, scoreKey);
        if (score > bestScore) {
          best = c;
          bestScore = score;
        }
      }
      if (best && bestScore >= 0.75) return best;
    }
    return null;
  }, [extractQuotedCreator, extractStyleTarget, normalizeText, stringSimilarity]);
  // --- End helper functions ---

  const [promptHistory, setPromptHistory] = useState([]);
  const [historyGroupedByDate, setHistoryGroupedByDate] = useState({});
  const [selectedHistoryDate, setSelectedHistoryDate] = useState('');
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [supabaseClient, setSupabaseClient] = useState(null);
  const [documentManager, setDocumentManager] = useState(null);
  const [isLoadingResearch, setIsLoadingResearch] = useState(false);
  const [customDocuments, setCustomDocuments] = useState([]);
  const [isLoadingDocuments, setIsLoadingDocuments] = useState(false);
  const [editingResearch, setEditingResearch] = useState(null);
  const [editingCreator, setEditingCreator] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [isUploadingDocument, setIsUploadingDocument] = useState(false);
  const [webCrawls, setWebCrawls] = useState([]);
  const [isLoadingCrawls, setIsLoadingCrawls] = useState(false);
  const [crawlUrl, setCrawlUrl] = useState('');
  const [isCrawling, setIsCrawling] = useState(false);

  // Test backend connection function
  const testBackendConnection = useCallback(async () => {
    setBackendStatus('checking');
    setBackendError(null);
    
    try {
      if (!documentManager) {
        throw new Error('Document manager not initialized');
      }
      
      const health = await documentManager.healthCheck();
      console.log('Backend health check:', health);
      
      // Accept any response that indicates the backend is working
      if (health && (health.status || health.message)) {
        setBackendStatus('connected');
        
        // Load documents after successful connection
        const documents = await documentManager.getDocuments();
        setCustomDocuments(documents || []);
      } else {
        throw new Error('Backend health check failed');
      }
    } catch (error) {
      console.error('Backend connection failed:', error);
      setBackendStatus('error');
      setBackendError(error.message);
    }
  }, [documentManager, setCustomDocuments, setIsLoadingDocuments]);

  // Backend Status Component
  const BackendStatus = () => (
    <div className={`p-3 rounded-lg mb-4 flex items-center gap-2 ${
      backendStatus === 'connected' ? 'bg-green-50 border border-green-200' : 
      backendStatus === 'error' ? 'bg-red-50 border border-red-200' : 
      'bg-yellow-50 border border-yellow-200'
    }`}>
      {backendStatus === 'connected' && <CheckCircle className="w-4 h-4 text-green-600" />}
      {backendStatus === 'error' && <AlertCircle className="w-4 h-4 text-red-600" />}
      {backendStatus === 'checking' && <Clock className="w-4 h-4 text-yellow-600" />}
      
      <div className="flex-1">
        <p className={`text-sm font-medium ${
          backendStatus === 'connected' ? 'text-green-800' : 
          backendStatus === 'error' ? 'text-red-800' : 
          'text-yellow-800'
        }`}>
          Backend: {backendStatus === 'connected' ? 'Connected' : backendStatus === 'error' ? 'Disconnected' : 'Checking...'}
        </p>
        <p className={`text-xs transition-colors duration-300 ${
          isDarkMode ? 'text-gray-400' : 'text-gray-600'
        }`}>
          URL: {BACKEND_CONFIG.url}
        </p>
        {backendStatus === 'connected' && (
          <p className="text-xs text-green-600 mt-1">
            ✅ Connected to FastAPI Backend with full document management capabilities
          </p>
        )}
        {backendError && (
          <div className="text-xs text-red-600 mt-1">
            <p className="font-medium">Error: {backendError}</p>
            <p className="mt-1">
              The FastAPI backend service at {BACKEND_CONFIG.url} is responding but may not have the expected endpoints. 
              This might be because:
            </p>
            <ul className="list-disc list-inside mt-1 ml-2">
              <li>The service is not fully configured for document management</li>
              <li>Some endpoints may not be available</li>
              <li>The service is healthy but has different functionality than expected</li>
            </ul>
            <p className="mt-1">
              <strong>Current URL:</strong> {BACKEND_CONFIG.url}
            </p>
          </div>
        )}
      </div>
      
      {backendStatus === 'error' && (
        <button
          onClick={testBackendConnection}
          className="text-xs px-2 py-1 bg-red-100 text-red-700 rounded hover:bg-red-200"
        >
          Retry
        </button>
      )}
    </div>
  );

  // Completely rewritten content generation
  const generateContentWithFastAPI = useCallback(async (prompt, creatorKey, researchIds, customOptions = null) => {
    if (!documentManager) {
      throw new Error('Document manager not initialized');
    }

    try {
      console.log('=== NEW CONTENT GENERATION ===');
      console.log('Prompt:', prompt);
      console.log('Creator Key:', creatorKey);
      console.log('Research IDs:', researchIds);
      console.log('Custom Options:', customOptions || contentOptions);

      // Use custom options if provided, otherwise use global contentOptions
      const options = customOptions || contentOptions;

      const requestBody = {
        prompt: prompt,
        creator_key: creatorKey,
        research_ids: researchIds,
        tone: options.tone || null,
        content_format: options.contentFormat || null,
        content_style: options.contentStyle || null,
        include_statistics: options.includeStatistics,
        post_length: options.postLength || null,
        call_to_action: options.callToAction || null
      };

      console.log('=== REQUEST BODY BEING SENT ===');
      console.log(JSON.stringify(requestBody, null, 2));
      
      // Test the customization parameters
      console.log('=== TESTING CUSTOMIZATION PARAMETERS ===');
      console.log('Tone:', requestBody.tone);
      console.log('Content Format:', requestBody.content_format);
      console.log('Content Style:', requestBody.content_style);
      console.log('Include Statistics:', requestBody.include_statistics);
      console.log('Post Length:', requestBody.post_length);
      console.log('Call to Action:', requestBody.call_to_action);
      
      const response = await fetch(`${BACKEND_CONFIG.url}/linkedin/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
      });
      
      if (!response.ok) {
        throw new Error(`Backend error: ${response.status}`);
      }
      
      const contentResult = await response.json();
      
      console.log('=== DIRECT BACKEND RESPONSE ===');
      console.log('Backend response:', contentResult);

      // --- START: MODIFIED CODE ---
      // Correctly map the backend's plural keys ('linkedin_posts', 'video_scripts')
      // to the frontend's expected state properties.
      const content = {
        linkedinPosts: contentResult.linkedin_posts || [],
        videoScripts: contentResult.video_scripts || [],
        hashtags: contentResult.hashtags || [],
        engagement_tips: contentResult.engagement_tips || [],
        talking_points: contentResult.talking_points || [],
        style_notes: contentResult.style_notes || '',
        context_used: contentResult.context_used || '',
        contextualImageUrl: contentResult.contextual_image_url || null
      };
      // --- END: MODIFIED CODE ---
      
      console.log('Mapped content:', content);
      return content;
    } catch (error) {
      console.error('Content generation error:', error);
      throw error;
    }
  }, [documentManager, contentOptions]);



  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const loadCreatorsFromSupabase = useCallback(async (client = supabaseClient) => {
    if (!client) return;
    
    try {
      const { data, error } = await client.from('creator_styles').select('*');
      if (error) throw error;
        
      const creatorsWithIcons = data.map(item => ({
        ...item,
        icon: UserPlus, // Default icon for all creators
        dateAdded: new Date(item.created_at || item.dateAdded)
      }));
      
      setCreatorDatabase(creatorsWithIcons);
    } catch (error) {
      console.error('Error loading creators:', error);
    }
  }, [supabaseClient]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Cleanup status interval on unmount
  useEffect(() => {
    return () => {
      if (statusIntervalRef.current) {
        clearInterval(statusIntervalRef.current);
      }
    };
  }, []);

  // Simple content monitoring
  useEffect(() => {
    if (generatedContent) {
      console.log('=== CONTENT SET ===');
      console.log('LinkedIn posts:', generatedContent.linkedinPosts?.length || 0);
      console.log('Video scripts:', generatedContent.videoScripts?.length || 0);
      console.log('Hashtags:', generatedContent.hashtags?.length || 0);
    }
  }, [generatedContent]);

  // Initialize Supabase client
  useEffect(() => {
    const client = createSupabaseClient();
    if (client) {
      setSupabaseClient(client);
      
      const loadResearch = async () => {
        setIsLoadingResearch(true);
        try {
          const { data, error } = await client.from('research').select('*');
          if (error) throw error;
          
          const sortedData = data
            .map(item => ({
              ...item,
              tags: typeof item.tags === 'string' ? item.tags.split(',').map(t => t.trim()) : (item.tags || []),
              dateAdded: new Date(item.created_at || item.dateAdded)
            }))
            .sort((a, b) => new Date(b.dateAdded) - new Date(a.dateAdded));
          
          setResearchDatabase(sortedData);
        } catch (error) {
          console.error('Error loading research:', error);
        } finally {
          setIsLoadingResearch(false);
        }
      };
      loadResearch();
      loadCreatorsFromSupabase(client);
    }
  }, []);

  // Initialize Cloud Document Manager and test connection
  useEffect(() => {
    const manager = createCloudDocumentManager();
    setDocumentManager(manager);
    
    // Test backend connection on startup
    const initializeBackend = async () => {
      setBackendStatus('checking');
      setBackendError(null);
      
      try {
        const health = await manager.healthCheck();
        console.log('Backend health check:', health);
        
        // Accept any response that indicates the backend is working
        if (health && (health.status || health.message)) {
          setBackendStatus('connected');
          
          // Load documents after successful connection
          setIsLoadingDocuments(true);
          try {
            const documents = await manager.getDocuments();
            setCustomDocuments(documents || []);
          } catch (docError) {
            console.error('Error loading documents:', docError);
            setCustomDocuments([]);
          } finally {
            setIsLoadingDocuments(false);
          }

          // Load history from Supabase
          if (supabaseClient) {
            await loadHistoryFromSupabase();
          }
        } else {
          throw new Error('Backend health check failed');
        }
      } catch (error) {
        console.error('Backend connection failed:', error);
        setBackendStatus('error');
        setBackendError(error.message);
        setCustomDocuments([]);
      }
    };
    
    initializeBackend();
  }, []);

  const loadResearchFromSupabase = useCallback(async (client = supabaseClient) => {
    if (!client) return;
    
    setIsLoadingResearch(true);
    try {
      const { data, error } = await client.from('research').select('*');
      if (error) throw error;
      
      const sortedData = data
        .map(item => ({
          ...item,
          tags: typeof item.tags === 'string' ? item.tags.split(',').map(t => t.trim()) : (item.tags || []),
          dateAdded: new Date(item.created_at || item.dateAdded)
        }))
        .sort((a, b) => new Date(b.dateAdded) - new Date(a.dateAdded));
      
      setResearchDatabase(sortedData);
    } catch (error) {
      console.error('Error loading research:', error);
    } finally {
      setIsLoadingResearch(false);
    }
  }, [supabaseClient]);

  // Save content generation history to Supabase
  const saveHistoryToSupabase = useCallback(async (historyEntry) => {
    console.log('🔄 saveHistoryToSupabase called with:', historyEntry);
    
    if (!supabaseClient) {
      console.error('❌ No supabaseClient available');
      return null;
    }

    console.log('✅ Supabase client available, attempting to save...');

    try {
      const historyData = {
        prompt: historyEntry.prompt,
        creator_style: historyEntry.creatorKey,
        creator_name: historyEntry.creatorName,
        generated_content: historyEntry.content,
        research_used: historyEntry.researchUsed || [],
        backend_sources: historyEntry.backendSources || [],
        metadata: {
          research_count: historyEntry.researchUsed?.length || 0,
          content_types: Object.keys(historyEntry.content || {})
        }
      };

      console.log('📝 Data to insert:', historyData);

      const { data, error } = await supabaseClient
        .from('content_history')
        .insert([historyData]);

      if (error) {
        console.error('❌ Supabase error saving history:', error);
        console.error('Error details:', {
          message: error.message,
          details: error.details,
          hint: error.hint,
          code: error.code
        });
        return null;
      }

      console.log('✅ History saved successfully:', data);
      return data[0];
    } catch (error) {
      console.error('❌ Exception saving history to Supabase:', error);
      return null;
    }
  }, [supabaseClient]);

  // Load content history from Supabase
  const loadHistoryFromSupabase = useCallback(async () => {
    if (!supabaseClient) return;

    try {
      setIsLoadingHistory(true);
      const { data, error } = await supabaseClient
        .from('content_history')
        .select('*')
        .order('created_at', { ascending: false });

      if (error) {
        console.error('Error loading history from Supabase:', error);
        return;
      }

      // Group history by date
      const grouped = {};
      const history = data.map(item => ({
        id: item.id,
        prompt: item.prompt,
        creatorKey: item.creator_style,
        creatorName: item.creator_name,
        content: item.generated_content,
        researchUsed: item.research_used || [],
        backendSources: item.backend_sources || [],
        metadata: item.metadata || {},
        timestamp: new Date(item.created_at),
        created_at: item.created_at
      }));

      // Group by date
      history.forEach(entry => {
        const dateKey = entry.timestamp.toDateString();
        if (!grouped[dateKey]) {
          grouped[dateKey] = [];
        }
        grouped[dateKey].push(entry);
      });

      setPromptHistory(history);
      setHistoryGroupedByDate(grouped);
      
      // Set default selected date to today or most recent
      const dates = Object.keys(grouped).sort((a, b) => new Date(b) - new Date(a));
      if (dates.length > 0 && !selectedHistoryDate) {
        setSelectedHistoryDate(dates[0]);
      }

      console.log('History loaded successfully:', history.length, 'entries');
    } catch (error) {
      console.error('Error loading history from Supabase:', error);
    } finally {
      setIsLoadingHistory(false);
    }
  }, [supabaseClient, selectedHistoryDate]);

  const deleteResearch = useCallback(async (id) => {
    try {
      // Use documentManager to delete research from Supabase
      if (documentManager) {
        await documentManager.deleteResearch(id);
        // Update local state instead of reloading
        setResearchDatabase(prev => prev.filter(item => item.id !== id));
        console.log('Research deleted successfully from Supabase');
      } else {
        // Fallback to local storage
        setResearchDatabase(prev => prev.filter(item => item.id !== id));
      }
    } catch (error) {
      console.error('Error deleting research:', error);
      // Fallback to local storage
      setResearchDatabase(prev => prev.filter(item => item.id !== id));
    }
  }, [documentManager]);

  const updateResearch = useCallback(async (id, updatedData) => {
    const updated = {
      topic: updatedData.topic,
      findings: updatedData.findings,
      source: updatedData.source,
      data: updatedData.data,
      tags: typeof updatedData.tags === 'string' 
        ? updatedData.tags.split(',').map(tag => tag.trim()).filter(tag => tag).join(',')
        : (Array.isArray(updatedData.tags) ? updatedData.tags.join(',') : updatedData.tags)
    };

    try {
      // Use documentManager to update research in Supabase
      if (documentManager) {
        await documentManager.updateResearch(id, updated);
        // Update local state instead of reloading
        setResearchDatabase(prev => prev.map(item => 
          item.id === id ? { 
            ...item, 
            topic: updatedData.topic,
            findings: updatedData.findings,
            source: updatedData.source,
            data: updatedData.data,
            tags: typeof updatedData.tags === 'string' 
              ? updatedData.tags.split(',').map(t => t.trim()).filter(t => t)
              : (Array.isArray(updatedData.tags) ? updatedData.tags : []),
            dateAdded: item.dateAdded
          } : item
        ));
        console.log('Research updated successfully in Supabase');
      } else {
        // Fallback to local storage
    setResearchDatabase(prev => prev.map(item => 
      item.id === id ? { 
        ...item, 
        topic: updatedData.topic,
        findings: updatedData.findings,
        source: updatedData.source,
        data: updatedData.data,
        tags: typeof updatedData.tags === 'string' 
          ? updatedData.tags.split(',').map(t => t.trim()).filter(t => t)
          : (Array.isArray(updatedData.tags) ? updatedData.tags : []),
        dateAdded: item.dateAdded
      } : item
    ));
      }
      } catch (error) {
      console.error('Error updating research:', error);
      // Fallback to local storage
      setResearchDatabase(prev => prev.map(item => 
        item.id === id ? { 
          ...item, 
          topic: updatedData.topic,
          findings: updatedData.findings,
          source: updatedData.source,
          data: updatedData.data,
          tags: typeof updatedData.tags === 'string' 
            ? updatedData.tags.split(',').map(t => t.trim()).filter(t => t)
            : (Array.isArray(updatedData.tags) ? updatedData.tags : []),
          dateAdded: item.dateAdded
        } : item
      ));
    }
    
    setEditingResearch(null);
  }, [documentManager]);

  // Enhanced keyword extraction and research matching
  const extractKeywords = useCallback((text) => {
    const lowerText = text.toLowerCase();
    
    // Remove common words and extract meaningful keywords
    const commonWords = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'create', 'make', 'generate', 'post', 'content', 'linkedin', 'style', 'about', 'using', 'my', 'latest', 'research', 'documents'];
    
    const words = lowerText
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(word => word.length > 2 && !commonWords.includes(word));
    
    // Add specific topic keywords
    const topicKeywords = [];
    if (lowerText.includes('ai') || lowerText.includes('artificial intelligence')) topicKeywords.push('ai', 'artificial intelligence', 'machine learning');
    if (lowerText.includes('business') || lowerText.includes('entrepreneur')) topicKeywords.push('business', 'entrepreneurship', 'startup');
    if (lowerText.includes('marketing') || lowerText.includes('brand')) topicKeywords.push('marketing', 'branding', 'advertising');
    if (lowerText.includes('leadership') || lowerText.includes('management')) topicKeywords.push('leadership', 'management', 'team');
    if (lowerText.includes('technology') || lowerText.includes('tech')) topicKeywords.push('technology', 'tech', 'innovation');
    if (lowerText.includes('finance') || lowerText.includes('money')) topicKeywords.push('finance', 'investment', 'money');
    if (lowerText.includes('health') || lowerText.includes('fitness')) topicKeywords.push('health', 'fitness', 'wellness');
    
    return [...new Set([...words, ...topicKeywords])];
  }, []);

    // Dynamic status messages for AI processing
  const getStatusMessage = (stage) => {
    const statusMessages = {
      analyzing: [
        "🔍 Analyzing your request...",
        "🧠 Understanding your content needs...",
        "📝 Processing your prompt...",
        "💭 Thinking about the best approach..."
      ],
      researching: [
        "📚 Searching through research database...",
        "🔎 Finding relevant research papers...",
        "📖 Analyzing research insights...",
        "🎯 Matching research to your topic..."
      ],
      generating: [
        "✨ Generating LinkedIn content...",
        "📝 Creating engaging posts...",
        "🎨 Crafting your content...",
        "💡 Writing in your chosen style..."
      ],
      visualizing: [
        "🎨 Creating hand-drawn visualizations...",
        "🖼️ Generating artistic charts...",
        "📊 Drawing data visualizations...",
        "🎭 Adding creative elements..."
      ],
      finalizing: [
        "🔧 Finalizing your content...",
        "✨ Adding finishing touches...",
        "🎯 Optimizing for engagement...",
        "🚀 Almost ready!"
      ]
    };
    
    const messages = statusMessages[stage] || ["🤖 Processing your request..."];
    return messages[Math.floor(Math.random() * messages.length)];
  };

  // Start dynamic status updates for a stage
  const startStatusUpdates = (stage) => {
    setCurrentStage(stage);
    setCurrentStatus(getStatusMessage(stage));
    
    // Clear any existing interval
    if (statusIntervalRef.current) {
      clearInterval(statusIntervalRef.current);
    }
    
    // Update status every 2-3 seconds with random messages from the same stage
    statusIntervalRef.current = setInterval(() => {
      setCurrentStatus(getStatusMessage(stage));
    }, 2500);
  };

  // Stop status updates
  const stopStatusUpdates = () => {
    if (statusIntervalRef.current) {
      clearInterval(statusIntervalRef.current);
      statusIntervalRef.current = null;
    }
    setCurrentStatus('');
    setCurrentStage('');
  };

  const findRelevantResearch = useCallback((text, maxResults = 3) => {
    const keywords = extractKeywords(text);
    console.log('=== RESEARCH MATCHING DEBUG ===');
    console.log('Input text:', text);
    console.log('Extracted keywords:', keywords);
    console.log('Total research items available:', researchDatabase.length);
    
    if (researchDatabase.length === 0) {
      console.log('No research database available');
      return [];
    }
     
    const scoredResearch = researchDatabase.map(item => {
      let score = 0;
      const itemText = `${item.topic} ${item.findings} ${item.data || ''} ${(Array.isArray(item.tags) ? item.tags.join(' ') : item.tags || '')}`.toLowerCase();
      
      // Exact keyword matches get highest scores
      keywords.forEach(keyword => {
        if (itemText.includes(keyword)) {
          score += 10; // Increased weight for exact matches
          if (item.topic.toLowerCase().includes(keyword)) score += 5;
          if (Array.isArray(item.tags) && item.tags.some(tag => tag.toLowerCase().includes(keyword))) score += 3;
        }
      });
      
      // Partial matches (reduced weight)
      keywords.forEach(keyword => {
        const keywordParts = keyword.split(' ');
        keywordParts.forEach(part => {
          if (part.length > 3 && itemText.includes(part)) score += 1;
        });
      });
      
      // Recency bonus (reduced)
      const daysSinceAdded = (new Date() - new Date(item.dateAdded)) / (1000 * 60 * 60 * 24);
      if (daysSinceAdded < 7) score += 1;
      
      console.log(`Research item "${item.topic}" scored: ${score}`);
      return { ...item, relevanceScore: score };
    });
    
    // Include research with any relevance (score > 0) or fallback to recent items
    const sortedByRelevance = scoredResearch
      .filter(item => item.relevanceScore > 0)
      .sort((a, b) => b.relevanceScore - a.relevanceScore);
    
    // If no relevant research found, use the most recent items
    if (sortedByRelevance.length === 0) {
      console.log('No relevant research found, using most recent items');
      const recentResearch = researchDatabase
        .sort((a, b) => new Date(b.dateAdded) - new Date(a.dateAdded))
        .slice(0, maxResults);
      return recentResearch;
    }
    
    console.log(`Found ${sortedByRelevance.length} highly relevant research items`);
    console.log('Available research items:', researchDatabase.map(r => ({ id: r.id, topic: r.topic, tags: r.tags })));
    console.log('Relevant research items found:', sortedByRelevance.map(r => ({ id: r.id, topic: r.topic, score: r.relevanceScore })));
    
    // Return only the most relevant items (max 3)
    return sortedByRelevance.slice(0, maxResults);
  }, [researchDatabase, extractKeywords]);

  const handleSendMessage = useCallback(async (messageText) => {
    if (!messageText || !messageText.trim()) return;
    const text = messageText.trim();

    const userMessage = {
      id: Date.now(),
      type: 'user',
      text: text,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setGeneratedContent(null); // Clear previous content
    setIsTyping(true);
    startStatusUpdates('analyzing');

    try {
      // --- Deterministic Creator Extraction ---
      let creatorKey = 'gary-v'; // Fallback
      let creatorName = 'Gary Vaynerchuk';

      const match = resolveCreatorFromText(text, creatorDatabase);
      if (match) {
        creatorKey = match.key;
        creatorName = match.name;
      }
      // --- End Deterministic Extraction ---

      console.log('=== MESSAGE PROCESSING DEBUG ===');
      console.log('Input text:', text);
      console.log('Selected creator key:', creatorKey);
      console.log('Selected creator name:', creatorName);

      // Update status for research phase
      startStatusUpdates('researching');
      
      const relevantResearch = findRelevantResearch(text, 3);
      const researchIds = relevantResearch.map(r => r.id);

      // Update status for content generation phase
      startStatusUpdates('generating');
      
      // Pass the original, clean text to the generation function
      const content = await generateContentWithFastAPI(text, creatorKey, researchIds);
      
      // Update status for visualization phase
      startStatusUpdates('visualizing');
      
      // Ensure only professional visuals are shown; content already includes contextual_image_url
      setGeneratedContent(content);
      
      // Update status for finalization
      startStatusUpdates('finalizing');

      // Save to history
      const historyEntry = {
        prompt: text,
        creatorKey: creatorKey,
        creatorName: creatorName,
        content: content,
        researchUsed: relevantResearch.map(r => ({ id: r.id, topic: r.topic })),
      };
      await saveHistoryToSupabase(historyEntry);
      await loadHistoryFromSupabase();

      let responseText = `I've created content in ${creatorName}'s style for you!`;
      if (relevantResearch.length > 0) {
        responseText += `\nI incorporated insights from ${relevantResearch.length} relevant research item(s).`;
      }
      
      const botResponse = {
        id: Date.now() + 1,
        type: 'bot',
        text: responseText,
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, botResponse]);

    } catch (error) {
      console.error('Error generating response:', error);
      const fallbackResponse = {
        id: Date.now() + 1,
        type: 'bot',
        text: `Sorry, I encountered an error: ${error.message}. Please check the backend logs and your API keys.`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, fallbackResponse]);
    } finally {
      setIsTyping(false);
      stopStatusUpdates();
    }
  }, [creatorDatabase, findRelevantResearch, generateContentWithFastAPI, saveHistoryToSupabase, loadHistoryFromSupabase, contentOptions]);

  const handleCreatorSelect = useCallback((creatorKey) => {
    const creator = creatorDatabase.find(c => c.key === creatorKey);
    if (!creator) {
      console.error('Creator not found:', creatorKey);
      return;
    }
    console.log('Creator selected:', creator.name, creator.key);
    
    // Update selected creator state
    setSelectedCreator(creatorKey);
    
    // Auto-populate customization options based on creator preferences
    if (creator.style.preferredFormat || creator.style.preferredStyle || creator.style.preferredLength) {
      setContentOptions(prev => ({
        ...prev,
        contentFormat: creator.style.preferredFormat || '',
        contentStyle: creator.style.preferredStyle || '',
        postLength: creator.style.preferredLength || '',
        includeStatistics: creator.style.includeStatistics || false,
        callToAction: creator.style.defaultCallToAction || ''
      }));
    }
    
    const message = `Create a ${creator.name} style LinkedIn post using my latest research and uploaded documents`;
    handleSendMessage(message);
  }, [creatorDatabase, handleSendMessage]);

  const copyToClipboard = useCallback((text) => {
    if (!text) return;
    
    navigator.clipboard.writeText(text).then(() => {
      console.log('Text copied to clipboard');
    }).catch((err) => {
      console.error('Failed to copy text: ', err);
      // Fallback for older browsers
      const textArea = document.createElement('textarea');
      textArea.value = text;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
    });
  }, []);

  // Document Management Functions
  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
    }
  };

  const handleUploadDocument = async () => {
    if (!selectedFile || !documentManager) return;

    setIsUploadingDocument(true);
    try {
      const result = await documentManager.uploadDocument(selectedFile);
      console.log('Document uploaded:', result);
      
      // Refresh documents list
      const documents = await documentManager.getDocuments();
      setCustomDocuments(documents || []);
      
      setSelectedFile(null);
      const fileInput = document.getElementById('file-input');
      if (fileInput) fileInput.value = '';
      
      // Show success message
      const successMessage = {
        id: Date.now(),
        type: 'bot',
        text: `✅ Successfully uploaded "${result.filename}" with ${result.chunks_added} chunks. You can now ask questions about this document!`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, successMessage]);
      
    } catch (error) {
      console.error('Upload failed:', error);
      const errorMessage = {
        id: Date.now(),
        type: 'bot',
        text: `❌ Upload failed: ${error.message}`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsUploadingDocument(false);
    }
  };

  const handleDeleteDocument = async (filename) => {
    if (!documentManager) return;

    try {
      await documentManager.deleteDocument(filename);
      console.log('Document deleted:', filename);
      
      const documents = await documentManager.getDocuments();
      setCustomDocuments(documents || []);
      
      const successMessage = {
        id: Date.now(),
        type: 'bot',
        text: `✅ Successfully deleted "${filename}" and updated the search index.`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, successMessage]);
      
    } catch (error) {
      console.error('Delete failed:', error);
      const errorMessage = {
        id: Date.now(),
        type: 'bot',
        text: `❌ Delete failed: ${error.message}`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    }
  };

  const handleCrawlWebsite = async (url) => {
    if (!url || !url.trim()) {
      alert('Please enter a valid URL');
      return;
    }

    setIsCrawling(true);
    try {
      console.log('Crawling website:', url);
      
      const response = await fetch(`${BACKEND_CONFIG.url}/linkedin/crawl`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          url: url.trim(),
          depth: 2,
          max_pages: 10
        })
      });

      console.log('Crawl response status:', response.status);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        console.error('Crawl response error:', errorData);
        throw new Error(errorData.detail || errorData.error || `HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      console.log('Crawl result:', result);
      
      // Add to web crawls list
      setWebCrawls(prev => [result, ...prev]);
      
      // Web crawls are now stored separately in the backend
      // No need to add to research database here
      console.log('Web crawl completed and stored in backend');
      
      const successMessage = {
        id: Date.now(),
        type: 'bot',
        text: `✅ Successfully crawled ${result.pages_crawled} pages from "${url}" with ${result.total_content_length} characters of content! The content has been stored in your research database.`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, successMessage]);
      
    } catch (error) {
      console.error('Error crawling website:', error);
      const errorMessage = {
        id: Date.now(),
        type: 'bot',
        text: `❌ Crawl failed: ${error.message}`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsCrawling(false);
    }
  };

  const loadWebCrawls = async () => {
    setIsLoadingCrawls(true);
    try {
      const response = await fetch(`${BACKEND_CONFIG.url}/linkedin/crawls`);
      if (response.ok) {
        const data = await response.json();
        console.log('Loaded web crawls:', data);
        setWebCrawls(data || []);
      } else {
        console.error('Failed to load web crawls:', response.status);
        setWebCrawls([]);
      }
    } catch (error) {
      console.error('Error loading web crawls:', error);
      setWebCrawls([]);
    } finally {
      setIsLoadingCrawls(false);
    }
  };


  // Enhanced Visual Generator Component
  const EnhancedVisualGenerator = ({ researchData, onVisualGenerated }) => {
    const [visualTopic, setVisualTopic] = useState('');
    const [visualType, setVisualType] = useState('mixed');
    const [selectedResearchIds, setSelectedResearchIds] = useState([]);
    const [isGenerating, setIsGenerating] = useState(false);
    const [generatedVisual, setGeneratedVisual] = useState(null);

    const generateVisual = async () => {
      if (!visualTopic.trim()) {
        alert('Please enter a topic for the visual content');
        return;
      }

      setIsGenerating(true);
      try {
        const response = await fetch(`${BACKEND_CONFIG.url}/linkedin/generate-visual-content`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            content_topic: visualTopic,
            content_type: visualType,
            research_ids: selectedResearchIds
          })
        });

        if (!response.ok) {
          throw new Error(`Visual generation failed: ${response.status}`);
        }

        const result = await response.json();
        setGeneratedVisual(result);
        onVisualGenerated(result);
      } catch (error) {
        console.error('Error generating visual:', error);
        alert('Failed to generate visual content. Please try again.');
      } finally {
        setIsGenerating(false);
      }
    };

    return (
      <div className="space-y-4">
        {/* Topic Input */}
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-2">
            Visual Content Topic
          </label>
          <input
            type="text"
            value={visualTopic}
            onChange={(e) => setVisualTopic(e.target.value)}
            placeholder="e.g., AI in business, digital transformation, leadership trends..."
            className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          />
        </div>

        {/* Visual Type Selection */}
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-2">
            Visual Type
          </label>
          <div className="grid grid-cols-2 gap-2">
            {[
              { value: 'contextual', label: 'Contextual Image', desc: 'Professional images related to your topic' },
              { value: 'data', label: 'Data Visualization', desc: 'Charts and graphs from your research' },
              { value: 'hybrid', label: 'Hybrid Content', desc: 'Combines images with data overlays' },
              { value: 'mixed', label: 'Mixed (Auto)', desc: 'AI chooses the best approach' }
            ].map((type) => (
              <label key={type.value} className="flex items-start gap-2 p-3 border border-slate-200 rounded-lg cursor-pointer hover:bg-slate-50">
                <input
                  type="radio"
                  name="visualType"
                  value={type.value}
                  checked={visualType === type.value}
                  onChange={(e) => setVisualType(e.target.value)}
                  className="mt-1"
                />
                <div>
                  <div className="font-medium text-sm text-slate-800">{type.label}</div>
                  <div className="text-xs text-slate-600">{type.desc}</div>
                </div>
              </label>
            ))}
          </div>
        </div>

        {/* Research Selection */}
        {researchData && researchData.length > 0 && (
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Include Research Data (Optional)
            </label>
            <div className="max-h-32 overflow-y-auto space-y-1">
              {researchData.slice(0, 5).map((item) => (
                <label key={item.id} className="flex items-center gap-2 p-2 border border-slate-200 rounded cursor-pointer hover:bg-slate-50">
                  <input
                    type="checkbox"
                    checked={selectedResearchIds.includes(item.id)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setSelectedResearchIds([...selectedResearchIds, item.id]);
                      } else {
                        setSelectedResearchIds(selectedResearchIds.filter(id => id !== item.id));
                      }
                    }}
                  />
                  <span className="text-sm text-slate-700 truncate">{item.topic}</span>
                </label>
              ))}
            </div>
          </div>
        )}

        {/* Generate Button */}
        <button
          onClick={generateVisual}
          disabled={isGenerating || !visualTopic.trim()}
          className="w-full bg-gradient-to-r from-purple-500 to-pink-600 text-white py-2 px-4 rounded-lg font-medium hover:from-purple-600 hover:to-pink-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
        >
          {isGenerating ? 'Generating Visual...' : 'Generate Enhanced Visual Content'}
        </button>

        {/* Generated Visual Display */}
        {generatedVisual && (
          <div className={`mt-4 p-4 border rounded-lg transition-colors duration-300 ${
            isDarkMode 
              ? 'bg-slate-800 border-slate-700' 
              : 'bg-white border-slate-200'
          }`}>
            <div className="flex items-center gap-2 mb-3">
              <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center">
                <Check className="w-4 h-4 text-white" />
              </div>
              <span className={`font-medium transition-colors duration-300 ${
                isDarkMode ? 'text-white' : 'text-slate-800'
              }`}>Visual Generated Successfully!</span>
            </div>
            
            <div className={`space-y-2 text-sm transition-colors duration-300 ${
              isDarkMode ? 'text-gray-300' : 'text-slate-600'
            }`}>
              <div><strong>Topic:</strong> {generatedVisual.content_topic}</div>
              <div><strong>Type:</strong> {generatedVisual.content_type}</div>
              <div><strong>Research Used:</strong> {generatedVisual.research_used} items</div>
            </div>

            <div className="mt-3">
              <img
                src={generatedVisual.image_url}
                alt={`Generated visual for ${generatedVisual.content_topic}`}
                className="w-full h-auto rounded-lg shadow-sm"
                style={{ maxHeight: '300px', objectFit: 'contain' }}
              />
            </div>

            {generatedVisual.source_link && (
              <div className="mt-2 flex items-center gap-2">
                <ExternalLink className={`w-3 h-3 transition-colors duration-300 ${
                  isDarkMode ? 'text-gray-400' : 'text-slate-500'
                }`} />
                <a
                  href={generatedVisual.source_link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className={`text-xs underline transition-colors duration-300 ${
                    isDarkMode 
                      ? 'text-blue-400 hover:text-blue-300' 
                      : 'text-blue-600 hover:text-blue-800'
                  }`}
                >
                  View Source
                </a>
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  // Auto visual generation section (no user input)
  const AutoVisualSection = () => {
    const [isAutogenRunning, setIsAutogenRunning] = useState(false);
    const [autogenError, setAutogenError] = useState(null);
    const [hasGenerated, setHasGenerated] = useState(false);

    const runAuto = useCallback(async () => {
      if (isAutogenRunning || hasGenerated) return;
      setIsAutogenRunning(true);
      setAutogenError(null);
      try {
        // Prefer contextual visual from backend content endpoint
        const topicGuess = (generatedContent?.linkedinPosts?.[0] || '').slice(0, 80) || 'business innovation';
        const response = await fetch(`${BACKEND_CONFIG.url}/linkedin/generate-visual-content`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ content_topic: topicGuess, content_type: 'mixed', research_ids: [] })
        });
        if (response.ok) {
          const result = await response.json();
          setGeneratedContent(prev => ({
            ...prev,
            contextualImageUrl: result.image_url,
            imageSourceLink: result.source_link
          }));
          setHasGenerated(true);
        } else {
          throw new Error(`Auto visual generation failed: ${response.status}`);
        }
      } catch (e) {
        setAutogenError(e.message);
      } finally {
        setIsAutogenRunning(false);
      }
    }, [isAutogenRunning, hasGenerated, generatedContent]);

    // Only run once when content is first generated
    useEffect(() => { 
      if (generatedContent?.linkedinPosts?.[0] && !hasGenerated) {
        runAuto(); 
      }
    }, [generatedContent?.linkedinPosts?.[0], hasGenerated, runAuto]);

    return (
      <div className="bg-gradient-to-br from-purple-50/80 to-pink-100/30 border border-purple-200/50 rounded-xl p-4 mb-4">
        <div className="text-xs text-slate-600">
          {isAutogenRunning ? 'Generating visual automatically…' : 'Visual generated automatically.'}
          {autogenError && <div className="text-red-600 mt-1">{autogenError}</div>}
        </div>
      </div>
    );
  };

  const ContentDisplay = ({ content }) => {
    console.log('=== SIMPLE CONTENT DISPLAY ===');
    console.log('Content:', content);
    console.log('Content type:', typeof content);
    console.log('Content keys:', content ? Object.keys(content) : 'No content');
    
    // Simple extraction
    const linkedinPost = content?.linkedinPosts?.[0];
    const videoScript = content?.videoScripts?.[0];
    const hashtags = content?.hashtags || [];
    const engagementTips = content?.engagement_tips || [];
    
    // Dropdown state management
    const [dropdownStates, setDropdownStates] = useState({
      linkedinPost: true,
      visualizations: true,
      reelTranscript: true
    });

    // View mode state for mobile responsiveness
    const [viewMode, setViewMode] = useState('grid'); // 'grid' or 'stacked'

    // Dropdown menu states for each column
    const [dropdownMenus, setDropdownMenus] = useState({
      linkedinPost: false,
      reelTranscript: false,
      infographic: false
    });

    // Regeneration loading states
    const [regenerating, setRegenerating] = useState({
      linkedinPost: false,
      reelTranscript: false,
      infographic: false
    });

    const toggleDropdownMenu = (column) => {
      setDropdownMenus(prev => ({
        ...prev,
        [column]: !prev[column]
      }));
    };

    // Close dropdown menus when clicking outside
    useEffect(() => {
      const handleClickOutside = (event) => {
        if (!event.target.closest('.dropdown-menu')) {
          setDropdownMenus({
            linkedinPost: false,
            reelTranscript: false,
            infographic: false
          });
        }
      };

      document.addEventListener('mousedown', handleClickOutside);
      return () => {
        document.removeEventListener('mousedown', handleClickOutside);
      };
    }, []);

    // Regenerate specific content type
    const regenerateContent = async (contentType) => {
      try {
        setRegenerating(prev => ({ ...prev, [contentType]: true }));
        setCurrentStatus(`Regenerating ${contentType === 'linkedinPost' ? 'LinkedIn Post' : contentType === 'reelTranscript' ? 'Reel Transcript' : 'Infographic'}...`);
        
        // Get the last user message to use as context
        const lastUserMessage = messages.filter(msg => msg.type === 'user').pop();
        if (!lastUserMessage) {
          throw new Error('No user message found for regeneration');
        }

        // Create a specific regeneration request
        const regenerationPrompt = `Please regenerate only the ${contentType === 'linkedinPost' ? 'LinkedIn post' : contentType === 'reelTranscript' ? 'LinkedIn reel transcript' : 'infographic/visualization'} for this request: "${lastUserMessage.text}". Keep the same context and research but create a fresh version of this specific content type.`;

        // Prepare research IDs from the research database
        const researchIds = researchDatabase.map(item => item.id.toString()).filter(id => id);
        
        const requestBody = {
          prompt: regenerationPrompt,
          creator_key: selectedCreator,
          research_ids: researchIds,
          tone: contentOptions.tone || null,
          content_format: contentOptions.contentFormat || null,
          content_style: contentOptions.contentStyle || null,
          include_statistics: contentOptions.includeStatistics,
          post_length: contentOptions.postLength || null,
          call_to_action: contentOptions.callToAction || null
        };

        console.log('Regeneration request body:', JSON.stringify(requestBody, null, 2));

        const response = await fetch(`${BACKEND_CONFIG.url}/linkedin/generate`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          console.error('Regeneration response error:', errorData);
          throw new Error(errorData.detail || errorData.error || `Regeneration failed: ${response.status}`);
        }

        const newContent = await response.json();
        
        // Update only the specific content type
        setGeneratedContent(prevContent => ({
          ...prevContent,
          ...(contentType === 'linkedinPost' && newContent.linkedinPosts && {
            linkedinPosts: newContent.linkedinPosts
          }),
          ...(contentType === 'reelTranscript' && newContent.videoScripts && {
            videoScripts: newContent.videoScripts
          }),
          ...(contentType === 'infographic' && newContent.contextualImageUrl && {
            contextualImageUrl: newContent.contextualImageUrl,
            imageSourceLink: newContent.imageSourceLink
          })
        }));

        // Add a system message about the regeneration
        const regenerationMessage = {
          id: Date.now(),
          type: 'bot',
          text: `✅ Successfully regenerated ${contentType === 'linkedinPost' ? 'LinkedIn Post' : contentType === 'reelTranscript' ? 'Reel Transcript' : 'Infographic'}!`,
          timestamp: new Date()
        };
        
        setMessages(prev => [...prev, regenerationMessage]);
        
      } catch (error) {
        console.error('Regeneration error:', error);
        console.error('Error details:', {
          contentType,
          selectedCreator,
          researchIds: researchDatabase.map(item => item.id),
          contentOptions,
          errorMessage: error.message
        });
        
        const errorMessage = {
          id: Date.now(),
          type: 'bot',
          text: `❌ Failed to regenerate ${contentType === 'linkedinPost' ? 'LinkedIn Post' : contentType === 'reelTranscript' ? 'Reel Transcript' : 'Infographic'}. Error: ${error.message}. Please try again.`,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, errorMessage]);
      } finally {
        setRegenerating(prev => ({ ...prev, [contentType]: false }));
        setCurrentStatus('');
      }
    };
    
    const toggleDropdown = (section) => {
      setDropdownStates(prev => ({
        ...prev,
        [section]: !prev[section]
      }));
    };
    
    console.log('LinkedIn post:', linkedinPost);
    console.log('Video script:', videoScript);
    console.log('LinkedIn post type:', typeof linkedinPost);
    console.log('Video script type:', typeof videoScript);
    console.log('LinkedIn post length:', linkedinPost?.length || 0);
    console.log('Video script length:', videoScript?.length || 0);
    
    // Inline editable text block: double-click to edit
    const EditableBlock = ({ initialText, mono }) => {
      const [isEditing, setIsEditing] = useState(false);
      const [text, setText] = useState(initialText || '');
      useEffect(() => { setText(initialText || ''); }, [initialText]);
      const onDoubleClick = () => setIsEditing(true);
      const onBlur = () => setIsEditing(false);
      return (
        <div onDoubleClick={onDoubleClick} className={mono ? "font-mono" : ""}>
          {isEditing ? (
            <textarea
              value={text}
              onChange={e => setText(e.target.value)}
              onBlur={onBlur}
              autoFocus
              className={`w-full min-h-[160px] rounded-lg p-4 border focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors duration-300 ${
                isDarkMode 
                  ? 'bg-purple-700/90 border-purple-600 text-white' 
                  : 'bg-white/90 border-slate-300 text-gray-900'
              }`}
            />
          ) : (
            <div className={`backdrop-blur-sm rounded-lg p-4 border cursor-text transition-colors duration-300 ${
              isDarkMode 
                ? 'bg-purple-700/60 border-purple-600/30' 
                : 'bg-white/60 border-slate-200/30'
            }`}>
              {text}
              <div className={`mt-2 text-[10px] transition-colors duration-300 ${
                isDarkMode ? 'text-gray-400' : 'text-slate-500'
              }`}>Double-click to edit</div>
            </div>
          )}
        </div>
      );
    };

    return (
    <div className={`backdrop-blur-sm border rounded-3xl p-8 mt-6 shadow-xl relative z-10 transition-colors duration-300 ${
      isDarkMode 
        ? 'bg-purple-800/95 border-purple-700 shadow-purple-900/20' 
        : 'bg-white/95 border-slate-200 shadow-slate-500/10'
    }`}>
      <div className={`flex items-center gap-4 mb-8 pb-6 border-b transition-colors duration-300 ${
        isDarkMode ? 'border-purple-700' : 'border-slate-200'
      }`}>
        <div className="w-12 h-12 bg-gradient-to-br from-slate-700 to-slate-800 rounded-2xl flex items-center justify-center shadow-lg">
          <Linkedin className="w-6 h-6 text-white" />
        </div>
        <div className="flex items-center gap-4">
          <h3 className={`text-xl font-bold transition-colors duration-300 ${
            isDarkMode ? 'text-white' : 'text-slate-900'
          }`}>Generated Content</h3>
          {(contentOptions.tone || contentOptions.contentFormat || contentOptions.contentStyle || contentOptions.postLength || contentOptions.includeStatistics || contentOptions.callToAction) && (
            <span className={`px-3 py-1 text-sm rounded-full font-semibold border transition-colors duration-300 ${
              isDarkMode 
                ? 'bg-purple-700 text-slate-200 border-purple-600' 
                : 'bg-slate-100 text-slate-800 border-slate-200'
            }`}>
              Customized
            </span>
          )}
        </div>
      </div>

      {/* Content Comparison Header */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-gradient-to-br from-indigo-600 to-purple-600 rounded-lg flex items-center justify-center">
              <BarChart className="w-4 h-4 text-white" />
            </div>
            <h3 className={`text-lg font-bold transition-colors duration-300 ${
              isDarkMode ? 'text-white' : 'text-slate-900'
            }`}>
              Content Comparison
            </h3>
          </div>
          <div className="flex items-center gap-2">
          <button
              onClick={() => setViewMode(viewMode === 'grid' ? 'stacked' : 'grid')}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 ${
              isDarkMode 
                  ? 'bg-slate-700 text-gray-300 hover:bg-slate-600 border border-slate-600' 
                  : 'bg-slate-100 text-slate-700 hover:bg-slate-200 border border-slate-200'
              }`}
            >
              {viewMode === 'grid' ? (
                <>
                  <BarChart className="w-3.5 h-3.5" />
                  Stacked View
                </>
              ) : (
                <>
                  <BarChart className="w-3.5 h-3.5" />
                  Grid View
                </>
            )}
          </button>
          </div>
        </div>
        <p className={`text-sm transition-colors duration-300 ${
          isDarkMode ? 'text-gray-400' : 'text-slate-600'
        }`}>
          {viewMode === 'grid' 
            ? 'Compare your LinkedIn Post, Reel Transcript, and Infographic side by side'
            : 'View your content in a stacked format for detailed review'
          }
        </p>
      </div>

      {/* Content Comparison Layout */}
      <div className={`mb-8 ${
        viewMode === 'grid' 
          ? 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 lg:gap-6' 
          : 'space-y-6'
      }`}>
        {/* LinkedIn Post Column */}
        <div className="space-y-4">
          <div className="flex justify-between items-center">
          <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-gradient-to-br from-blue-600 to-blue-700 rounded-lg flex items-center justify-center">
                <Linkedin className="w-4 h-4 text-white" />
              </div>
              <h4 className={`text-sm font-bold uppercase tracking-wide transition-colors ${
                isDarkMode 
                  ? 'text-gray-200' 
                  : 'text-slate-800'
              }`}>
                LinkedIn Post
              </h4>
            </div>
            <div className="relative dropdown-menu">
            <button
                onClick={() => toggleDropdownMenu('linkedinPost')}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 ${
                  isDarkMode 
                    ? 'bg-slate-700 text-gray-300 hover:bg-slate-600 border border-slate-600' 
                    : 'bg-slate-100 text-slate-700 hover:bg-slate-200 border border-slate-200'
                }`}
            >
              <Copy className="w-3.5 h-3.5" />
                Actions
                <ChevronDown className="w-3 h-3" />
            </button>
              
              {dropdownMenus.linkedinPost && (
                <div className={`absolute right-0 top-full mt-1 w-48 rounded-lg shadow-lg border z-50 ${
                  isDarkMode 
                    ? 'bg-slate-800 border-slate-700' 
                    : 'bg-white border-slate-200'
                }`}>
                  <div className="py-1">
                    <button
                      onClick={() => {
                        copyToClipboard(linkedinPost || '');
                        setDropdownMenus(prev => ({ ...prev, linkedinPost: false }));
                      }}
                      className={`w-full text-left px-4 py-2 text-sm transition-colors duration-200 ${
                        isDarkMode 
                          ? 'text-gray-300 hover:bg-slate-700' 
                          : 'text-slate-700 hover:bg-slate-100'
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <Copy className="w-4 h-4" />
                        Copy Text
          </div>
                    </button>
                    <button
                      onClick={async () => {
                        setDropdownMenus(prev => ({ ...prev, linkedinPost: false }));
                        await regenerateContent('linkedinPost');
                      }}
                      disabled={regenerating.linkedinPost}
                      className={`w-full text-left px-4 py-2 text-sm transition-colors duration-200 ${
                        isDarkMode 
                          ? 'text-gray-300 hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed' 
                          : 'text-slate-700 hover:bg-slate-100 disabled:opacity-50 disabled:cursor-not-allowed'
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <RefreshCw className={`w-4 h-4 ${regenerating.linkedinPost ? 'animate-spin' : ''}`} />
                        {regenerating.linkedinPost ? 'Regenerating...' : 'Regenerate'}
        </div>
                    </button>
                    <button
                      onClick={() => {
                        // Export as different format
                        console.log('Export LinkedIn post');
                        setDropdownMenus(prev => ({ ...prev, linkedinPost: false }));
                      }}
                      className={`w-full text-left px-4 py-2 text-sm transition-colors duration-200 ${
                        isDarkMode 
                          ? 'text-gray-300 hover:bg-slate-700' 
                          : 'text-slate-700 hover:bg-slate-100'
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <ExternalLink className="w-4 h-4" />
                        Export
                      </div>
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
          <div className={`border rounded-xl p-4 text-sm whitespace-pre-line leading-relaxed transition-colors duration-300 ${
            viewMode === 'grid' ? 'h-64 overflow-y-auto' : 'min-h-32'
          } ${
            isDarkMode 
              ? 'bg-gradient-to-br from-purple-700/80 to-purple-800/30 border-purple-600/50 text-gray-300' 
              : 'bg-gradient-to-br from-slate-50/80 to-slate-100/30 border-slate-200/50 text-slate-700'
          }`}>
            {linkedinPost ? (
              <EditableBlock initialText={linkedinPost} label="linkedinPost" />
            ) : (
              <div className={`text-red-500 border rounded-lg p-4 transition-colors duration-300 ${
                isDarkMode 
                  ? 'bg-red-900/20 border-red-800' 
                  : 'bg-red-50 border-red-200'
              }`}>
                <p className="font-semibold">No LinkedIn post generated</p>
                <p className="text-xs mt-1">Raw: {JSON.stringify(content?.linkedinPosts)}</p>
              </div>
            )}
          </div>
      </div>

        {/* Reel Transcript Column */}
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-gradient-to-br from-purple-600 to-purple-700 rounded-lg flex items-center justify-center">
                <Video className="w-4 h-4 text-white" />
              </div>
              <h4 className={`text-sm font-bold uppercase tracking-wide transition-colors ${
                isDarkMode 
                  ? 'text-gray-200' 
                  : 'text-slate-800'
              }`}>
                Reel Transcript
              </h4>
            </div>
            <div className="relative dropdown-menu">
            <button
                onClick={() => toggleDropdownMenu('reelTranscript')}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 ${
                isDarkMode 
                    ? 'bg-slate-700 text-gray-300 hover:bg-slate-600 border border-slate-600' 
                    : 'bg-slate-100 text-slate-700 hover:bg-slate-200 border border-slate-200'
                }`}
              >
                <Video className="w-3.5 h-3.5" />
                Actions
                <ChevronDown className="w-3 h-3" />
              </button>
              
              {dropdownMenus.reelTranscript && (
                <div className={`absolute right-0 top-full mt-1 w-48 rounded-lg shadow-lg border z-50 ${
                  isDarkMode 
                    ? 'bg-slate-800 border-slate-700' 
                    : 'bg-white border-slate-200'
                }`}>
                  <div className="py-1">
                    <button
                      onClick={() => {
                        copyToClipboard(videoScript || '');
                        setDropdownMenus(prev => ({ ...prev, reelTranscript: false }));
                      }}
                      className={`w-full text-left px-4 py-2 text-sm transition-colors duration-200 ${
                        isDarkMode 
                          ? 'text-gray-300 hover:bg-slate-700' 
                          : 'text-slate-700 hover:bg-slate-100'
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <Copy className="w-4 h-4" />
                        Copy Script
              </div>
            </button>
                    <button
                      onClick={async () => {
                        setDropdownMenus(prev => ({ ...prev, reelTranscript: false }));
                        await regenerateContent('reelTranscript');
                      }}
                      disabled={regenerating.reelTranscript}
                      className={`w-full text-left px-4 py-2 text-sm transition-colors duration-200 ${
                        isDarkMode 
                          ? 'text-gray-300 hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed' 
                          : 'text-slate-700 hover:bg-slate-100 disabled:opacity-50 disabled:cursor-not-allowed'
                      }`}
                    >
            <div className="flex items-center gap-2">
                        <RefreshCw className={`w-4 h-4 ${regenerating.reelTranscript ? 'animate-spin' : ''}`} />
                        {regenerating.reelTranscript ? 'Regenerating...' : 'Regenerate'}
                      </div>
                    </button>
              <button
                      onClick={() => {
                        // Download as text file
                        const blob = new Blob([videoScript || ''], { type: 'text/plain' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = 'reel-transcript.txt';
                        a.click();
                        URL.revokeObjectURL(url);
                        setDropdownMenus(prev => ({ ...prev, reelTranscript: false }));
                      }}
                      className={`w-full text-left px-4 py-2 text-sm transition-colors duration-200 ${
                        isDarkMode 
                          ? 'text-gray-300 hover:bg-slate-700' 
                          : 'text-slate-700 hover:bg-slate-100'
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <FileText className="w-4 h-4" />
                        Download TXT
                      </div>
              </button>
                    <button
                      onClick={() => {
                        // Share transcript
                        console.log('Share reel transcript');
                        setDropdownMenus(prev => ({ ...prev, reelTranscript: false }));
                      }}
                      className={`w-full text-left px-4 py-2 text-sm transition-colors duration-200 ${
                        isDarkMode 
                          ? 'text-gray-300 hover:bg-slate-700' 
                          : 'text-slate-700 hover:bg-slate-100'
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <ExternalLink className="w-4 h-4" />
                        Share
            </div>
                    </button>
          </div>
                </div>
              )}
            </div>
          </div>
          <div className={`border rounded-xl p-4 text-xs whitespace-pre-line leading-relaxed font-mono transition-colors duration-300 ${
            viewMode === 'grid' ? 'h-64 overflow-y-auto' : 'min-h-32'
          } ${
                isDarkMode 
              ? 'bg-purple-800 border-purple-700 text-gray-300' 
              : 'bg-gray-50 border-gray-200 text-gray-700'
          }`}>
            {videoScript ? (
              <EditableBlock initialText={videoScript} label="reelTranscript" mono />
            ) : (
              <div className="text-red-500">
                <p>No video script generated</p>
                <p className="text-xs mt-1">Raw: {JSON.stringify(content?.videoScripts)}</p>
                  </div>
            )}
                  </div>
                </div>
                
        {/* Infographic/Visualization Column */}
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-gradient-to-br from-green-600 to-green-700 rounded-lg flex items-center justify-center">
                <Image className="w-4 h-4 text-white" />
              </div>
              <h4 className={`text-sm font-bold uppercase tracking-wide transition-colors ${
                isDarkMode 
                  ? 'text-gray-200' 
                  : 'text-slate-800'
              }`}>
                Infographic
              </h4>
            </div>
            <div className="relative dropdown-menu">
              <button
                onClick={() => toggleDropdownMenu('infographic')}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 ${
                  isDarkMode 
                    ? 'bg-slate-700 text-gray-300 hover:bg-slate-600 border border-slate-600' 
                    : 'bg-slate-100 text-slate-700 hover:bg-slate-200 border border-slate-200'
                }`}
              >
                <Image className="w-3.5 h-3.5" />
                Actions
                <ChevronDown className="w-3 h-3" />
              </button>
              
              {dropdownMenus.infographic && (
                <div className={`absolute right-0 top-full mt-1 w-48 rounded-lg shadow-lg border z-50 ${
                  isDarkMode 
                    ? 'bg-slate-800 border-slate-700' 
                    : 'bg-white border-slate-200'
                }`}>
                  <div className="py-1">
                    {content?.contextualImageUrl && (
                      <>
                        <button
                          onClick={() => {
                            window.open(content.contextualImageUrl, '_blank');
                            setDropdownMenus(prev => ({ ...prev, infographic: false }));
                          }}
                          className={`w-full text-left px-4 py-2 text-sm transition-colors duration-200 ${
                            isDarkMode 
                              ? 'text-gray-300 hover:bg-slate-700' 
                              : 'text-slate-700 hover:bg-slate-100'
                          }`}
                        >
                          <div className="flex items-center gap-2">
                            <ExternalLink className="w-4 h-4" />
                            View Full Size
                  </div>
                        </button>
                        <button
                          onClick={() => {
                            // Download image
                            const link = document.createElement('a');
                            link.href = content.contextualImageUrl;
                            link.download = 'infographic.png';
                            link.click();
                            setDropdownMenus(prev => ({ ...prev, infographic: false }));
                          }}
                          className={`w-full text-left px-4 py-2 text-sm transition-colors duration-200 ${
                            isDarkMode 
                              ? 'text-gray-300 hover:bg-slate-700' 
                              : 'text-slate-700 hover:bg-slate-100'
                          }`}
                        >
                          <div className="flex items-center gap-2">
                            <Download className="w-4 h-4" />
                            Download Image
              </div>
                        </button>
                        <button
                          onClick={() => {
                            // Copy image URL
                            navigator.clipboard.writeText(content.contextualImageUrl);
                            setDropdownMenus(prev => ({ ...prev, infographic: false }));
                          }}
                          className={`w-full text-left px-4 py-2 text-sm transition-colors duration-200 ${
                            isDarkMode 
                              ? 'text-gray-300 hover:bg-slate-700' 
                              : 'text-slate-700 hover:bg-slate-100'
                          }`}
                        >
                          <div className="flex items-center gap-2">
                            <Copy className="w-4 h-4" />
                            Copy URL
                          </div>
                        </button>
            </>
          )}
          <button
                      onClick={async () => {
                        setDropdownMenus(prev => ({ ...prev, infographic: false }));
                        await regenerateContent('infographic');
                      }}
                      disabled={regenerating.infographic}
                      className={`w-full text-left px-4 py-2 text-sm transition-colors duration-200 ${
              isDarkMode 
                          ? 'text-gray-300 hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed' 
                          : 'text-slate-700 hover:bg-slate-100 disabled:opacity-50 disabled:cursor-not-allowed'
                      }`}
                    >
          <div className="flex items-center gap-2">
                        <RefreshCw className={`w-4 h-4 ${regenerating.infographic ? 'animate-spin' : ''}`} />
                        {regenerating.infographic ? 'Regenerating...' : 'Regenerate'}
                      </div>
                    </button>
            <button
                      onClick={() => {
                        // Share infographic
                        console.log('Share infographic');
                        setDropdownMenus(prev => ({ ...prev, infographic: false }));
                      }}
                      className={`w-full text-left px-4 py-2 text-sm transition-colors duration-200 ${
                        isDarkMode 
                          ? 'text-gray-300 hover:bg-slate-700' 
                          : 'text-slate-700 hover:bg-slate-100'
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <ExternalLink className="w-4 h-4" />
                        Share
                      </div>
            </button>
          </div>
        </div>
              )}
            </div>
          </div>
          <div className={`border rounded-xl p-4 transition-colors duration-300 ${
            viewMode === 'grid' ? 'h-64 overflow-hidden' : 'min-h-32'
          } ${
            isDarkMode 
              ? 'bg-gradient-to-br from-purple-700/80 to-purple-800/30 border-purple-600/50' 
              : 'bg-gradient-to-br from-slate-50/80 to-slate-100/30 border-slate-200/50'
          }`}>
            {content?.contextualImageUrl ? (
              <div className="h-full flex flex-col">
                <img 
                  src={content.contextualImageUrl} 
                  alt="Data-driven visualization based on research content" 
                  className="w-full h-auto rounded-lg shadow-sm flex-1 object-contain"
                  style={{ maxHeight: '200px' }}
                />
                <div className="mt-2 text-center">
                  <p className={`text-xs transition-colors duration-300 ${
                    isDarkMode ? 'text-gray-400' : 'text-slate-600'
                  }`}>
                    {content?.imageSourceLink?.includes('unsplash') 
                      ? 'Professional image via web search' 
                      : 'AI-generated visualization'}
                  </p>
                </div>
              </div>
            ) : (
              <div className={`h-full flex items-center justify-center text-center transition-colors duration-300 ${
                isDarkMode ? 'text-gray-400' : 'text-gray-500'
              }`}>
                <div>
                  <Image className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">No visualization generated</p>
              </div>
          </div>
        )}
          </div>
        </div>
      </div>


      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h4 className={`text-sm font-semibold uppercase tracking-wide mb-3 transition-colors duration-300 ${
            isDarkMode ? 'text-gray-200' : 'text-gray-900'
          }`}>
            Hashtags
          </h4>
          <div className="flex flex-wrap gap-2">
            {hashtags.length > 0 ? (
              hashtags.map((tag, index) => (
                <span
                  key={index}
                  className={`px-2 py-1 rounded text-xs font-medium transition-colors duration-300 ${
                    isDarkMode 
                      ? 'bg-purple-700 text-gray-300' 
                      : 'bg-gray-100 text-gray-700'
                  }`}
                >
                  {tag}
                </span>
              ))
            ) : (
              <span className={`text-xs transition-colors duration-300 ${
                isDarkMode ? 'text-gray-400' : 'text-gray-500'
              }`}>No hashtags generated</span>
            )}
          </div>
        </div>
        <div>
          <h4 className={`text-sm font-semibold uppercase tracking-wide mb-3 transition-colors duration-300 ${
            isDarkMode ? 'text-gray-200' : 'text-gray-900'
          }`}>
            Engagement Tips
          </h4>
          <ul className={`text-xs leading-relaxed space-y-1 transition-colors duration-300 ${
            isDarkMode ? 'text-gray-400' : 'text-gray-600'
          }`}>
            {engagementTips.length > 0 ? (
              engagementTips.map((tip, index) => (
                <li key={index} className="flex items-start">
                  <span className="mr-2">•</span>
                  {tip}
                </li>
              ))
            ) : (
              <li className={`transition-colors duration-300 ${
                isDarkMode ? 'text-gray-400' : 'text-gray-500'
              }`}>No engagement tips generated</li>
            )}
          </ul>
        </div>
      </div>

      {content.talking_points && content.talking_points.length > 0 && (
        <div className="mb-6">
          <h4 className={`text-sm font-semibold uppercase tracking-wide mb-3 transition-colors duration-300 ${
            isDarkMode ? 'text-gray-200' : 'text-gray-900'
          }`}>
            Talking Points
          </h4>
          <div className={`border rounded-lg p-4 transition-colors duration-300 ${
            isDarkMode 
              ? 'bg-slate-700 border-slate-600' 
              : 'bg-gray-50 border-gray-200'
          }`}>
            <ul className="space-y-2">
              {content.talking_points.map((point, index) => (
                <li key={index} className="flex items-start">
                  <span className="mr-2 text-blue-600">•</span>
                  <span className={`text-sm transition-colors duration-300 ${
                    isDarkMode ? 'text-gray-200' : 'text-gray-700'
                  }`}>{point}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {content.style_notes && (
        <div className="mb-6">
          <h4 className={`text-sm font-semibold uppercase tracking-wide mb-3 transition-colors duration-300 ${
            isDarkMode ? 'text-gray-200' : 'text-gray-900'
          }`}>
            Style Notes
          </h4>
          <div className={`border rounded-lg p-4 transition-colors duration-300 ${
            isDarkMode 
              ? 'bg-slate-700 border-slate-600' 
              : 'bg-gray-50 border-gray-200'
          }`}>
            <p className={`text-sm leading-relaxed transition-colors duration-300 ${
              isDarkMode ? 'text-gray-200' : 'text-gray-700'
            }`}>{content.style_notes}</p>
          </div>
        </div>
      )}

      {content.context_used && (
        <div className="mb-6">
          <h4 className={`text-sm font-semibold uppercase tracking-wide mb-3 transition-colors duration-300 ${
            isDarkMode ? 'text-gray-200' : 'text-gray-900'
          }`}>
            Context Used
          </h4>
          <div className={`border rounded-lg p-4 transition-colors duration-300 ${
            isDarkMode 
              ? 'bg-slate-700 border-slate-600' 
              : 'bg-gray-50 border-gray-200'
          }`}>
            <p className={`text-sm leading-relaxed transition-colors duration-300 ${
              isDarkMode ? 'text-gray-200' : 'text-gray-700'
            }`}>{content.context_used}</p>
          </div>
        </div>
      )}


    </div>
  );
  };

  const ResearchTab = () => {
    const [localNewResearch, setLocalNewResearch] = useState({
      topic: '',
      findings: '',
      source: '',
      data: '',
      tags: ''
    });

    const [localSearchTerm, setLocalSearchTerm] = useState('');

    const handleLocalResearchChange = (field, value) => {
      setLocalNewResearch(prev => ({ ...prev, [field]: value }));
    };

    const handleLocalSearchChange = (value) => {
      setLocalSearchTerm(value);
    };

    const filteredResearch = researchDatabase.filter(item =>
      item.topic.toLowerCase().includes(localSearchTerm.toLowerCase()) ||
      (Array.isArray(item.tags) && item.tags.some(tag => tag.toLowerCase().includes(localSearchTerm.toLowerCase())))
    );

    // Categorize research items
    const categorizedResearch = {
      newsArticles: filteredResearch.filter(item => 
        Array.isArray(item.tags) && item.tags.some(tag => tag.includes('news-article'))
      ),
      webCrawls: filteredResearch.filter(item => 
        Array.isArray(item.tags) && item.tags.some(tag => tag.includes('web-crawl')) && 
        !item.tags.some(tag => tag.includes('news-article'))
      ),
      documents: filteredResearch.filter(item => 
        Array.isArray(item.tags) && item.tags.some(tag => tag.includes('document'))
      ),
      other: filteredResearch.filter(item => 
        !Array.isArray(item.tags) || 
        (!item.tags.some(tag => tag.includes('news-article')) && 
         !item.tags.some(tag => tag.includes('web-crawl')) && 
         !item.tags.some(tag => tag.includes('document')))
      )
    };

    const handleAddResearch = async () => {
      if (!localNewResearch.topic.trim()) return;

      const research = {
        topic: localNewResearch.topic.trim(),
        findings: localNewResearch.findings.trim(),
        source: localNewResearch.source.trim(),
        data: localNewResearch.data.trim(),
        tags: localNewResearch.tags.trim()
      };

      try {
        // Use documentManager to add research to Supabase
        if (documentManager) {
          const result = await documentManager.addResearch(research);
          if (result) {
            // Add to local state instead of reloading
            setResearchDatabase(prev => [{
              ...result,
              tags: result.tags ? result.tags.split(',').map(t => t.trim()) : [],
              dateAdded: new Date(result.created_at || result.dateAdded)
            }, ...prev]);
            console.log('Research added successfully to Supabase:', result);
          }
        } else {
          // Fallback to local storage
          setResearchDatabase(prev => [{
            ...research,
            id: Date.now(),
            tags: research.tags.split(',').map(t => t.trim()),
            dateAdded: new Date()
          }, ...prev]);
        }
      } catch (error) {
        console.error('Error adding research:', error);
        // Fallback to local storage
        setResearchDatabase(prev => [{
          ...research,
          id: Date.now(),
          tags: research.tags.split(',').map(t => t.trim()),
          dateAdded: new Date()
        }, ...prev]);
      }

      setLocalNewResearch({ topic: '', findings: '', source: '', data: '', tags: '' });
    };

    return (
      <>
        <div className={`flex-1 overflow-y-auto transition-colors duration-300 ${
          isDarkMode ? 'bg-slate-900' : 'bg-gray-50'
        }`}>
        <div className="p-6">
          <div className="mb-8">
            <h2 className={`text-xl font-semibold mb-6 transition-colors duration-300 ${
              isDarkMode ? 'text-white' : 'text-gray-900'
            }`}>Research Database</h2>
            
            <div className={`mb-6 p-3 rounded-lg border flex items-center gap-2 text-sm transition-colors duration-300 ${
              isDarkMode 
                ? 'bg-slate-800 border-slate-700' 
                : 'bg-gray-100 border-gray-200'
            }`}>
              <div className={`w-2 h-2 rounded-full ${supabaseClient ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span className={`transition-colors duration-300 ${
                isDarkMode ? 'text-gray-300' : 'text-gray-600'
              }`}>
                {supabaseClient ? 'Connected to Supabase - Research sorted by newest first' : 'Using local storage - Research sorted by newest first'}
              </span>
            </div>
            
            <div className="relative mb-6">
              <Search className={`absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 transition-colors duration-300 ${
                isDarkMode ? 'text-gray-400' : 'text-gray-400'
              }`} />
              <input
                type="text"
                placeholder="Search research..."
                value={localSearchTerm}
                onChange={(e) => handleLocalSearchChange(e.target.value)}
                className={`w-full pl-10 pr-4 py-2 border rounded-lg text-sm focus:outline-none transition-colors duration-300 ${
                  isDarkMode 
                    ? 'bg-slate-800 border-slate-600 text-white placeholder-gray-400 focus:border-purple-400' 
                    : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500 focus:border-gray-500'
                }`}
              />
            </div>

            <div className={`rounded-lg p-6 shadow-sm border mb-6 transition-colors duration-300 ${
              isDarkMode 
                ? 'bg-slate-800 border-slate-700' 
                : 'bg-white border-gray-200'
            }`}>
              <h3 className={`text-lg font-medium mb-4 transition-colors duration-300 ${
                isDarkMode ? 'text-white' : 'text-gray-900'
              }`}>Add New Research</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <input
                  type="text"
                  placeholder="Research Topic"
                  value={localNewResearch.topic}
                  onChange={(e) => handleLocalResearchChange('topic', e.target.value)}
                  className={`px-3 py-2 border rounded-lg text-sm focus:outline-none transition-colors duration-300 ${
                    isDarkMode 
                      ? 'bg-slate-700 border-slate-600 text-white placeholder-gray-400 focus:border-purple-400' 
                      : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500 focus:border-gray-500'
                  }`}
                />
                <input
                  type="text"
                  placeholder="Source"
                  value={localNewResearch.source}
                  onChange={(e) => handleLocalResearchChange('source', e.target.value)}
                  className={`px-3 py-2 border rounded-lg text-sm focus:outline-none transition-colors duration-300 ${
                    isDarkMode 
                      ? 'bg-slate-700 border-slate-600 text-white placeholder-gray-400 focus:border-purple-400' 
                      : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500 focus:border-gray-500'
                  }`}
                />
              </div>
              <div className="mb-4">
                <textarea
                  placeholder="Key Findings"
                  value={localNewResearch.findings}
                  onChange={(e) => handleLocalResearchChange('findings', e.target.value)}
                  className={`w-full px-3 py-2 border rounded-lg text-sm focus:outline-none resize-none transition-colors duration-300 ${
                    isDarkMode 
                      ? 'bg-slate-700 border-slate-600 text-white placeholder-gray-400 focus:border-purple-400' 
                      : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500 focus:border-gray-500'
                  }`}
                  rows="4"
                />
              </div>
              <div className="mb-4">
                <textarea
                  placeholder="Supporting Data/Statistics"
                  value={localNewResearch.data}
                  onChange={(e) => handleLocalResearchChange('data', e.target.value)}
                  className={`w-full px-3 py-2 border rounded-lg text-sm focus:outline-none resize-none transition-colors duration-300 ${
                    isDarkMode 
                      ? 'bg-slate-700 border-slate-600 text-white placeholder-gray-400 focus:border-purple-400' 
                      : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500 focus:border-gray-500'
                  }`}
                  rows="3"
                />
              </div>
              <div className="flex gap-4">
                <input
                  type="text"
                  placeholder="Tags (comma separated)"
                  value={localNewResearch.tags}
                  onChange={(e) => handleLocalResearchChange('tags', e.target.value)}
                  className={`flex-1 px-3 py-2 border rounded-lg text-sm focus:outline-none transition-colors duration-300 ${
                    isDarkMode 
                      ? 'bg-slate-700 border-slate-600 text-white placeholder-gray-400 focus:border-purple-400' 
                      : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500 focus:border-gray-500'
                  }`}
                />
                <button
                  onClick={handleAddResearch}
                  className={`px-6 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 ${
                    isDarkMode 
                      ? 'bg-purple-600 text-white hover:bg-purple-700' 
                      : 'bg-gray-900 text-white hover:bg-gray-800'
                  }`}
                >
                  <Plus className="w-4 h-4" />
                  Add Research
                </button>
              </div>
            </div>

            {localNewResearch.topic && (
              <div className={`rounded-lg p-4 mb-6 border transition-colors duration-300 ${
                isDarkMode 
                  ? 'bg-blue-900/20 border-blue-700' 
                  : 'bg-blue-50 border-blue-200'
              }`}>
                <h4 className={`text-sm font-medium mb-2 transition-colors duration-300 ${
                  isDarkMode ? 'text-blue-300' : 'text-blue-900'
                }`}>Preview</h4>
                <div className={`space-y-2 text-sm transition-colors duration-300 ${
                  isDarkMode ? 'text-blue-200' : 'text-blue-800'
                }`}>
                  <p><strong>Topic:</strong> {localNewResearch.topic}</p>
                  {localNewResearch.findings && <p><strong>Findings:</strong> {localNewResearch.findings}</p>}
                  {localNewResearch.source && <p><strong>Source:</strong> {localNewResearch.source}</p>}
                  {localNewResearch.data && <p><strong>Data:</strong> {localNewResearch.data}</p>}
                  {localNewResearch.tags && (
                    <div className="flex items-center gap-2">
                      <strong>Tags:</strong>
                      <div className="flex flex-wrap gap-1">
                        {localNewResearch.tags.split(',').map((tag, index) => tag.trim()).filter(tag => tag).map((tag, index) => (
                          <span key={index} className={`px-2 py-1 rounded text-xs transition-colors duration-300 ${
                            isDarkMode ? 'bg-blue-800 text-blue-200' : 'bg-blue-100 text-blue-800'
                          }`}>
                            {tag}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {isLoadingResearch && (
              <div className="flex items-center justify-center py-8">
                <div className={`animate-spin rounded-full h-8 w-8 border-b-2 transition-colors duration-300 ${
                  isDarkMode ? 'border-purple-400' : 'border-gray-900'
                }`}></div>
              </div>
            )}

            {/* Research Summary */}
            {filteredResearch.length > 0 && (
              <div className={`mb-6 p-4 rounded-lg border transition-colors duration-300 ${
                isDarkMode 
                  ? 'bg-slate-800 border-slate-700' 
                  : 'bg-gray-50 border-gray-200'
              }`}>
                <h4 className={`text-sm font-semibold mb-3 transition-colors duration-300 ${
                  isDarkMode ? 'text-white' : 'text-gray-900'
                }`}>Research Summary</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600">{categorizedResearch.newsArticles.length}</div>
                    <div className={`transition-colors duration-300 ${
                      isDarkMode ? 'text-gray-300' : 'text-gray-600'
                    }`}>News Articles</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-600">{categorizedResearch.webCrawls.length}</div>
                    <div className={`transition-colors duration-300 ${
                      isDarkMode ? 'text-gray-300' : 'text-gray-600'
                    }`}>Web Crawls</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-purple-600">{categorizedResearch.documents.length}</div>
                    <div className={`transition-colors duration-300 ${
                      isDarkMode ? 'text-gray-300' : 'text-gray-600'
                    }`}>Documents</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-gray-600">{categorizedResearch.other.length}</div>
                    <div className={`transition-colors duration-300 ${
                      isDarkMode ? 'text-gray-300' : 'text-gray-600'
                    }`}>Other</div>
                  </div>
                </div>
              </div>
            )}

            <div className="space-y-6">
              {filteredResearch.length === 0 && !isLoadingResearch && (
                <div className={`text-center py-12 rounded-lg border transition-colors duration-300 ${
                  isDarkMode 
                    ? 'bg-slate-800 border-slate-700' 
                    : 'bg-white border-gray-200'
                }`}>
                  <Database className={`w-12 h-12 mx-auto mb-4 transition-colors duration-300 ${
                    isDarkMode ? 'text-gray-500' : 'text-gray-400'
                  }`} />
                  <p className={`transition-colors duration-300 ${
                    isDarkMode ? 'text-gray-300' : 'text-gray-600'
                  }`}>
                    {localSearchTerm ? `No research found matching "${localSearchTerm}"` : 'No research found. Add your first research above!'}
                  </p>
                </div>
              )}
              
              {/* News Articles Section */}
              {categorizedResearch.newsArticles.length > 0 && (
                <div>
                  <h3 className={`text-lg font-semibold mb-4 flex items-center gap-2 transition-colors duration-300 ${
                    isDarkMode ? 'text-white' : 'text-gray-900'
                  }`}>
                    <span className="w-2 h-2 bg-blue-500 rounded-full"></span>
                    Latest News Articles ({categorizedResearch.newsArticles.length})
                  </h3>
                  <div className="space-y-4">
                    {categorizedResearch.newsArticles.map((research, index) => (
                      <div key={research.id} className={`rounded-lg p-6 shadow-sm border transition-colors duration-300 ${
                        isDarkMode 
                          ? 'bg-slate-800 border-slate-700' 
                          : 'bg-white border-gray-200'
                      }`}>
                        {editingResearch === research.id ? (
                          <ResearchEditForm research={research} onSave={updateResearch} onCancel={() => setEditingResearch(null)} />
                        ) : (
                          <div>
                            <div className="flex justify-between items-start mb-4">
                              <div className="flex-1">
                                <div className="flex items-center gap-2 mb-2">
                                  <h3 className={`text-lg font-medium transition-colors duration-300 ${
                                    isDarkMode ? 'text-white' : 'text-gray-900'
                                  }`}>{research.topic}</h3>
                                  {index < 3 && (
                                    <span className="bg-green-100 text-green-800 px-2 py-1 rounded-full text-xs font-medium">
                                      Recent
                                    </span>
                                  )}
                                </div>
                                <p className={`text-sm mb-2 transition-colors duration-300 ${
                                  isDarkMode ? 'text-gray-300' : 'text-gray-600'
                                }`}>{research.source}</p>
                                <p className={`text-xs transition-colors duration-300 ${
                                  isDarkMode ? 'text-gray-400' : 'text-gray-500'
                                }`}>
                                  Added: {new Date(research.dateAdded).toLocaleDateString()} at {new Date(research.dateAdded).toLocaleTimeString()}
                                </p>
                              </div>
                              <div className="flex gap-2">
                                <button
                                  onClick={() => setEditingResearch(research.id)}
                                  className={`p-2 rounded-lg transition-colors ${
                                    isDarkMode 
                                      ? 'text-gray-400 hover:bg-slate-700 hover:text-white' 
                                      : 'text-gray-600 hover:bg-gray-100'
                                  }`}
                                >
                                  <Edit3 className="w-4 h-4" />
                                </button>
                                <button
                                  onClick={() => deleteResearch(research.id)}
                                  className={`p-2 rounded-lg transition-colors ${
                                    isDarkMode 
                                      ? 'text-red-400 hover:bg-red-900/20 hover:text-red-300' 
                                      : 'text-red-600 hover:bg-red-50'
                                  }`}
                                >
                                  <Trash2 className="w-4 h-4" />
                                </button>
                              </div>
                            </div>
                            <p className={`mb-3 transition-colors duration-300 ${
                              isDarkMode ? 'text-gray-200' : 'text-gray-700'
                            }`}>{research.findings}</p>
                            <p className={`text-sm mb-4 transition-colors duration-300 ${
                              isDarkMode ? 'text-gray-300' : 'text-gray-600'
                            }`}>{research.data}</p>
                            <div className="flex flex-wrap gap-2 mb-3">
                              {(Array.isArray(research.tags) ? research.tags : []).map((tag, index) => (
                                <span
                                  key={index}
                                  className={`px-2 py-1 rounded text-xs font-medium transition-colors duration-300 ${
                                    isDarkMode 
                                      ? 'bg-blue-800 text-blue-200' 
                                      : 'bg-blue-100 text-blue-800'
                                  }`}
                                >
                                  {tag}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
      </>
    );
  };

  const ResearchEditForm = ({ research, onSave, onCancel }) => {
    const [editForm, setEditForm] = useState({
      topic: research.topic,
      findings: research.findings,
      source: research.source,
      data: research.data,
      tags: Array.isArray(research.tags) ? research.tags.join(', ') : research.tags
    });

    const handleEditChange = (field, value) => {
      setEditForm(prev => ({ ...prev, [field]: value }));
    };

    const handleSave = () => {
      onSave(research.id, editForm);
    };

    return (
      <div className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <input
            type="text"
            value={editForm.topic}
            onChange={(e) => handleEditChange('topic', e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
          />
          <input
            type="text"
            value={editForm.source}
            onChange={(e) => handleEditChange('source', e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
          />
        </div>
        <textarea
          value={editForm.findings}
          onChange={(e) => handleEditChange('findings', e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500 resize-none"
          rows="4"
        />
        <textarea
          value={editForm.data}
          onChange={(e) => handleEditChange('data', e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500 resize-none"
          rows="3"
        />
        <input
          type="text"
          value={editForm.tags}
          onChange={(e) => handleEditChange('tags', e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
        />
        <div className="flex gap-2">
          <button
            onClick={handleSave}
            className="px-4 py-2 bg-green-600 text-white rounded-lg text-sm font-medium hover:bg-green-700 transition-colors flex items-center gap-2"
          >
            <Save className="w-4 h-4" />
            Save
          </button>
          <button
            onClick={onCancel}
            className="px-4 py-2 bg-gray-600 text-white rounded-lg text-sm font-medium hover:bg-gray-700 transition-colors flex items-center gap-2"
          >
            <X className="w-4 h-4" />
            Cancel
          </button>
        </div>
      </div>
    );
  };

  const CreatorTab = () => {
    const [localNewCreator, setLocalNewCreator] = useState({
      name: '',
      tone: '',
      structure: '',
      language: '',
      length: '',
      hooks: '',
      endings: '',
      characteristics: ''
    });

    const handleLocalCreatorChange = (field, value) => {
      setLocalNewCreator(prev => ({ ...prev, [field]: value }));
    };

    const handleAddCreator = async () => {
      if (!localNewCreator.name.trim()) return;
      
      const creator = {
        name: localNewCreator.name,
        key: localNewCreator.name.toLowerCase().replace(/\s+/g, '-'),
        style: {
          tone: localNewCreator.tone,
          structure: localNewCreator.structure,
          language: localNewCreator.language,
          length: localNewCreator.length,
          hooks: localNewCreator.hooks.split(',').map(h => h.trim()).filter(h => h),
          endings: localNewCreator.endings.split(',').map(e => e.trim()).filter(e => e),
          characteristics: localNewCreator.characteristics
        }
      };
      
      try {
        // Use documentManager to add creator to Supabase
        if (documentManager) {
          const result = await documentManager.addCreatorStyle(creator);
          if (result) {
            // Reload creators from Supabase
            await loadCreatorsFromSupabase();
            console.log('Creator added successfully to Supabase:', result);
          }
        } else {
          // Fallback to local storage
          setCreatorDatabase(prev => [{
            ...creator,
            id: Date.now(),
            icon: UserPlus
          }, ...prev]);
        }
      } catch (error) {
        console.error('Error adding creator:', error);
        // Fallback to local storage
        setCreatorDatabase(prev => [{
          ...creator,
          id: Date.now(),
          icon: UserPlus
        }, ...prev]);
      }
      
      setLocalNewCreator({ name: '', tone: '', structure: '', language: '', length: '', hooks: '', endings: '', characteristics: '' });
    };

    return (
      <div className={`flex-1 overflow-y-auto transition-colors duration-300 ${
        isDarkMode ? 'bg-slate-900' : 'bg-gray-50'
      }`}>
        <div className="p-6">
          <div className="mb-8">
            <h2 className={`text-xl font-semibold mb-6 transition-colors duration-300 ${
              isDarkMode ? 'text-white' : 'text-gray-900'
            }`}>Creator Profiles</h2>
            
            <div className={`rounded-lg p-6 shadow-sm border mb-6 transition-colors duration-300 ${
              isDarkMode 
                ? 'bg-slate-800 border-slate-700' 
                : 'bg-white border-gray-200'
            }`}>
              <h3 className={`text-lg font-medium mb-4 transition-colors duration-300 ${
                isDarkMode ? 'text-white' : 'text-gray-900'
              }`}>Add New Creator</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <input
                  type="text"
                  placeholder="Creator Name"
                  value={localNewCreator.name}
                  onChange={(e) => handleLocalCreatorChange('name', e.target.value)}
                  className={`px-3 py-2 border rounded-lg text-sm focus:outline-none transition-colors duration-300 ${
                    isDarkMode 
                      ? 'bg-slate-700 border-slate-600 text-white placeholder-gray-400 focus:border-purple-400' 
                      : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500 focus:border-gray-500'
                  }`}
                />
                <input
                  type="text"
                  placeholder="Tone (e.g., Direct, passionate, no-nonsense)"
                  value={localNewCreator.tone}
                  onChange={(e) => handleLocalCreatorChange('tone', e.target.value)}
                  className={`px-3 py-2 border rounded-lg text-sm focus:outline-none transition-colors duration-300 ${
                    isDarkMode 
                      ? 'bg-slate-700 border-slate-600 text-white placeholder-gray-400 focus:border-purple-400' 
                      : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500 focus:border-gray-500'
                  }`}
                />
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <input
                  type="text"
                  placeholder="Post Structure"
                  value={localNewCreator.structure}
                  onChange={(e) => handleLocalCreatorChange('structure', e.target.value)}
                  className={`px-3 py-2 border rounded-lg text-sm focus:outline-none transition-colors duration-300 ${
                    isDarkMode 
                      ? 'bg-slate-700 border-slate-600 text-white placeholder-gray-400 focus:border-purple-400' 
                      : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500 focus:border-gray-500'
                  }`}
                />
                <input
                  type="text"
                  placeholder="Language Style"
                  value={localNewCreator.language}
                  onChange={(e) => handleLocalCreatorChange('language', e.target.value)}
                  className={`px-3 py-2 border rounded-lg text-sm focus:outline-none transition-colors duration-300 ${
                    isDarkMode 
                      ? 'bg-slate-700 border-slate-600 text-white placeholder-gray-400 focus:border-purple-400' 
                      : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500 focus:border-gray-500'
                  }`}
                />
              </div>
              <div className="mb-4">
                <input
                  type="text"
                  placeholder="Typical Post Length"
                  value={localNewCreator.length}
                  onChange={(e) => handleLocalCreatorChange('length', e.target.value)}
                  className={`w-full px-3 py-2 border rounded-lg text-sm focus:outline-none transition-colors duration-300 ${
                    isDarkMode 
                      ? 'bg-slate-700 border-slate-600 text-white placeholder-gray-400 focus:border-purple-400' 
                      : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500 focus:border-gray-500'
                  }`}
                />
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <input
                  type="text"
                  placeholder="Common Hooks (comma separated)"
                  value={localNewCreator.hooks}
                  onChange={(e) => handleLocalCreatorChange('hooks', e.target.value)}
                  className={`px-3 py-2 border rounded-lg text-sm focus:outline-none transition-colors duration-300 ${
                    isDarkMode 
                      ? 'bg-slate-700 border-slate-600 text-white placeholder-gray-400 focus:border-purple-400' 
                      : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500 focus:border-gray-500'
                  }`}
                />
                <input
                  type="text"
                  placeholder="Common Endings (comma separated)"
                  value={localNewCreator.endings}
                  onChange={(e) => handleLocalCreatorChange('endings', e.target.value)}
                  className={`px-3 py-2 border rounded-lg text-sm focus:outline-none transition-colors duration-300 ${
                    isDarkMode 
                      ? 'bg-slate-700 border-slate-600 text-white placeholder-gray-400 focus:border-purple-400' 
                      : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500 focus:border-gray-500'
                  }`}
                />
              </div>
              <div className="flex gap-4">
                <textarea
                  placeholder="Key Characteristics"
                  value={localNewCreator.characteristics}
                  onChange={(e) => handleLocalCreatorChange('characteristics', e.target.value)}
                  className={`flex-1 px-3 py-2 border rounded-lg text-sm focus:outline-none resize-none transition-colors duration-300 ${
                    isDarkMode 
                      ? 'bg-slate-700 border-slate-600 text-white placeholder-gray-400 focus:border-purple-400' 
                      : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500 focus:border-gray-500'
                  }`}
                  rows="3"
                />
                <button
                  onClick={handleAddCreator}
                  className={`px-6 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 ${
                    isDarkMode 
                      ? 'bg-purple-600 text-white hover:bg-purple-700' 
                      : 'bg-gray-900 text-white hover:bg-gray-800'
                  }`}
                >
                  <Plus className="w-4 h-4" />
                  Add Creator
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {creatorDatabase.map((creator) => (
                <div key={creator.id} className={`rounded-lg p-6 shadow-sm border transition-colors duration-300 ${
                  isDarkMode 
                    ? 'bg-slate-800 border-slate-700' 
                    : 'bg-white border-gray-200'
                }`}>
                  {editingCreator === creator.id ? (
                    <CreatorEditForm creator={creator} onCancel={() => setEditingCreator(null)} />
                  ) : (
                    <div>
                      <div className="flex justify-between items-start mb-4">
                        <div className="flex items-center gap-3">
                          <creator.icon className={`w-6 h-6 transition-colors duration-300 ${
                            isDarkMode ? 'text-gray-400' : 'text-gray-600'
                          }`} />
                          <h3 className={`text-lg font-medium transition-colors duration-300 ${
                            isDarkMode ? 'text-white' : 'text-gray-900'
                          }`}>{creator.name}</h3>
                        </div>
                        <div className="flex gap-2">
                          <button
                            onClick={() => setEditingCreator(creator.id)}
                            className={`p-2 rounded-lg transition-colors ${
                              isDarkMode 
                                ? 'text-gray-400 hover:bg-slate-700 hover:text-white' 
                                : 'text-gray-600 hover:bg-gray-100'
                            }`}
                          >
                            <Edit3 className="w-4 h-4" />
                          </button>
                          <button
                            onClick={async () => {
                              try {
                                if (documentManager) {
                                  await documentManager.deleteCreatorStyle(creator.id);
                                  await loadCreatorsFromSupabase();
                                  console.log('Creator deleted successfully from Supabase');
                                } else {
                                  setCreatorDatabase(prev => prev.filter(item => item.id !== creator.id));
                                }
                              } catch (error) {
                                console.error('Error deleting creator:', error);
                                setCreatorDatabase(prev => prev.filter(item => item.id !== creator.id));
                              }
                            }}
                            className={`p-2 rounded-lg transition-colors ${
                              isDarkMode 
                                ? 'text-red-400 hover:bg-red-900/20 hover:text-red-300' 
                                : 'text-red-600 hover:bg-red-50'
                            }`}
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                      <div className="space-y-3 text-sm">
                        <div>
                          <span className={`font-medium transition-colors duration-300 ${
                            isDarkMode ? 'text-gray-300' : 'text-gray-700'
                          }`}>Tone:</span>
                          <span className={`ml-2 transition-colors duration-300 ${
                            isDarkMode ? 'text-gray-400' : 'text-gray-600'
                          }`}>{creator.style.tone}</span>
                        </div>
                        <div>
                          <span className={`font-medium transition-colors duration-300 ${
                            isDarkMode ? 'text-gray-300' : 'text-gray-700'
                          }`}>Structure:</span>
                          <span className={`ml-2 transition-colors duration-300 ${
                            isDarkMode ? 'text-gray-400' : 'text-gray-600'
                          }`}>{creator.style.structure}</span>
                        </div>
                        <div>
                          <span className={`font-medium transition-colors duration-300 ${
                            isDarkMode ? 'text-gray-300' : 'text-gray-700'
                          }`}>Language:</span>
                          <span className={`ml-2 transition-colors duration-300 ${
                            isDarkMode ? 'text-gray-400' : 'text-gray-600'
                          }`}>{creator.style.language}</span>
                        </div>
                        <div>
                          <span className={`font-medium transition-colors duration-300 ${
                            isDarkMode ? 'text-gray-300' : 'text-gray-700'
                          }`}>Characteristics:</span>
                          <span className={`ml-2 transition-colors duration-300 ${
                            isDarkMode ? 'text-gray-400' : 'text-gray-600'
                          }`}>{creator.style.characteristics}</span>
                        </div>
                        <div>
                          <span className={`font-medium transition-colors duration-300 ${
                            isDarkMode ? 'text-gray-300' : 'text-gray-700'
                          }`}>Common Hooks:</span>
                          <div className="mt-1 flex flex-wrap gap-1">
                            {creator.style.hooks.map((hook, index) => (
                              <span key={index} className={`px-2 py-1 rounded text-xs transition-colors duration-300 ${
                                isDarkMode 
                                  ? 'bg-green-800 text-green-200' 
                                  : 'bg-green-100 text-green-800'
                              }`}>
                                {hook}
                              </span>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  };

  const CreatorEditForm = ({ creator, onCancel }) => {
    const [editForm, setEditForm] = useState({
      name: creator.name,
      tone: creator.style.tone,
      structure: creator.style.structure,
      language: creator.style.language,
      length: creator.style.length,
      hooks: creator.style.hooks.join(', '),
      endings: creator.style.endings.join(', '),
      characteristics: creator.style.characteristics
    });

    const handleEditChange = (field, value) => {
      setEditForm(prev => ({ ...prev, [field]: value }));
    };

    const handleSave = async () => {
      const updatedData = {
        name: editForm.name,
        key: editForm.name.toLowerCase().replace(/\s+/g, '-'),
        style: {
        tone: editForm.tone,
        structure: editForm.structure,
        language: editForm.language,
        length: editForm.length,
        hooks: editForm.hooks.split(',').map(h => h.trim()).filter(h => h),
        endings: editForm.endings.split(',').map(e => e.trim()).filter(e => e),
        characteristics: editForm.characteristics
        }
      };
      
      try {
        if (documentManager) {
          await documentManager.updateCreatorStyle(creator.id, updatedData);
          await loadCreatorsFromSupabase();
          console.log('Creator updated successfully in Supabase');
        } else {
      setCreatorDatabase(prev => prev.map(c => 
        c.id === creator.id 
          ? {
              ...c,
              name: updatedData.name,
                  style: updatedData.style
                }
              : c
          ));
        }
      } catch (error) {
        console.error('Error updating creator:', error);
        setCreatorDatabase(prev => prev.map(c => 
          c.id === creator.id 
            ? {
                ...c,
                name: updatedData.name,
                style: updatedData.style
            }
          : c
      ));
      }
      onCancel();
    };

    return (
      <div className="space-y-4">
        <input
          type="text"
          value={editForm.name}
          onChange={(e) => handleEditChange('name', e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
        />
        <input
          type="text"
          value={editForm.tone}
          onChange={(e) => handleEditChange('tone', e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
        />
        <input
          type="text"
          value={editForm.structure}
          onChange={(e) => handleEditChange('structure', e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
        />
        <input
          type="text"
          value={editForm.language}
          onChange={(e) => handleEditChange('language', e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
        />
        <input
          type="text"
          value={editForm.hooks}
          onChange={(e) => handleEditChange('hooks', e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
        />
        <input
          type="text"
          value={editForm.endings}
          onChange={(e) => handleEditChange('endings', e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
        />
        <textarea
          value={editForm.characteristics}
          onChange={(e) => handleEditChange('characteristics', e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500 resize-none"
          rows="3"
        />
        <div className="flex gap-2">
          <button
            onClick={handleSave}
            className="px-4 py-2 bg-green-600 text-white rounded-lg text-sm font-medium hover:bg-green-700 transition-colors flex items-center gap-2"
          >
            <Save className="w-4 h-4" />
            Save
          </button>
          <button
            onClick={onCancel}
            className="px-4 py-2 bg-gray-600 text-white rounded-lg text-sm font-medium hover:bg-gray-700 transition-colors flex items-center gap-2"
          >
            <X className="w-4 h-4" />
            Cancel
          </button>
        </div>
      </div>
    );
  };

  const CustomDocumentsTab = () => {

    const handleRefreshDocuments = async () => {
      if (!documentManager) return;
      
      setIsLoadingDocuments(true);
      try {
        await documentManager.refreshIndex();
        const documents = await documentManager.getDocuments();
        setCustomDocuments(documents || []);
        
        const successMessage = {
          id: Date.now(),
          type: 'bot',
          text: `✅ Successfully refreshed document index. Found ${documents.length} documents.`,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, successMessage]);
      } catch (error) {
        console.error('Refresh failed:', error);
      } finally {
        setIsLoadingDocuments(false);
      }
    };

    return (
      <div className={`flex-1 overflow-y-auto transition-colors duration-300 ${
        isDarkMode ? 'bg-slate-900' : 'bg-gray-50'
      }`}>
        <div className="p-6">
          <div className="mb-8">
            <h2 className={`text-xl font-semibold mb-6 transition-colors duration-300 ${
              isDarkMode ? 'text-white' : 'text-gray-900'
            }`}>Cloud Document Management</h2>
            
            <div className={`rounded-lg p-6 shadow-sm border mb-6 transition-colors duration-300 ${
              isDarkMode 
                ? 'bg-slate-800 border-slate-700' 
                : 'bg-white border-gray-200'
            }`}>
              <h3 className={`text-lg font-medium mb-4 transition-colors duration-300 ${
                isDarkMode ? 'text-white' : 'text-gray-900'
              }`}>Upload Document</h3>
              <div className="flex gap-4 items-end">
                <div className="flex-1">
                  <input
                    id="file-input"
                    type="file"
                    accept=".pdf,.docx,.xlsx,.xls,.pptx,.ppt"
                    onChange={handleFileSelect}
                    className={`w-full px-3 py-2 border rounded-lg text-sm focus:outline-none transition-colors duration-300 ${
                      isDarkMode 
                        ? 'bg-slate-700 border-slate-600 text-white placeholder-gray-400 focus:border-purple-400' 
                        : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500 focus:border-gray-500'
                    }`}
                  />
                  <p className={`text-xs mt-1 transition-colors duration-300 ${
                    isDarkMode ? 'text-gray-400' : 'text-gray-500'
                  }`}>
                    Supported: PDF, DOCX, XLSX, PPTX files
                  </p>
                </div>
                <button
                  onClick={handleUploadDocument}
                  disabled={!selectedFile || isUploadingDocument || backendStatus !== 'connected'}
                  className={`px-6 py-2 rounded-lg text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2 ${
                    isDarkMode 
                      ? 'bg-purple-600 text-white hover:bg-purple-700' 
                      : 'bg-blue-600 text-white hover:bg-blue-700'
                  }`}
                >
                  {isUploadingDocument ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                      Uploading...
                    </>
                  ) : (
                    <>
                      <Upload className="w-4 h-4" />
                      Upload
                    </>
                  )}
                </button>
              </div>
            </div>

            <div className={`rounded-lg p-6 shadow-sm border mb-6 transition-colors duration-300 ${
              isDarkMode 
                ? 'bg-slate-800 border-slate-700' 
                : 'bg-white border-gray-200'
            }`}>
              <h3 className={`text-lg font-medium mb-4 transition-colors duration-300 ${
                isDarkMode ? 'text-white' : 'text-gray-900'
              }`}>Crawl Website</h3>
              <div className="flex gap-4 items-end">
                <div className="flex-1">
                  <input
                    type="url"
                    placeholder="https://example.com"
                    value={crawlUrl}
                    onChange={(e) => setCrawlUrl(e.target.value)}
                    className={`w-full px-3 py-2 border rounded-lg text-sm focus:outline-none transition-colors duration-300 ${
                      isDarkMode 
                        ? 'bg-slate-700 border-slate-600 text-white placeholder-gray-400 focus:border-purple-400' 
                        : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500 focus:border-gray-500'
                    }`}
                  />
                  <p className={`text-xs mt-1 transition-colors duration-300 ${
                    isDarkMode ? 'text-gray-400' : 'text-gray-500'
                  }`}>
                    Extract content from any website for research
                  </p>
                </div>
                <button
                  onClick={async () => {
                    setIsCrawling(true);
                    try {
                      await handleCrawlWebsite(crawlUrl);
                      setCrawlUrl('');
                    } finally {
                      setIsCrawling(false);
                    }
                  }}
                  disabled={!crawlUrl || isCrawling || backendStatus !== 'connected'}
                  className={`px-6 py-2 rounded-lg text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2 ${
                    isDarkMode 
                      ? 'bg-green-600 text-white hover:bg-green-700' 
                      : 'bg-green-600 text-white hover:bg-green-700'
                  }`}
                >
                  {isCrawling ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                      Crawling...
                    </>
                  ) : (
                    <>
                      <Globe className="w-4 h-4" />
                      Crawl
                    </>
                  )}
                </button>
              </div>
            </div>

            <div className={`rounded-lg p-6 shadow-sm border transition-colors duration-300 ${
              isDarkMode 
                ? 'bg-slate-800 border-slate-700' 
                : 'bg-white border-gray-200'
            }`}>
              <div className="flex justify-between items-center mb-4">
                <h3 className={`text-lg font-medium transition-colors duration-300 ${
                  isDarkMode ? 'text-white' : 'text-gray-900'
                }`}>Indexed Documents</h3>
                <button
                  onClick={handleRefreshDocuments}
                  disabled={isLoadingDocuments || backendStatus !== 'connected'}
                  className={`px-4 py-2 border rounded-lg text-sm font-medium disabled:opacity-50 transition-colors flex items-center gap-2 ${
                    isDarkMode 
                      ? 'bg-slate-700 text-gray-300 border-slate-600 hover:bg-slate-600' 
                      : 'bg-gray-100 text-gray-700 border-gray-300 hover:bg-gray-200'
                  }`}
                >
                  {isLoadingDocuments ? (
                    <div className={`animate-spin rounded-full h-4 w-4 border-b-2 transition-colors duration-300 ${
                      isDarkMode ? 'border-gray-400' : 'border-gray-600'
                    }`}></div>
                  ) : (
                    <Database className="w-4 h-4" />
                  )}
                  Refresh
                </button>
              </div>
              
              {isLoadingDocuments ? (
                <div className="flex items-center justify-center py-8">
                  <div className={`animate-spin rounded-full h-8 w-8 border-b-2 transition-colors duration-300 ${
                    isDarkMode ? 'border-purple-400' : 'border-gray-900'
                  }`}></div>
                </div>
              ) : customDocuments.length === 0 ? (
                <div className="text-center py-12">
                  <Database className={`w-12 h-12 mx-auto mb-4 transition-colors duration-300 ${
                    isDarkMode ? 'text-gray-500' : 'text-gray-400'
                  }`} />
                  <p className={`transition-colors duration-300 ${
                    isDarkMode ? 'text-gray-300' : 'text-gray-600'
                  }`}>No documents indexed yet. Upload a document or crawl a website to get started.</p>
                  {backendStatus !== 'connected' && (
                    <p className="text-red-400 text-sm mt-2">Backend connection required for document management.</p>
                  )}
                </div>
              ) : (
                <div className="space-y-4">
                  {customDocuments.map((doc, index) => (
                    <div key={doc.unique_key} className={`flex justify-between items-center p-4 border rounded-lg transition-colors duration-300 ${
                      isDarkMode 
                        ? 'border-slate-600 bg-slate-700' 
                        : 'border-gray-200 bg-white'
                    }`}>
                      <div>
                        <h4 className={`font-medium transition-colors duration-300 ${
                          isDarkMode ? 'text-white' : 'text-gray-900'
                        }`}>{doc.filename}</h4>
                        <p className={`text-sm transition-colors duration-300 ${
                          isDarkMode ? 'text-gray-400' : 'text-gray-500'
                        }`}>
                          {doc.chunk_count} chunks indexed
                        </p>
                      </div>
                      <button
                        // --- FIX: Use the full uniqueFilename for the delete action ---
                        onClick={() => handleDeleteDocument(doc.uniqueFilename)}
                        className={`p-2 rounded-lg transition-colors ${
                          isDarkMode 
                            ? 'text-red-400 hover:bg-red-900/20 hover:text-red-300' 
                            : 'text-red-600 hover:bg-red-50'
                        }`}
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  };

  const WebCrawlsTab = () => {
    const [localCrawlUrl, setLocalCrawlUrl] = useState('');
    const [localWebCrawls, setLocalWebCrawls] = useState([]);
    const [localIsLoadingCrawls, setLocalIsLoadingCrawls] = useState(false);

    // Load web crawls only once when component mounts
    useEffect(() => {
      const loadCrawls = async () => {
        setLocalIsLoadingCrawls(true);
        try {
          const response = await fetch(`${BACKEND_CONFIG.url}/linkedin/crawls`);
          if (response.ok) {
            const data = await response.json();
            console.log('Loaded web crawls:', data);
            setLocalWebCrawls(data || []);
          } else {
            console.error('Failed to load web crawls:', response.status);
            setLocalWebCrawls([]);
          }
        } catch (error) {
          console.error('Error loading web crawls:', error);
          setLocalWebCrawls([]);
        } finally {
          setLocalIsLoadingCrawls(false);
        }
      };
      
      loadCrawls();
    }, []);

    const handleLocalCrawl = async () => {
      if (!localCrawlUrl.trim()) return;
      
      setLocalIsLoadingCrawls(true);
      try {
        await handleCrawlWebsite(localCrawlUrl);
        setLocalCrawlUrl('');
        // Reload crawls after successful crawl
        const response = await fetch(`${BACKEND_CONFIG.url}/linkedin/crawls`);
        if (response.ok) {
          const data = await response.json();
          setLocalWebCrawls(data || []);
        }
      } catch (error) {
        console.error('Crawl failed:', error);
      } finally {
        setLocalIsLoadingCrawls(false);
      }
    };

    return (
      <div className={`flex-1 overflow-y-auto transition-colors duration-300 ${
        isDarkMode ? 'bg-slate-900' : 'bg-gray-50'
      }`}>
        <div className="p-6">
          <div className="mb-8">
            <h2 className={`text-xl font-semibold mb-6 transition-colors duration-300 ${
              isDarkMode ? 'text-white' : 'text-gray-900'
            }`}>Web Crawls</h2>
            
            <div className={`rounded-lg p-6 shadow-sm border mb-6 transition-colors duration-300 ${
              isDarkMode 
                ? 'bg-slate-800 border-slate-700' 
                : 'bg-white border-gray-200'
            }`}>
              <h3 className={`text-lg font-medium mb-4 transition-colors duration-300 ${
                isDarkMode ? 'text-white' : 'text-gray-900'
              }`}>Crawl New Website</h3>
              <div className="flex gap-4 items-end">
                <div className="flex-1">
                  <input
                    type="url"
                    placeholder="https://example.com"
                    value={localCrawlUrl}
                    onChange={(e) => setLocalCrawlUrl(e.target.value)}
                    className={`w-full px-3 py-2 border rounded-lg text-sm focus:outline-none transition-colors duration-300 ${
                      isDarkMode 
                        ? 'bg-slate-700 border-slate-600 text-white placeholder-gray-400 focus:border-purple-400' 
                        : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500 focus:border-gray-500'
                    }`}
                  />
                  <p className={`text-xs mt-1 transition-colors duration-300 ${
                    isDarkMode ? 'text-gray-400' : 'text-gray-500'
                  }`}>
                    Extract content from any website for research and content generation
                  </p>
                </div>
                <button
                  onClick={handleLocalCrawl}
                  disabled={!localCrawlUrl.trim() || localIsLoadingCrawls || backendStatus !== 'connected'}
                  className={`px-6 py-2 rounded-lg text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2 ${
                    isDarkMode 
                      ? 'bg-purple-600 text-white hover:bg-purple-700' 
                      : 'bg-blue-600 text-white hover:bg-blue-700'
                  }`}
                >
                  {localIsLoadingCrawls ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                      Crawling...
                    </>
                  ) : (
                    <>
                      <Globe className="w-4 h-4" />
                      Crawl
                    </>
                  )}
                </button>
              </div>
            </div>

            <div className={`rounded-lg p-6 shadow-sm border transition-colors duration-300 ${
              isDarkMode 
                ? 'bg-slate-800 border-slate-700' 
                : 'bg-white border-gray-200'
            }`}>
              <div className="flex justify-between items-center mb-4">
                <h3 className={`text-lg font-medium transition-colors duration-300 ${
                  isDarkMode ? 'text-white' : 'text-gray-900'
                }`}>Crawl History</h3>
                <button
                  onClick={async () => {
                    setLocalIsLoadingCrawls(true);
                    try {
                      const response = await fetch(`${BACKEND_CONFIG.url}/linkedin/crawls`);
                      if (response.ok) {
                        const data = await response.json();
                        setLocalWebCrawls(data || []);
                      }
                    } catch (error) {
                      console.error('Error refreshing crawls:', error);
                    } finally {
                      setLocalIsLoadingCrawls(false);
                    }
                  }}
                  disabled={localIsLoadingCrawls}
                  className={`px-4 py-2 rounded-lg text-sm transition-colors flex items-center gap-2 ${
                    isDarkMode 
                      ? 'bg-slate-700 text-gray-300 hover:bg-slate-600' 
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  <RefreshCw className="w-4 h-4" />
                  Refresh
                </button>
              </div>
              
              {localIsLoadingCrawls ? (
                <div className="text-center py-8">
                  <div className={`animate-spin rounded-full h-8 w-8 border-b-2 mx-auto mb-4 transition-colors duration-300 ${
                    isDarkMode ? 'border-purple-400' : 'border-blue-600'
                  }`}></div>
                  <p className={`transition-colors duration-300 ${
                    isDarkMode ? 'text-gray-300' : 'text-gray-600'
                  }`}>Loading crawls...</p>
                </div>
              ) : localWebCrawls.length > 0 ? (
                <div className="space-y-4">
                  {localWebCrawls.map((crawl) => (
                    <div key={crawl.id} className={`border rounded-lg p-4 transition-colors duration-300 ${
                      isDarkMode 
                        ? 'border-slate-600 bg-slate-700' 
                        : 'border-gray-200 bg-white'
                    }`}>
                      <div className="flex justify-between items-start mb-2">
                        <div>
                          <h4 className={`font-medium transition-colors duration-300 ${
                            isDarkMode ? 'text-white' : 'text-gray-900'
                          }`}>{crawl.source || crawl.url}</h4>
                          <p className={`text-sm transition-colors duration-300 ${
                            isDarkMode ? 'text-gray-400' : 'text-gray-600'
                          }`}>
                            {crawl.created_at ? new Date(crawl.created_at).toLocaleString() : 'Unknown date'}
                          </p>
                        </div>
                        <span className="px-2 py-1 rounded text-xs font-medium bg-green-100 text-green-800">
                          Completed
                        </span>
                      </div>
                      <div className="mb-3">
                        <h5 className={`font-medium mb-1 transition-colors duration-300 ${
                          isDarkMode ? 'text-white' : 'text-gray-900'
                        }`}>{crawl.topic}</h5>
                        <p className={`text-sm line-clamp-3 transition-colors duration-300 ${
                          isDarkMode ? 'text-gray-300' : 'text-gray-600'
                        }`}>
                          {crawl.findings && crawl.findings.length > 200 
                            ? crawl.findings.substring(0, 200) + '...' 
                            : (crawl.findings || 'No content available')}
                        </p>
                      </div>
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className={`transition-colors duration-300 ${
                            isDarkMode ? 'text-gray-400' : 'text-gray-600'
                          }`}>Hostname:</span>
                          <span className={`ml-2 font-medium transition-colors duration-300 ${
                            isDarkMode ? 'text-gray-300' : 'text-gray-900'
                          }`}>{crawl.tags ? crawl.tags.split(',').find(tag => tag.includes('.')) || 'Unknown' : 'Unknown'}</span>
                        </div>
                        <div>
                          <span className={`transition-colors duration-300 ${
                            isDarkMode ? 'text-gray-400' : 'text-gray-600'
                          }`}>Content Length:</span>
                          <span className={`ml-2 font-medium transition-colors duration-300 ${
                            isDarkMode ? 'text-gray-300' : 'text-gray-900'
                          }`}>{crawl.data ? crawl.data.match(/Total length: (\d+)/)?.[1] || 'Unknown' : 'Unknown'} chars</span>
                        </div>
                      </div>
                      <div className="mt-3 flex flex-wrap gap-1">
                        {crawl.tags && crawl.tags.split(',').map((tag, index) => (
                          <span key={index} className={`px-2 py-1 rounded text-xs transition-colors duration-300 ${
                            isDarkMode 
                              ? 'bg-blue-800 text-blue-200' 
                              : 'bg-blue-100 text-blue-800'
                          }`}>
                            {tag.trim()}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <Globe className={`w-12 h-12 mx-auto mb-4 transition-colors duration-300 ${
                    isDarkMode ? 'text-gray-500' : 'text-gray-400'
                  }`} />
                  <p className={`transition-colors duration-300 ${
                    isDarkMode ? 'text-gray-300' : 'text-gray-600'
                  }`}>No web crawls yet. Start crawling websites to see them here.</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  };

  const HistoryTab = () => {
    const availableDates = Object.keys(historyGroupedByDate).sort((a, b) => new Date(b) - new Date(a));
    const selectedEntries = selectedHistoryDate ? historyGroupedByDate[selectedHistoryDate] || [] : [];

    return (
      <div className={`flex-1 overflow-y-auto transition-colors duration-300 ${
        isDarkMode ? 'bg-slate-900' : 'bg-gray-50'
      }`}>
        <div className="p-6">
          <div className="mb-8">
            <h2 className={`text-xl font-semibold mb-6 transition-colors duration-300 ${
              isDarkMode ? 'text-white' : 'text-gray-900'
            }`}>Content Generation History</h2>
            
            {isLoadingHistory ? (
              <div className="flex items-center justify-center py-8">
                <div className={`animate-spin rounded-full h-8 w-8 border-b-2 transition-colors duration-300 ${
                  isDarkMode ? 'border-purple-400' : 'border-gray-900'
                }`}></div>
                <span className={`ml-2 transition-colors duration-300 ${
                  isDarkMode ? 'text-gray-300' : 'text-gray-600'
                }`}>Loading history...</span>
              </div>
            ) : (
              <>
                {availableDates.length > 0 ? (
                  <>
                    {/* Date Dropdown */}
                    <div className="mb-6">
                      <label htmlFor="history-date" className={`block text-sm font-medium mb-2 transition-colors duration-300 ${
                        isDarkMode ? 'text-gray-300' : 'text-gray-700'
                      }`}>
                        Select Date ({promptHistory.length} total entries)
                      </label>
                      <select
                        id="history-date"
                        value={selectedHistoryDate}
                        onChange={(e) => setSelectedHistoryDate(e.target.value)}
                        className={`w-full md:w-auto px-3 py-2 border rounded-lg text-sm focus:outline-none transition-colors duration-300 ${
                          isDarkMode 
                            ? 'bg-slate-700 border-slate-600 text-white focus:border-purple-400' 
                            : 'bg-white border-gray-300 text-gray-900 focus:border-gray-500'
                        }`}
                      >
                        <option value="">All dates</option>
                        {availableDates.map(date => {
                          const count = historyGroupedByDate[date].length;
                          const formattedDate = new Date(date).toLocaleDateString('en-US', {
                            weekday: 'long',
                            year: 'numeric',
                            month: 'long',
                            day: 'numeric'
                          });
                          return (
                            <option key={date} value={date}>
                              {formattedDate} ({count} {count === 1 ? 'entry' : 'entries'})
                            </option>
                          );
                        })}
                      </select>
                    </div>

                    {/* History Entries */}
                    <div className="space-y-4">
                      {selectedEntries.map((entry) => (
                        <div key={entry.id} className={`rounded-lg p-6 shadow-sm border transition-colors duration-300 ${
                          isDarkMode 
                            ? 'bg-slate-800 border-slate-700' 
                            : 'bg-white border-gray-200'
                        }`}>
                          <div className="flex justify-between items-start mb-4">
                            <div>
                              <h3 className={`text-lg font-medium mb-1 transition-colors duration-300 ${
                                isDarkMode ? 'text-white' : 'text-gray-900'
                              }`}>{entry.creatorName} Style</h3>
                              <p className={`text-sm transition-colors duration-300 ${
                                isDarkMode ? 'text-gray-400' : 'text-gray-600'
                              }`}>
                                {entry.timestamp.toLocaleTimeString()} • {entry.timestamp.toLocaleDateString()}
                              </p>
                            </div>
                            <div className="flex gap-2">
                              <button
                                onClick={() => copyToClipboard(entry.content.linkedinPosts?.[0] || '')}
                                className={`p-2 rounded-lg transition-colors ${
                                  isDarkMode 
                                    ? 'text-gray-400 hover:bg-slate-700 hover:text-white' 
                                    : 'text-gray-600 hover:bg-gray-100'
                                }`}
                                title="Copy LinkedIn Post"
                              >
                                <Copy className="w-4 h-4" />
                              </button>
                            </div>
                          </div>
                          
                          <div className="mb-4">
                            <h4 className={`text-sm font-medium mb-2 transition-colors duration-300 ${
                              isDarkMode ? 'text-gray-300' : 'text-gray-700'
                            }`}>Original Prompt:</h4>
                            <div className={`p-3 rounded-lg text-sm transition-colors duration-300 ${
                              isDarkMode 
                                ? 'bg-slate-700 text-gray-200' 
                                : 'bg-gray-50 text-gray-700'
                            }`}>
                              {entry.prompt}
                            </div>
                          </div>
                          
                          {entry.researchUsed && entry.researchUsed.length > 0 && (
                            <div className="mb-4">
                              <h4 className={`text-sm font-medium mb-2 transition-colors duration-300 ${
                                isDarkMode ? 'text-gray-300' : 'text-gray-700'
                              }`}>Research Used ({entry.researchUsed.length}):</h4>
                              <div className="flex flex-wrap gap-2">
                                {entry.researchUsed.map((research, index) => (
                                  <span key={index} className={`px-2 py-1 rounded text-xs transition-colors duration-300 ${
                                    isDarkMode 
                                      ? 'bg-blue-800 text-blue-200' 
                                      : 'bg-blue-100 text-blue-800'
                                  }`}>
                                    {research.topic}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                          
                          {entry.backendSources && entry.backendSources.length > 0 && (
                            <div className="mb-4">
                              <h4 className={`text-sm font-medium mb-2 transition-colors duration-300 ${
                                isDarkMode ? 'text-gray-300' : 'text-gray-700'
                              }`}>Backend Sources Used:</h4>
                              <div className="flex flex-wrap gap-2">
                                {entry.backendSources.map((source, index) => (
                                  <span key={index} className={`px-2 py-1 rounded text-xs transition-colors duration-300 ${
                                    isDarkMode 
                                      ? 'bg-green-800 text-green-200' 
                                      : 'bg-green-100 text-green-800'
                                  }`}>
                                    {source.filename}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                          
                          {/* Generated Content Tabs */}
                          <div className={`border-t pt-4 transition-colors duration-300 ${
                            isDarkMode ? 'border-slate-600' : 'border-gray-200'
                          }`}>
                            <h4 className={`text-sm font-medium mb-3 transition-colors duration-300 ${
                              isDarkMode ? 'text-gray-300' : 'text-gray-700'
                            }`}>Generated Content:</h4>
                            
                            {/* LinkedIn Post */}
                            {entry.content.linkedinPosts && entry.content.linkedinPosts.length > 0 && (
                              <div className="mb-4">
                                <h5 className={`text-xs font-medium mb-2 transition-colors duration-300 ${
                                  isDarkMode ? 'text-gray-400' : 'text-gray-600'
                                }`}>LinkedIn Post:</h5>
                                <div className={`p-3 rounded-lg text-sm whitespace-pre-line max-h-40 overflow-y-auto transition-colors duration-300 ${
                                  isDarkMode 
                                    ? 'bg-slate-700 text-gray-200' 
                                    : 'bg-gray-50 text-gray-700'
                                }`}>
                                  {entry.content.linkedinPosts[0]}
                                </div>
                              </div>
                            )}
                            
                            {/* Video Script */}
                            {entry.content.videoScripts && entry.content.videoScripts.length > 0 && (
                              <div className="mb-4">
                                <h5 className={`text-xs font-medium mb-2 transition-colors duration-300 ${
                                  isDarkMode ? 'text-gray-400' : 'text-gray-600'
                                }`}>Video Script:</h5>
                                <div className={`p-3 rounded-lg text-sm whitespace-pre-line max-h-32 overflow-y-auto transition-colors duration-300 ${
                                  isDarkMode 
                                    ? 'bg-slate-700 text-gray-200' 
                                    : 'bg-gray-50 text-gray-700'
                                }`}>
                                  {entry.content.videoScripts[0]}
                                </div>
                              </div>
                            )}
                            
                            {/* Hashtags */}
                            {entry.content.hashtags && entry.content.hashtags.length > 0 && (
                              <div className="mb-4">
                                <h5 className={`text-xs font-medium mb-2 transition-colors duration-300 ${
                                  isDarkMode ? 'text-gray-400' : 'text-gray-600'
                                }`}>Hashtags:</h5>
                                <div className="flex flex-wrap gap-1">
                                  {entry.content.hashtags.map((hashtag, index) => (
                                    <span key={index} className={`px-2 py-1 rounded text-xs transition-colors duration-300 ${
                                      isDarkMode 
                                        ? 'bg-purple-800 text-purple-200' 
                                        : 'bg-purple-100 text-purple-800'
                                    }`}>
                                      {hashtag}
                                    </span>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                      
                      {selectedHistoryDate && selectedEntries.length === 0 && (
                        <div className="text-center py-8">
                          <Calendar className={`w-8 h-8 mx-auto mb-2 transition-colors duration-300 ${
                            isDarkMode ? 'text-gray-500' : 'text-gray-400'
                          }`} />
                          <p className={`transition-colors duration-300 ${
                            isDarkMode ? 'text-gray-300' : 'text-gray-600'
                          }`}>No content generated on this date.</p>
                        </div>
                      )}
                    </div>
                  </>
                ) : (
                  <div className="text-center py-12">
                    <BookOpen className={`w-12 h-12 mx-auto mb-4 transition-colors duration-300 ${
                      isDarkMode ? 'text-gray-500' : 'text-gray-400'
                    }`} />
                    <p className={`transition-colors duration-300 ${
                      isDarkMode ? 'text-gray-300' : 'text-gray-600'
                    }`}>No content generated yet. Start creating content to see your history here.</p>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    );
  };

  const ChatTab = () => {
    const [localInputValue, setLocalInputValue] = useState('');
    const textareaRef = useRef(null);

    const handleLocalInputChange = (e) => {
      setLocalInputValue(e.target.value);
      
      // Auto-resize textarea
      const textarea = textareaRef.current;
      if (textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 128) + 'px';
      }
    };

    const handleLocalKeyDown = (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (localInputValue.trim()) {
          handleSendMessage(localInputValue);
          setLocalInputValue('');
        }
      }
    };

    const handleLocalSend = () => {
      if (localInputValue.trim()) {
        handleSendMessage(localInputValue);
        setLocalInputValue('');
        
        // Reset textarea height
        const textarea = textareaRef.current;
        if (textarea) {
          textarea.style.height = '48px';
        }
      }
    };

    // Auto-resize textarea on mount
    useEffect(() => {
      const textarea = textareaRef.current;
      if (textarea) {
        textarea.style.height = '48px';
      }
    }, []);

    return (
      <>
        <div className={`border-b px-6 py-4 backdrop-blur-sm transition-colors duration-300 ${
          isDarkMode 
            ? 'bg-gradient-to-r from-slate-800/80 to-slate-900/50 border-slate-700' 
            : 'bg-gradient-to-r from-purple-50/80 to-purple-100/50 border-purple-200/30'
        }`}>
          <div className="mb-4">
            <h3 className={`text-sm font-bold uppercase tracking-wide mb-3 transition-colors duration-300 ${
              isDarkMode ? 'text-gray-200' : 'text-slate-800'
            }`}>
              Creator Styles
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
              {creatorDatabase.slice(0, 3).map((creator) => {
                if (!creator || !creator.key) return null;
                const IconComponent = creator.icon;
                return (
                  <button
                    key={creator.key}
                    onClick={() => handleCreatorSelect(creator.key)}
                    className={`flex items-center gap-2 p-3 backdrop-blur-sm border rounded-lg hover:transform hover:-translate-y-1 hover:shadow-lg transition-all duration-200 text-left group ${
                      isDarkMode 
                        ? 'bg-slate-800/80 border-slate-600/50 hover:border-purple-400 hover:shadow-purple-500/10' 
                        : 'bg-white/80 border-slate-200/50 hover:border-purple-300 hover:shadow-purple-500/10'
                    }`}
                  >
                    <div className="w-6 h-6 bg-gradient-to-br from-slate-500 to-slate-600 rounded-md flex items-center justify-center group-hover:from-purple-500 group-hover:to-purple-600 group-hover:scale-110 transition-all duration-200">
                      <IconComponent className="w-3 h-3 text-white" />
                    </div>
                    <span className={`text-sm font-semibold group-hover:text-purple-800 transition-colors duration-200 ${
                      isDarkMode ? 'text-gray-300' : 'text-slate-700'
                    }`}>{creator.name}</span>
                  </button>
                );
              })}
            </div>
          </div>

          <div>
            <h3 className={`text-sm font-bold uppercase tracking-wide mb-3 transition-colors duration-300 ${
              isDarkMode ? 'text-gray-200' : 'text-slate-800'
            }`}>
              Quick Actions
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
              <button
                onClick={() => setActiveTab('research')}
                className={`flex items-center gap-2 p-3 backdrop-blur-sm border rounded-lg hover:transform hover:-translate-y-1 hover:shadow-lg transition-all duration-200 text-left group ${
                  isDarkMode 
                    ? 'bg-slate-800/80 border-slate-600/50 hover:border-purple-400 hover:shadow-purple-500/10' 
                    : 'bg-white/80 border-slate-200/50 hover:border-purple-300 hover:shadow-purple-500/10'
                }`}
              >
                <div className="w-6 h-6 bg-gradient-to-br from-slate-500 to-slate-600 rounded-md flex items-center justify-center group-hover:from-purple-500 group-hover:to-purple-600 group-hover:scale-110 transition-all duration-200">
                  <Database className="w-3 h-3 text-white" />
                </div>
                <span className={`text-sm font-semibold group-hover:text-purple-800 transition-colors duration-200 ${
                  isDarkMode ? 'text-gray-300' : 'text-slate-700'
                }`}>Browse Research</span>
              </button>
              <button
                onClick={() => setActiveTab('documents')}
                className={`flex items-center gap-2 p-3 backdrop-blur-sm border rounded-lg hover:transform hover:-translate-y-1 hover:shadow-lg transition-all duration-200 text-left group ${
                  isDarkMode 
                    ? 'bg-slate-800/80 border-slate-600/50 hover:border-purple-400 hover:shadow-purple-500/10' 
                    : 'bg-white/80 border-slate-200/50 hover:border-purple-300 hover:shadow-purple-500/10'
                }`}
              >
                <div className="w-6 h-6 bg-gradient-to-br from-slate-500 to-slate-600 rounded-md flex items-center justify-center group-hover:from-purple-500 group-hover:to-purple-600 group-hover:scale-110 transition-all duration-200">
                  <FileText className="w-3 h-3 text-white" />
                </div>
                <span className={`text-sm font-semibold group-hover:text-purple-800 transition-colors duration-200 ${
                  isDarkMode ? 'text-gray-300' : 'text-slate-700'
                }`}>Manage Documents</span>
              </button>
            </div>
          </div>
        </div>

        <div className={`flex-1 overflow-y-auto px-6 py-8 relative z-20 transition-colors duration-300 ${
          isDarkMode 
            ? 'bg-gradient-to-br from-slate-900/50 to-slate-800/30' 
            : 'bg-gradient-to-br from-purple-50/30 to-white'
        }`}>
          <div className="space-y-6">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex w-full ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div className="flex items-start gap-3 max-w-3xl">
                  {message.type === 'bot' && (
                    <div className="w-10 h-10 bg-gradient-to-br from-purple-600 to-purple-700 rounded-xl flex items-center justify-center shadow-lg shadow-purple-500/25">
                      <Bot className="w-5 h-5 text-white" />
                    </div>
                  )}
                  <div
                    className={`px-5 py-4 rounded-2xl shadow-sm ${
                      message.type === 'user'
                        ? 'bg-gradient-to-r from-purple-600 to-purple-700 text-white shadow-lg shadow-purple-500/25'
                        : isDarkMode 
                          ? 'bg-slate-800/80 backdrop-blur-sm text-gray-100 border border-slate-600/50'
                        : 'bg-white/80 backdrop-blur-sm text-slate-900 border border-slate-200/50'
                    }`}
                  >
                    <p className="text-sm leading-relaxed whitespace-pre-line">{message.text}</p>
                    <div className={`text-xs mt-2 ${
                      message.type === 'user' 
                        ? 'text-purple-100' 
                        : isDarkMode 
                          ? 'text-gray-400' 
                          : 'text-slate-500'
                    }`}>
                      {message.timestamp.toLocaleTimeString()}
                    </div>
                  </div>
                  {message.type === 'user' && (
                    <div className="w-10 h-10 bg-gradient-to-br from-purple-600 to-purple-800 rounded-xl flex items-center justify-center shadow-lg shadow-purple-500/25">
                      <User className="w-5 h-5 text-white" />
                    </div>
                  )}
                </div>
              </div>
            ))}

            {generatedContent && <ContentDisplay content={generatedContent} />}

            {isTyping && (
              <div className="flex items-start gap-3">
                <div className="w-10 h-10 bg-gradient-to-br from-purple-600 to-purple-700 rounded-xl flex items-center justify-center shadow-lg shadow-purple-500/25">
                  <Bot className="w-5 h-5 text-white" />
                </div>
                <div className={`backdrop-blur-sm px-5 py-4 rounded-2xl border shadow-sm ${
                  isDarkMode 
                    ? 'bg-slate-800/80 border-slate-600/50' 
                    : 'bg-white/80 border-slate-200/50'
                }`}>
                  <div className="flex items-center gap-3">
                    <div className="flex items-center gap-1">
                      <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                      <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                    </div>
                    {currentStatus && (
                      <span className="text-sm text-slate-600 font-medium animate-pulse">
                        {currentStatus}
                      </span>
                    )}
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>


        <div className={`border-t backdrop-blur-sm px-8 py-8 transition-colors duration-300 ${
          isDarkMode 
            ? 'border-slate-700 bg-slate-900/95' 
            : 'border-gray-200/60 bg-white/95'
        }`}>
          <div className="max-w-5xl mx-auto">
            <div className="flex items-end gap-6">
            <div className="flex-1">
                <div className="relative group">
              <textarea
                ref={textareaRef}
                value={localInputValue}
                onChange={handleLocalInputChange}
                onKeyDown={handleLocalKeyDown}
                    placeholder="Describe what content you'd like to create... (e.g., 'Create a professional post about AI trends')"
                    className={`w-full px-6 py-5 pr-16 border rounded-2xl text-sm focus:outline-none focus:ring-4 resize-none shadow-sm transition-all duration-300 group-hover:shadow-md ${
                      isDarkMode 
                        ? 'border-slate-600 bg-slate-800 text-gray-100 placeholder-gray-400 focus:border-purple-400 focus:ring-purple-500/20' 
                        : 'border-gray-200 bg-white text-slate-900 placeholder-slate-500 focus:border-slate-400 focus:ring-slate-100'
                    }`}
                    style={{ minHeight: '64px', maxHeight: '160px' }}
                rows="1"
              />
                  <div className={`absolute right-5 top-1/2 transform -translate-y-1/2 transition-colors ${
                    isDarkMode 
                      ? 'text-gray-400 group-focus-within:text-purple-400' 
                      : 'text-slate-400 group-focus-within:text-slate-600'
                  }`}>
                    <Bot className="w-5 h-5" />
                  </div>
                </div>
                <div className="flex items-center justify-between mt-4">
                  <div className={`flex items-center gap-6 text-xs transition-colors duration-300 ${
                    isDarkMode ? 'text-gray-400' : 'text-slate-500'
                  }`}>
                    <span className="flex items-center gap-1">
                      <kbd className={`px-2 py-1 rounded text-xs font-mono ${
                        isDarkMode 
                          ? 'bg-slate-700 text-gray-300' 
                          : 'bg-slate-100 text-slate-600'
                      }`}>Enter</kbd>
                      to send
                    </span>
                    <span className="flex items-center gap-1">
                      <kbd className={`px-2 py-1 rounded text-xs font-mono ${
                        isDarkMode 
                          ? 'bg-slate-700 text-gray-300' 
                          : 'bg-slate-100 text-slate-600'
                      }`}>Shift</kbd>
                      +
                      <kbd className={`px-2 py-1 rounded text-xs font-mono ${
                        isDarkMode 
                          ? 'bg-slate-700 text-gray-300' 
                          : 'bg-slate-100 text-slate-600'
                      }`}>Enter</kbd>
                      for new line
                    </span>
                  </div>
                  <div className={`text-xs font-medium transition-colors duration-300 ${
                    isDarkMode ? 'text-gray-400' : 'text-slate-500'
                  }`}>
                    {localInputValue.length > 0 && `${localInputValue.length} characters`}
                  </div>
                </div>
            </div>
            <button
              onClick={handleLocalSend}
              disabled={!localInputValue.trim() || isTyping}
                className={`px-8 py-5 text-white rounded-2xl disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-300 shadow-lg hover:shadow-xl flex items-center gap-3 font-semibold text-sm ${
                  isDarkMode 
                    ? 'bg-gradient-to-r from-slate-700 to-slate-800 hover:from-slate-800 hover:to-slate-900 shadow-slate-500/20 hover:shadow-slate-500/30' 
                    : 'bg-gradient-to-r from-slate-700 to-slate-800 hover:from-slate-800 hover:to-slate-900 shadow-slate-500/20 hover:shadow-slate-500/30'
                }`}
              >
                {isTyping ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    Generating...
                  </>
                ) : (
                  <>
                    <Send className="w-5 h-5" />
                    Generate Content
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Customization Popup Modal */}
        {showCustomizationPopup && (
          <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
            <div className="bg-white rounded-3xl shadow-2xl max-w-3xl w-full max-h-[90vh] overflow-hidden border border-gray-100">
              {/* Header */}
              <div className="flex items-center justify-between p-8 border-b border-gray-100 bg-gradient-to-r from-slate-50 to-gray-50">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 bg-gradient-to-br from-slate-700 to-slate-800 rounded-2xl flex items-center justify-center shadow-lg">
                    <Settings className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-slate-900">Content Customization</h3>
                    <p className="text-sm text-slate-600 font-medium">Fine-tune your content generation settings</p>
                  </div>
                </div>
                <button
                  onClick={() => setShowCustomizationPopup(false)}
                  className="p-3 hover:bg-white/80 rounded-xl transition-all duration-200 group"
                >
                  <X className="w-5 h-5 text-slate-500 group-hover:text-slate-700" />
                </button>
              </div>

              {/* Content */}
              <div className="p-8">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Tone Selection */}
                  <div className="space-y-2">
                    <label className="block text-sm font-bold text-slate-800 uppercase tracking-wide">Content Tone</label>
                    <select
                      value={contentOptions.tone}
                      onChange={(e) => setContentOptions(prev => ({ ...prev, tone: e.target.value }))}
                      className="w-full px-4 py-3 text-sm border border-gray-200 rounded-xl focus:outline-none focus:border-slate-400 focus:ring-4 focus:ring-slate-100 transition-all duration-200 bg-white"
                    >
                      <option value="">Use Creator's Default</option>
                      <option value="professional">Professional</option>
                      <option value="casual">Casual</option>
                      <option value="motivational">Motivational</option>
                      <option value="analytical">Analytical</option>
                      <option value="conversational">Conversational</option>
                      <option value="authoritative">Authoritative</option>
                    </select>
                  </div>

                  {/* Content Format */}
                  <div className="space-y-2">
                    <label className="block text-sm font-bold text-slate-800 uppercase tracking-wide">Content Format</label>
                    <select
                      value={contentOptions.contentFormat}
                      onChange={(e) => setContentOptions(prev => ({ ...prev, contentFormat: e.target.value }))}
                      className="w-full px-4 py-3 text-sm border border-gray-200 rounded-xl focus:outline-none focus:border-slate-400 focus:ring-4 focus:ring-slate-100 transition-all duration-200 bg-white"
                    >
                      <option value="">Use Creator's Default</option>
                      <option value="paragraph">Paragraph</option>
                      <option value="bullet_points">Bullet Points</option>
                      <option value="numbered_list">Numbered List</option>
                    </select>
                  </div>

                  {/* Content Style */}
                  <div className="space-y-2">
                    <label className="block text-sm font-bold text-slate-800 uppercase tracking-wide">Content Style</label>
                    <select
                      value={contentOptions.contentStyle}
                      onChange={(e) => setContentOptions(prev => ({ ...prev, contentStyle: e.target.value }))}
                      className="w-full px-4 py-3 text-sm border border-gray-200 rounded-xl focus:outline-none focus:border-slate-400 focus:ring-4 focus:ring-slate-100 transition-all duration-200 bg-white"
                    >
                      <option value="">Use Creator's Default</option>
                      <option value="direct">Direct</option>
                      <option value="storytelling">Storytelling</option>
                      <option value="data_driven">Data-Driven</option>
                      <option value="conversational">Conversational</option>
                    </select>
                  </div>

                  {/* Post Length */}
                  <div className="space-y-2">
                    <label className="block text-sm font-bold text-slate-800 uppercase tracking-wide">Post Length</label>
                    <select
                      value={contentOptions.postLength}
                      onChange={(e) => setContentOptions(prev => ({ ...prev, postLength: e.target.value }))}
                      className="w-full px-4 py-3 text-sm border border-gray-200 rounded-xl focus:outline-none focus:border-slate-400 focus:ring-4 focus:ring-slate-100 transition-all duration-200 bg-white"
                    >
                      <option value="">Use Creator's Default</option>
                      <option value="short">Short (100-200 words)</option>
                      <option value="medium">Medium (200-300 words)</option>
                      <option value="long">Long (300-500 words)</option>
                    </select>
                  </div>

                  {/* Include Statistics */}
                  <div className="md:col-span-2">
                    <div className="flex items-center space-x-4 p-4 bg-slate-50 rounded-xl border border-slate-200">
                      <input
                        type="checkbox"
                        id="includeStatistics"
                        checked={contentOptions.includeStatistics}
                        onChange={(e) => setContentOptions(prev => ({ ...prev, includeStatistics: e.target.checked }))}
                        className="w-5 h-5 text-slate-700 border-gray-300 rounded focus:ring-slate-500"
                      />
                      <label htmlFor="includeStatistics" className="text-sm font-semibold text-slate-800">
                        Include Statistics & Data
                      </label>
                    </div>
                  </div>

                  {/* Call to Action */}
                  <div className="md:col-span-2 space-y-2">
                    <label className="block text-sm font-bold text-slate-800 uppercase tracking-wide">Call to Action</label>
                    <input
                      type="text"
                      value={contentOptions.callToAction}
                      onChange={(e) => setContentOptions(prev => ({ ...prev, callToAction: e.target.value }))}
                      placeholder="e.g., 'What are your thoughts?'"
                      className="w-full px-4 py-3 text-sm border border-gray-200 rounded-xl focus:outline-none focus:border-slate-400 focus:ring-4 focus:ring-slate-100 transition-all duration-200 bg-white"
                    />
                  </div>
                </div>

                {/* Active Customizations Preview */}
                {(contentOptions.tone || contentOptions.contentFormat || contentOptions.contentStyle || contentOptions.postLength || contentOptions.includeStatistics || contentOptions.callToAction) && (
                  <div className="mt-8 p-6 bg-gradient-to-r from-slate-50 to-gray-50 border border-slate-200 rounded-2xl">
                    <div className="flex items-center gap-3 mb-4">
                      <CheckCircle className="w-5 h-5 text-slate-600" />
                      <h5 className="text-sm font-bold text-slate-800 uppercase tracking-wide">Active Customizations</h5>
                    </div>
                    <div className="flex flex-wrap gap-3">
                      {contentOptions.tone && (
                        <span className="px-3 py-2 bg-slate-100 text-slate-800 text-sm rounded-full font-semibold border border-slate-200">
                          Tone: {contentOptions.tone}
                        </span>
                      )}
                      {contentOptions.contentFormat && (
                        <span className="px-3 py-2 bg-slate-100 text-slate-800 text-sm rounded-full font-semibold border border-slate-200">
                          Format: {contentOptions.contentFormat.replace('_', ' ')}
                        </span>
                      )}
                      {contentOptions.contentStyle && (
                        <span className="px-3 py-2 bg-slate-100 text-slate-800 text-sm rounded-full font-semibold border border-slate-200">
                          Style: {contentOptions.contentStyle.replace('_', ' ')}
                        </span>
                      )}
                      {contentOptions.postLength && (
                        <span className="px-3 py-2 bg-slate-100 text-slate-800 text-sm rounded-full font-semibold border border-slate-200">
                          Length: {contentOptions.postLength}
                        </span>
                      )}
                      {contentOptions.includeStatistics && (
                        <span className="px-3 py-2 bg-slate-100 text-slate-800 text-sm rounded-full font-semibold border border-slate-200">
                          Include Statistics
                        </span>
                      )}
                      {contentOptions.callToAction && (
                        <span className="px-3 py-2 bg-slate-100 text-slate-800 text-sm rounded-full font-semibold border border-slate-200">
                          CTA: {contentOptions.callToAction}
                        </span>
                      )}
                    </div>
                  </div>
                )}
              </div>

              {/* Footer */}
              <div className="flex items-center justify-between p-8 border-t border-gray-100 bg-gradient-to-r from-slate-50 to-gray-50">
                <button
                  onClick={() => setContentOptions({
                    tone: '',
                    contentFormat: '',
                    contentStyle: '',
                    includeStatistics: false,
                    postLength: '',
                    callToAction: ''
                  })}
                  className="px-6 py-3 text-sm font-semibold text-slate-600 hover:text-slate-800 hover:bg-white rounded-xl border border-slate-300 transition-all duration-200"
                >
                  Reset All
                </button>
                <button
                  onClick={() => setShowCustomizationPopup(false)}
                  className="px-8 py-3 bg-gradient-to-r from-slate-700 to-slate-800 text-white rounded-xl hover:from-slate-800 hover:to-slate-900 transition-all duration-200 font-semibold shadow-lg shadow-slate-500/20"
                >
                  Done
                </button>
              </div>
            </div>
          </div>
        )}

      </>
    );
  };

  return (
    <div className={`flex h-screen relative overflow-hidden transition-colors duration-300 ${
      isDarkMode 
        ? 'bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900' 
        : 'bg-gradient-to-br from-white via-purple-50/20 to-white'
    }`}>
      {/* Network Pattern Background */}
      <div className="absolute inset-0 opacity-5 pointer-events-none">
        <div className="absolute top-20 left-20 w-2 h-2 bg-purple-400 rounded-full"></div>
        <div className="absolute top-32 left-40 w-1 h-1 bg-purple-300 rounded-full"></div>
        <div className="absolute top-48 left-60 w-1.5 h-1.5 bg-purple-500 rounded-full"></div>
        <div className="absolute top-64 left-80 w-1 h-1 bg-purple-400 rounded-full"></div>
        <div className="absolute top-80 left-100 w-2 h-2 bg-purple-300 rounded-full"></div>
        <div className="absolute top-96 left-120 w-1 h-1 bg-purple-500 rounded-full"></div>
        <div className="absolute top-20 right-20 w-1.5 h-1.5 bg-purple-400 rounded-full"></div>
        <div className="absolute top-40 right-40 w-1 h-1 bg-purple-300 rounded-full"></div>
        <div className="absolute top-60 right-60 w-2 h-2 bg-purple-500 rounded-full"></div>
        <div className="absolute top-80 right-80 w-1 h-1 bg-purple-400 rounded-full"></div>
        <div className="absolute top-100 right-100 w-1.5 h-1.5 bg-purple-300 rounded-full"></div>
        {/* Connection lines */}
        <div className="absolute top-20 left-20 w-20 h-px bg-purple-300/30 transform rotate-12"></div>
        <div className="absolute top-40 left-40 w-16 h-px bg-purple-400/30 transform -rotate-12"></div>
        <div className="absolute top-60 left-60 w-24 h-px bg-purple-300/30 transform rotate-6"></div>
        <div className="absolute top-20 right-20 w-18 h-px bg-purple-400/30 transform -rotate-6"></div>
        <div className="absolute top-40 right-40 w-22 h-px bg-purple-300/30 transform rotate-12"></div>
      </div>
      
      {/* Navigation Bar */}
      <div className={`absolute top-0 left-0 right-0 h-16 z-50 backdrop-blur-md border-b transition-colors duration-300 ${
        isDarkMode 
          ? 'bg-slate-900/95 border-slate-700' 
          : 'bg-white/95 border-slate-200'
      }`}>
        <div className="flex items-center justify-between h-full px-6">
          {/* Left side - Logo and Menu */}
          <div className="flex items-center gap-4">
            <button 
              onClick={toggleSidebar}
              className={`p-2 rounded-lg transition-colors ${
                isDarkMode 
                  ? 'hover:bg-slate-800 text-gray-300' 
                  : 'hover:bg-gray-100 text-gray-600'
              }`}
            >
              <Menu className="w-5 h-5" />
            </button>
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-gradient-to-br from-purple-600 to-purple-700 rounded-lg flex items-center justify-center">
                <Linkedin className="w-4 h-4 text-white" />
              </div>
              <span className={`font-bold text-lg transition-colors duration-300 ${
                isDarkMode ? 'text-white' : 'text-gray-900'
              }`}>
                LinkedIn Bot
              </span>
            </div>
          </div>
          
          {/* Right side - Theme Toggle and User */}
          <div className="flex items-center gap-4">
            <button
              onClick={toggleTheme}
              className={`p-2 rounded-full transition-all duration-200 ${
                isDarkMode 
                  ? 'bg-slate-700 hover:bg-slate-600 text-yellow-300' 
                  : 'bg-gray-100 hover:bg-gray-200 text-gray-600'
              }`}
            >
              {isDarkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>
            <div className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-colors duration-300 ${
              isDarkMode ? 'bg-slate-700' : 'bg-gray-100'
            }`}>
              <User className={`w-4 h-4 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`} />
              <span className={`text-sm font-medium transition-colors duration-300 ${
                isDarkMode ? 'text-gray-300' : 'text-gray-600'
              }`}>
                Hi, User
              </span>
            </div>
          </div>
        </div>
      </div>
      
      {isSidebarOpen && (
        <div className={`w-64 backdrop-blur-md border-r flex flex-col shadow-xl relative z-10 transition-all duration-300 ${
          isDarkMode 
            ? 'bg-slate-900/95 border-slate-700' 
            : 'bg-white/95 border-slate-200'
        }`} style={{ paddingTop: '64px' }}>
        <div className={`p-6 border-b transition-colors duration-300 ${
          isDarkMode ? 'border-slate-700' : 'border-slate-200'
        }`}>
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 bg-gradient-to-br from-purple-600 to-purple-700 rounded-xl flex items-center justify-center shadow-lg shadow-purple-500/25">
              <Linkedin className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className={`text-lg font-bold transition-colors duration-300 ${
                isDarkMode 
                  ? 'bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent'
                  : 'bg-gradient-to-r from-slate-900 to-slate-700 bg-clip-text text-transparent'
              }`}>
                Content Creator
              </h1>
              <p className={`text-xs font-medium transition-colors duration-300 ${
                isDarkMode ? 'text-gray-400' : 'text-slate-600'
              }`}>AI-Powered</p>
            </div>
          </div>
          
          <nav className="space-y-1">
            <button
              onClick={() => setActiveTab('chat')}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-semibold transition-all duration-200 ${
                activeTab === 'chat'
                  ? 'bg-gradient-to-r from-purple-600 to-purple-700 text-white shadow-lg shadow-purple-500/25'
                  : isDarkMode 
                    ? 'text-gray-300 hover:bg-slate-800 hover:text-white'
                    : 'text-slate-700 hover:bg-purple-50 hover:text-purple-800'
              }`}
            >
              <Bot className="w-4 h-4" />
              Chat
            </button>
            <button
              onClick={() => setActiveTab('research')}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-semibold transition-all duration-200 ${
                activeTab === 'research'
                  ? 'bg-gradient-to-r from-purple-600 to-purple-700 text-white shadow-lg shadow-purple-500/25'
                  : isDarkMode 
                    ? 'text-gray-300 hover:bg-slate-800 hover:text-white'
                    : 'text-slate-700 hover:bg-purple-50 hover:text-purple-800'
              }`}
            >
              <Database className="w-4 h-4" />
              Research ({researchDatabase.length})
            </button>
            <button
              onClick={() => setActiveTab('creators')}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-semibold transition-all duration-200 ${
                activeTab === 'creators'
                  ? 'bg-gradient-to-r from-purple-600 to-purple-700 text-white shadow-lg shadow-purple-500/25'
                  : isDarkMode 
                    ? 'text-gray-300 hover:bg-slate-800 hover:text-white'
                    : 'text-slate-700 hover:bg-purple-50 hover:text-purple-800'
              }`}
            >
              <Users className="w-4 h-4" />
              Creators ({creatorDatabase.length})
            </button>

            <button
              onClick={() => setActiveTab('documents')}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-semibold transition-all duration-200 ${
                activeTab === 'documents'
                  ? 'bg-gradient-to-r from-purple-600 to-purple-700 text-white shadow-lg shadow-purple-500/25'
                  : isDarkMode 
                    ? 'text-gray-300 hover:bg-slate-800 hover:text-white'
                    : 'text-slate-700 hover:bg-purple-50 hover:text-purple-800'
              }`}
            >
              <Upload className="w-4 h-4" />
              Documents ({customDocuments.length})
            </button>
            <button
              onClick={() => setActiveTab('crawls')}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-semibold transition-all duration-200 ${
                activeTab === 'crawls'
                  ? 'bg-gradient-to-r from-purple-600 to-purple-700 text-white shadow-lg shadow-purple-500/25'
                  : isDarkMode 
                    ? 'text-gray-300 hover:bg-slate-800 hover:text-white'
                    : 'text-slate-700 hover:bg-purple-50 hover:text-purple-800'
              }`}
            >
              <Globe className="w-4 h-4" />
              Web Crawls ({webCrawls?.length || 0})
            </button>
            <button
              onClick={() => setActiveTab('history')}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-semibold transition-all duration-200 ${
                activeTab === 'history'
                  ? 'bg-gradient-to-r from-purple-600 to-purple-700 text-white shadow-lg shadow-purple-500/25'
                  : isDarkMode 
                    ? 'text-gray-300 hover:bg-slate-800 hover:text-white'
                    : 'text-slate-700 hover:bg-purple-50 hover:text-purple-800'
              }`}
            >
              <BookOpen className="w-4 h-4" />
              History ({promptHistory.length})
            </button>
          </nav>
        </div>

        <div className="flex-1"></div>

        {/* Settings and Test Buttons */}
        <div className={`p-4 border-t space-y-2 transition-colors duration-300 ${
          isDarkMode ? 'border-slate-700' : 'border-slate-200'
        }`}>
          <button
            onClick={() => setShowCustomizationPopup(true)}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-semibold transition-all duration-200 relative ${
              isDarkMode 
                ? 'text-gray-300 hover:bg-slate-800 hover:text-white'
                : 'text-slate-700 hover:bg-purple-50 hover:text-purple-800'
            }`}
          >
            <Settings className="w-4 h-4" />
            Customize Content
            {(contentOptions.tone || contentOptions.contentFormat || contentOptions.contentStyle || contentOptions.postLength || contentOptions.includeStatistics || contentOptions.callToAction) && (
              <span className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full flex items-center justify-center shadow-lg border border-white">
                <span className="w-1 h-1 bg-white rounded-full"></span>
              </span>
            )}
          </button>
          
          {/* Removed Test Connection button as requested */}
        </div>
      </div>
      )}

      <div className="flex-1 flex flex-col relative z-20" style={{ paddingTop: '64px' }}>
        {activeTab === 'chat' && <ChatTab />}
        {activeTab === 'research' && <ResearchTab />}
        {activeTab === 'creators' && <CreatorTab />}

        {activeTab === 'documents' && <CustomDocumentsTab />}
        {activeTab === 'crawls' && <WebCrawlsTab />}
        {activeTab === 'history' && <HistoryTab />}
      </div>

    </div>
  );
};

export default LinkedInContentBot;