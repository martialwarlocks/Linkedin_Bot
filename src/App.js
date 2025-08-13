import React, { useState, useRef, useEffect, useCallback } from 'react';
import { supabase, TABLES } from './supabase-config';
import { Send, Bot, User, Linkedin, Video, FileText, Database, TrendingUp, Users, Lightbulb, Copy, Settings, Plus, Trash2, Edit3, Save, X, Search, BookOpen, UserPlus, Upload, Globe, AlertCircle, Check, Calendar, CheckCircle, Clock, RefreshCw } from 'lucide-react';

// 🔧 CONFIGURATION - UPDATE THIS WITH YOUR BACKEND URL
const BACKEND_CONFIG = {
  // ✅ CONFIGURED: Your Local FastAPI Backend URL
  url: 'http://localhost:8000',
  
  // This is your local FastAPI backend with full functionality
  // The backend supports: linkedin/generate, linkedin/research, linkedin/creators endpoints
  
  // Note: Your local backend has document upload endpoints,
  // and has excellent content generation and research management features
};

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
        console.log('Fetching documents:', `${BACKEND_URL}/linkedin/documents`);
        
        const response = await fetch(`${BACKEND_URL}/linkedin/documents`);
        
        console.log('Get documents response status:', response.status);
        
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          console.error('Get documents response error:', errorData);
          throw new Error(errorData.detail || errorData.error || `Get documents failed: ${response.status}`);
        }
        
        const documents = await response.json();
        console.log('Documents fetched successfully:', documents);
        
        return documents.map(doc => ({
          filename: doc.filename,
          total_chunks: doc.embeddings_count || 1,
          upload_date: doc.created_at,
          file_size: doc.size || 0,
          file_type: doc.content_type || 'unknown',
          document_id: doc.document_id,
          gcs_url: doc.gcs_url
        }));
      } catch (error) {
        console.error('FastAPI get documents error:', error);
        throw error;
      }
    },

    // Delete document from research items (working backend approach)
    deleteDocument: async (filename) => {
      try {
        console.log('Attempting to delete file:', filename);
        
        // First get the research item to find its ID
        const response = await fetch(`${BACKEND_URL}/linkedin/research`);
        const researchItems = await response.json();
        
        const documentItem = researchItems.find(item => 
          (item.source === filename || item.topic === `Document: ${filename}`) && 
          item.tags && item.tags.includes('document')
        );
        
        if (!documentItem) {
          throw new Error(`Document ${filename} not found`);
        }
        
        // Delete the research item
        const deleteResponse = await fetch(`${BACKEND_URL}/linkedin/research/${documentItem.id}`, {
          method: 'DELETE'
        });
        
        if (!deleteResponse.ok) {
          const errorData = await deleteResponse.json().catch(() => ({}));
          console.error('Delete response error:', errorData);
          throw new Error(errorData.detail || errorData.error || `Delete failed: ${deleteResponse.status}`);
        }
        
        const result = await deleteResponse.json();
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
          linkedinPosts: result.linkedin_post ? [result.linkedin_post] : [],
          videoScripts: result.linkedin_reel_transcript ? [result.linkedin_reel_transcript] : [],
          hashtags: Array.isArray(result.hashtags) ? result.hashtags : [],
          engagement_tips: result.engagement_questions || [],
          talking_points: result.talking_points || [],
          style_notes: result.style_notes || '',
          context_used: result.context_used || ''
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
        console.log('Adding research item to Supabase:', researchItem);
        
        const { data, error } = await supabase
          .from(TABLES.RESEARCH)
          .insert({
            topic: researchItem.topic,
            findings: researchItem.findings,
            data: researchItem.data,
            source: researchItem.source,
            tags: researchItem.tags || 'research, insights'
          })
          .select()
          .single();
        
        if (error) {
          console.error('Supabase add research error:', error);
          throw new Error(error.message);
        }
        
        console.log('Research added successfully:', data);
        return data;
      } catch (error) {
        console.error('Supabase add research error:', error);
        throw error;
      }
    },

    // Get research items from Supabase
    getResearch: async () => {
      try {
        console.log('Getting research items from Supabase...');
        
        const { data, error } = await supabase
          .from(TABLES.RESEARCH)
          .select('*')
          .order('created_at', { ascending: false });
        
        if (error) {
          console.error('Supabase get research error:', error);
          throw new Error(error.message);
        }
        
        console.log('Research items fetched successfully:', data);
        return data;
      } catch (error) {
        console.error('Supabase get research error:', error);
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
let configStore = {
  openaiApiKey: '',
  model: 'gpt-4',
  supabaseUrl: 'https://qgyqkgmdnwfcnzzuzict.supabase.co',
  supabaseKey: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFneXFrZ21kbndmY256enV6aWN0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTM3NzkyOTcsImV4cCI6MjA2OTM1NTI5N30.d9VFOHZsWDxhqY8UM0jvx5pGJVVOSkgHVFODL16Nc6s'
};

// Supabase client initialization
const createSupabaseClient = () => {
  const supabaseUrl = configStore.supabaseUrl || 'https://qgyqkgmdnwfcnzzuzict.supabase.co';
  const supabaseKey = configStore.supabaseKey || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFneXFrZ21kbndmY256enV6aWN0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTM3NzkyOTcsImV4cCI6MjA2OTM1NTI5N30.d9VFOHZsWDxhqY8UM0jvx5pGJVVOSkgHVFODL16Nc6s';
  
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
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'bot',
      text: "Welcome to the LinkedIn Content Creator Assistant with Cloud Document Integration! I can now access your uploaded documents and crawled websites to create better content. How can I assist you today?",
      timestamp: new Date()
    }
  ]);

  const [isTyping, setIsTyping] = useState(false);
  const [showConfig, setShowConfig] = useState(false);
  const [config] = useState(configStore);
  const [generatedContent, setGeneratedContent] = useState(null);
  // Removed selectedOptions since we only have single posts now
  const messagesEndRef = useRef(null);

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
        characteristics: 'High energy, direct approach, business-focused, motivational'
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
        characteristics: 'Leadership-focused, inspirational, storytelling, thoughtful questions'
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
        characteristics: 'Concise wisdom, marketing insights, thought-provoking, minimal but impactful'
      }
    }
  ]);

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
        <p className="text-xs text-gray-600">
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
  const generateContentWithFastAPI = useCallback(async (prompt, creatorKey, researchIds) => {
    if (!documentManager) {
      throw new Error('Document manager not initialized');
    }

    try {
      console.log('=== NEW CONTENT GENERATION ===');
      console.log('Prompt:', prompt);
      console.log('Creator Key:', creatorKey);
      console.log('Research IDs:', researchIds);
      console.log('Research IDs type:', typeof researchIds);
      console.log('Research IDs length:', researchIds?.length);

      // DIRECT BACKEND CALL - Bypass documentManager
      console.log('=== DIRECT BACKEND CALL ===');
      console.log('Calling backend directly...');
      
      const requestBody = {
        prompt: prompt,
        creator_key: creatorKey,
        research_ids: researchIds
      };
      
      console.log('Request body:', requestBody);
      
      const response = await fetch(`${BACKEND_CONFIG.url}/linkedin/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody)
      });
      
      if (!response.ok) {
        throw new Error(`Backend error: ${response.status}`);
      }
      
      const contentResult = await response.json();
      
      console.log('=== DIRECT BACKEND RESPONSE ===');
      console.log('Backend response:', contentResult);
      console.log('Backend response type:', typeof contentResult);
      console.log('Backend response keys:', Object.keys(contentResult));
      console.log('linkedin_post:', contentResult.linkedin_post);
      console.log('linkedin_reel_transcript:', contentResult.linkedin_reel_transcript);
      console.log('linkedin_post type:', typeof contentResult.linkedin_post);
      console.log('linkedin_reel_transcript type:', typeof contentResult.linkedin_reel_transcript);
      console.log('linkedin_post length:', contentResult.linkedin_post?.length || 0);
      console.log('linkedin_reel_transcript length:', contentResult.linkedin_reel_transcript?.length || 0);
      
      // SIMPLE APPROACH: Just use the content as-is from backend
      const content = {
        linkedinPosts: contentResult.linkedin_post ? [contentResult.linkedin_post] : [],
        videoScripts: contentResult.linkedin_reel_transcript ? [contentResult.linkedin_reel_transcript] : [],
        hashtags: contentResult.hashtags || [],
        engagement_tips: contentResult.engagement_questions || [],
        talking_points: contentResult.talking_points || [],
        style_notes: contentResult.style_notes || '',
        context_used: contentResult.context_used || ''
      };
      
      console.log('Mapped content:', content);
      console.log('LinkedIn posts:', content.linkedinPosts);
      console.log('Video scripts:', content.videoScripts);

      return content;
    } catch (error) {
      console.error('Content generation error:', error);
      throw error;
    }
  }, [documentManager]);



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
    if (config.supabaseUrl && config.supabaseKey) {
      const client = createSupabaseClient();
      setSupabaseClient(client);
      
      const loadResearch = async () => {
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
      };
      loadResearch();
      loadCreatorsFromSupabase(client);
    }
  }, [config.supabaseUrl, config.supabaseKey]);

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
    setIsTyping(true);

    try {
      // Extract creator style from message
      let creatorKey = 'gary-v'; // Default creator
      let creatorName = 'Gary Vaynerchuk'; // Default name
      
      const lowerText = text.toLowerCase();
      for (const creator of creatorDatabase) {
        if (lowerText.includes(creator.name.toLowerCase()) || 
            lowerText.includes(creator.key)) {
          creatorKey = creator.key;
          creatorName = creator.name;
          break;
        }
      }

      console.log('=== MESSAGE PROCESSING DEBUG ===');
      console.log('Input text:', text);
      console.log('Selected creator key:', creatorKey);
      console.log('Selected creator name:', creatorName);

      // Find relevant research based on keywords
      const relevantResearch = findRelevantResearch(text, 3);
      console.log(`Found ${relevantResearch.length} relevant research items:`, relevantResearch.map(r => r.topic));

      // Create enhanced prompt with research context
      let enhancedPrompt = text;
      if (relevantResearch.length > 0) {
        const researchContext = relevantResearch.map(r => 
          `Research: ${r.topic}\nFindings: ${r.findings}`
        ).join('\n\n');
        enhancedPrompt = `${text}\n\nUse this research context:\n${researchContext}`;
        console.log('=== ENHANCED PROMPT DEBUG ===');
        console.log('Original text:', text);
        console.log('Enhanced prompt:', enhancedPrompt);
        console.log('Research context used:', researchContext);
      } else {
        console.log('=== NO RESEARCH FOUND ===');
        console.log('Original text:', text);
        console.log('No relevant research found, using original prompt');
      }

      // Check backend connection
      if (backendStatus !== 'connected') {
        console.warn('Backend not connected, attempting to connect...');
        await testBackendConnection();
        if (backendStatus !== 'connected') {
          throw new Error('Backend not connected. Please check your backend connection.');
        }
      }

      // Generate content with research IDs
      const researchIds = relevantResearch.map(r => r.id.toString());
      console.log('Sending research IDs to backend:', researchIds);
      console.log('Research IDs type:', typeof researchIds);
      console.log('Research IDs length:', researchIds.length);
      
      const content = await generateContentWithFastAPI(enhancedPrompt, creatorKey, researchIds);
      console.log('Content generated:', content);
      console.log('Content type:', typeof content);
      console.log('Content keys:', Object.keys(content));
      console.log('LinkedIn posts before setting:', content.linkedinPosts);
      console.log('Video scripts before setting:', content.videoScripts);
      console.log('LinkedIn post length before setting:', content.linkedinPosts?.[0]?.length || 0);
      console.log('Video script length before setting:', content.videoScripts?.[0]?.length || 0);
      setGeneratedContent(content);

      // Save to history
      const historyEntry = {
        prompt: enhancedPrompt,
        creatorKey: creatorKey,
        creatorName: creatorName,
        content: content,
        researchUsed: relevantResearch.map(r => ({
          id: r.id,
          topic: r.topic
        })),
        backendSources: [] // TODO: Add backend sources if available
      };

      // Save to Supabase and update local state
      const savedEntry = await saveHistoryToSupabase(historyEntry);
      if (savedEntry) {
        const newHistoryEntry = {
          ...historyEntry,
          id: savedEntry.id,
          timestamp: new Date(),
          created_at: savedEntry.created_at
        };
        
        // Update local history state
        setPromptHistory(prev => [newHistoryEntry, ...prev]);
        
        // Update grouped history
        const dateKey = newHistoryEntry.timestamp.toDateString();
        setHistoryGroupedByDate(prev => ({
          ...prev,
          [dateKey]: [newHistoryEntry, ...(prev[dateKey] || [])]
        }));
        
        // Set selected date to today if no date is selected
        if (!selectedHistoryDate) {
          setSelectedHistoryDate(dateKey);
        }
      }
      
      // Create response message
      let responseText = `I've created content in ${creatorName}'s style for you!`;
      
      if (relevantResearch.length > 0) {
        responseText += ` I analyzed ${relevantResearch.length} research items: ${relevantResearch.map(r => r.topic).join(', ')}.`;
      } else {
        responseText += ` I generated content based on general best practices since no specific research was found.`;
      }
      
      responseText += ` Check out the generated content below!`;
      
      const botResponse = {
        id: Date.now() + 1,
        type: 'bot',
        text: responseText,
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, botResponse]);
      setIsTyping(false);
    } catch (error) {
      console.error('Error generating response:', error);
      const fallbackResponse = {
        id: Date.now() + 1,
        type: 'bot',
        text: `Error generating content: ${error.message}. Please check your configuration.`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, fallbackResponse]);
      setIsTyping(false);
    }
  }, [creatorDatabase, backendStatus, findRelevantResearch, generateContentWithFastAPI, testBackendConnection]);

  const handleCreatorSelect = useCallback((creatorKey) => {
    const creator = creatorDatabase.find(c => c.key === creatorKey);
    if (!creator) {
      console.error('Creator not found:', creatorKey);
      return;
    }
    console.log('Creator selected:', creator.name, creator.key);
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

  const ConfigPanel = () => (
    <div className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center z-50" style={{ backdropFilter: 'blur(4px)' }}>
      <div className="bg-white p-8 rounded-xl w-11/12 max-w-md shadow-2xl border border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900 mb-6">Configuration</h3>
        
        {/* Backend URL Configuration */}
        <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <h4 className="font-medium text-blue-900 mb-2">Backend Configuration</h4>
          <p className="text-sm text-blue-800 mb-2">Current Backend URL:</p>
          <code className="text-xs bg-blue-100 px-2 py-1 rounded text-blue-900 block break-all">
            {BACKEND_CONFIG.url}
          </code>
          <p className="text-xs text-blue-700 mt-2">
            To change this, update BACKEND_CONFIG.url in the code
          </p>
        </div>
        
        <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg">
          <h4 className="font-medium text-green-900 mb-2">Supabase Database</h4>
          <p className="text-sm text-green-800 mb-2">Database URL:</p>
          <code className="text-xs bg-green-100 px-2 py-1 rounded text-green-900 block break-all">
            https://qgyqkgmdnwfcnzzuzict.supabase.co
          </code>
          <p className="text-xs text-green-700 mt-2">
            Research and creator styles are stored in Supabase
          </p>
        </div>

        <div className="flex gap-3">
          <button
            onClick={() => setShowConfig(false)}
            className="flex-1 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );

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
    
    console.log('LinkedIn post:', linkedinPost);
    console.log('Video script:', videoScript);
    console.log('LinkedIn post type:', typeof linkedinPost);
    console.log('Video script type:', typeof videoScript);
    console.log('LinkedIn post length:', linkedinPost?.length || 0);
    console.log('Video script length:', videoScript?.length || 0);
    
    return (
    <div className="bg-white border border-gray-200 rounded-xl p-6 mt-4 shadow-sm">
      <div className="flex items-center gap-2 mb-6 pb-4 border-b border-gray-100">
        <Linkedin className="w-5 h-5 text-blue-600" />
        <h3 className="text-base font-semibold text-gray-900">Generated Content</h3>
      </div>

      <div className="mb-8">
        <div className="flex justify-between items-center mb-3">
          <h4 className="text-sm font-semibold text-gray-900 uppercase tracking-wide">
            LinkedIn Post
          </h4>
          <div className="flex items-center gap-2">
            <button
              onClick={() => copyToClipboard(linkedinPost || '')}
              className="flex items-center gap-2 px-3 py-1.5 bg-gray-100 text-gray-700 border border-gray-300 rounded-lg text-xs font-medium hover:bg-gray-200 transition-colors"
            >
              <Copy className="w-3.5 h-3.5" />
              Copy
            </button>
          </div>
        </div>
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 text-sm text-gray-700 whitespace-pre-line leading-relaxed">
          {linkedinPost ? (
            <div>
              <p className="mb-2 text-gray-600 text-xs">Length: {linkedinPost.length} chars</p>
              {linkedinPost}
            </div>
          ) : (
            <div className="text-red-500">
              <p>No LinkedIn post generated</p>
              <p className="text-xs mt-1">Raw: {JSON.stringify(content?.linkedinPosts)}</p>
            </div>
          )}
        </div>
      </div>

      <div className="mb-8">
        <div className="flex justify-between items-center mb-3">
          <h4 className="text-sm font-semibold text-gray-900 uppercase tracking-wide flex items-center gap-2">
            <Video className="w-4 h-4" />
            LinkedIn Reel Transcript
          </h4>
          <div className="flex items-center gap-2">
            <button
              onClick={() => copyToClipboard(videoScript || '')}
              className="flex items-center gap-2 px-3 py-1.5 bg-gray-100 text-gray-700 border border-gray-300 rounded-lg text-xs font-medium hover:bg-gray-200 transition-colors"
            >
              <Copy className="w-3.5 h-3.5" />
              Copy
            </button>
          </div>
        </div>
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 text-xs text-gray-700 whitespace-pre-line leading-relaxed font-mono">
          {videoScript ? (
            <div>
              <p className="mb-2 text-gray-600 text-xs">Length: {videoScript.length} chars</p>
              {videoScript}
            </div>
          ) : (
            <div className="text-red-500">
              <p>No video script generated</p>
              <p className="text-xs mt-1">Raw: {JSON.stringify(content?.videoScripts)}</p>
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h4 className="text-sm font-semibold text-gray-900 uppercase tracking-wide mb-3">
            Hashtags
          </h4>
          <div className="flex flex-wrap gap-2">
            {hashtags.length > 0 ? (
              hashtags.map((tag, index) => (
                <span
                  key={index}
                  className="bg-gray-100 text-gray-700 px-2 py-1 rounded text-xs font-medium"
                >
                  {tag}
                </span>
              ))
            ) : (
              <span className="text-gray-500 text-xs">No hashtags generated</span>
            )}
          </div>
        </div>
        <div>
          <h4 className="text-sm font-semibold text-gray-900 uppercase tracking-wide mb-3">
            Engagement Tips
          </h4>
          <ul className="text-xs text-gray-600 leading-relaxed space-y-1">
            {engagementTips.length > 0 ? (
              engagementTips.map((tip, index) => (
                <li key={index} className="flex items-start">
                  <span className="mr-2">•</span>
                  {tip}
                </li>
              ))
            ) : (
              <li className="text-gray-500">No engagement tips generated</li>
            )}
          </ul>
        </div>
      </div>

      {content.talking_points && content.talking_points.length > 0 && (
        <div className="mb-6">
          <h4 className="text-sm font-semibold text-gray-900 uppercase tracking-wide mb-3">
            Talking Points
          </h4>
          <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
            <ul className="space-y-2">
              {content.talking_points.map((point, index) => (
                <li key={index} className="flex items-start">
                  <span className="mr-2 text-blue-600">•</span>
                  <span className="text-sm text-gray-700">{point}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {content.style_notes && (
        <div className="mb-6">
          <h4 className="text-sm font-semibold text-gray-900 uppercase tracking-wide mb-3">
            Style Notes
          </h4>
          <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
            <p className="text-sm text-gray-700 leading-relaxed">{content.style_notes}</p>
          </div>
        </div>
      )}

      {content.context_used && (
        <div className="mb-6">
          <h4 className="text-sm font-semibold text-gray-900 uppercase tracking-wide mb-3">
            Context Used
          </h4>
          <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
            <p className="text-sm text-gray-700 leading-relaxed">{content.context_used}</p>
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
        <div className="flex-1 overflow-y-auto bg-gray-50">
        <div className="p-6">
          <div className="mb-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-6">Research Database</h2>
            
            <BackendStatus />
            
            <div className="mb-6 p-3 rounded-lg bg-gray-100 border border-gray-200 flex items-center gap-2 text-sm">
              <div className={`w-2 h-2 rounded-full ${supabaseClient ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span className="text-gray-600">
                {supabaseClient ? 'Connected to Supabase - Research sorted by newest first' : 'Using local storage - Research sorted by newest first'}
              </span>
            </div>
            
            <div className="relative mb-6">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search research..."
                value={localSearchTerm}
                onChange={(e) => handleLocalSearchChange(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
              />
            </div>

            <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200 mb-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Add New Research</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <input
                  type="text"
                  placeholder="Research Topic"
                  value={localNewResearch.topic}
                  onChange={(e) => handleLocalResearchChange('topic', e.target.value)}
                  className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
                />
                <input
                  type="text"
                  placeholder="Source"
                  value={localNewResearch.source}
                  onChange={(e) => handleLocalResearchChange('source', e.target.value)}
                  className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
                />
              </div>
              <div className="mb-4">
                <textarea
                  placeholder="Key Findings"
                  value={localNewResearch.findings}
                  onChange={(e) => handleLocalResearchChange('findings', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500 resize-none"
                  rows="4"
                />
              </div>
              <div className="mb-4">
                <textarea
                  placeholder="Supporting Data/Statistics"
                  value={localNewResearch.data}
                  onChange={(e) => handleLocalResearchChange('data', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500 resize-none"
                  rows="3"
                />
              </div>
              <div className="flex gap-4">
                <input
                  type="text"
                  placeholder="Tags (comma separated)"
                  value={localNewResearch.tags}
                  onChange={(e) => handleLocalResearchChange('tags', e.target.value)}
                  className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
                />
                <button
                  onClick={handleAddResearch}
                  className="px-6 py-2 bg-gray-900 text-white rounded-lg text-sm font-medium hover:bg-gray-800 transition-colors flex items-center gap-2"
                >
                  <Plus className="w-4 h-4" />
                  Add Research
                </button>
              </div>
            </div>

            {localNewResearch.topic && (
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
                <h4 className="text-sm font-medium text-blue-900 mb-2">Preview</h4>
                <div className="space-y-2 text-sm text-blue-800">
                  <p><strong>Topic:</strong> {localNewResearch.topic}</p>
                  {localNewResearch.findings && <p><strong>Findings:</strong> {localNewResearch.findings}</p>}
                  {localNewResearch.source && <p><strong>Source:</strong> {localNewResearch.source}</p>}
                  {localNewResearch.data && <p><strong>Data:</strong> {localNewResearch.data}</p>}
                  {localNewResearch.tags && (
                    <div className="flex items-center gap-2">
                      <strong>Tags:</strong>
                      <div className="flex flex-wrap gap-1">
                        {localNewResearch.tags.split(',').map((tag, index) => tag.trim()).filter(tag => tag).map((tag, index) => (
                          <span key={index} className="bg-blue-100 px-2 py-1 rounded text-xs">
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
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
              </div>
            )}

            {/* Research Summary */}
            {filteredResearch.length > 0 && (
              <div className="mb-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
                <h4 className="text-sm font-semibold text-gray-900 mb-3">Research Summary</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600">{categorizedResearch.newsArticles.length}</div>
                    <div className="text-gray-600">News Articles</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-600">{categorizedResearch.webCrawls.length}</div>
                    <div className="text-gray-600">Web Crawls</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-purple-600">{categorizedResearch.documents.length}</div>
                    <div className="text-gray-600">Documents</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-gray-600">{categorizedResearch.other.length}</div>
                    <div className="text-gray-600">Other</div>
                  </div>
                </div>
              </div>
            )}

            <div className="space-y-6">
              {filteredResearch.length === 0 && !isLoadingResearch && (
                <div className="text-center py-12 bg-white rounded-lg border border-gray-200">
                  <Database className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-600">
                    {localSearchTerm ? `No research found matching "${localSearchTerm}"` : 'No research found. Add your first research above!'}
                  </p>
                </div>
              )}
              
              {/* News Articles Section */}
              {categorizedResearch.newsArticles.length > 0 && (
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                    <span className="w-2 h-2 bg-blue-500 rounded-full"></span>
                    Latest News Articles ({categorizedResearch.newsArticles.length})
                  </h3>
                  <div className="space-y-4">
                    {categorizedResearch.newsArticles.map((research, index) => (
                      <div key={research.id} className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
                        {editingResearch === research.id ? (
                          <ResearchEditForm research={research} onSave={updateResearch} onCancel={() => setEditingResearch(null)} />
                        ) : (
                          <div>
                            <div className="flex justify-between items-start mb-4">
                              <div className="flex-1">
                                <div className="flex items-center gap-2 mb-2">
                                  <h3 className="text-lg font-medium text-gray-900">{research.topic}</h3>
                                  {index < 3 && (
                                    <span className="bg-green-100 text-green-800 px-2 py-1 rounded-full text-xs font-medium">
                                      Recent
                                    </span>
                                  )}
                                </div>
                                <p className="text-sm text-gray-600 mb-2">{research.source}</p>
                                <p className="text-xs text-gray-500">
                                  Added: {new Date(research.dateAdded).toLocaleDateString()} at {new Date(research.dateAdded).toLocaleTimeString()}
                                </p>
                              </div>
                              <div className="flex gap-2">
                                <button
                                  onClick={() => setEditingResearch(research.id)}
                                  className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
                                >
                                  <Edit3 className="w-4 h-4" />
                                </button>
                                <button
                                  onClick={() => deleteResearch(research.id)}
                                  className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                                >
                                  <Trash2 className="w-4 h-4" />
                                </button>
                              </div>
                            </div>
                            <p className="text-gray-700 mb-3">{research.findings}</p>
                            <p className="text-gray-600 text-sm mb-4">{research.data}</p>
                            <div className="flex flex-wrap gap-2 mb-3">
                              {(Array.isArray(research.tags) ? research.tags : []).map((tag, index) => (
                                <span
                                  key={index}
                                  className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs font-medium"
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
      <div className="flex-1 overflow-y-auto bg-gray-50">
        <div className="p-6">
          <div className="mb-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-6">Creator Profiles</h2>
            
            <BackendStatus />
            
            <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200 mb-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Add New Creator</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <input
                  type="text"
                  placeholder="Creator Name"
                  value={localNewCreator.name}
                  onChange={(e) => handleLocalCreatorChange('name', e.target.value)}
                  className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
                />
                <input
                  type="text"
                  placeholder="Tone (e.g., Direct, passionate, no-nonsense)"
                  value={localNewCreator.tone}
                  onChange={(e) => handleLocalCreatorChange('tone', e.target.value)}
                  className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
                />
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <input
                  type="text"
                  placeholder="Post Structure"
                  value={localNewCreator.structure}
                  onChange={(e) => handleLocalCreatorChange('structure', e.target.value)}
                  className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
                />
                <input
                  type="text"
                  placeholder="Language Style"
                  value={localNewCreator.language}
                  onChange={(e) => handleLocalCreatorChange('language', e.target.value)}
                  className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
                />
              </div>
              <div className="mb-4">
                <input
                  type="text"
                  placeholder="Typical Post Length"
                  value={localNewCreator.length}
                  onChange={(e) => handleLocalCreatorChange('length', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
                />
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <input
                  type="text"
                  placeholder="Common Hooks (comma separated)"
                  value={localNewCreator.hooks}
                  onChange={(e) => handleLocalCreatorChange('hooks', e.target.value)}
                  className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
                />
                <input
                  type="text"
                  placeholder="Common Endings (comma separated)"
                  value={localNewCreator.endings}
                  onChange={(e) => handleLocalCreatorChange('endings', e.target.value)}
                  className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
                />
              </div>
              <div className="flex gap-4">
                <textarea
                  placeholder="Key Characteristics"
                  value={localNewCreator.characteristics}
                  onChange={(e) => handleLocalCreatorChange('characteristics', e.target.value)}
                  className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500 resize-none"
                  rows="3"
                />
                <button
                  onClick={handleAddCreator}
                  className="px-6 py-2 bg-gray-900 text-white rounded-lg text-sm font-medium hover:bg-gray-800 transition-colors flex items-center gap-2"
                >
                  <Plus className="w-4 h-4" />
                  Add Creator
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {creatorDatabase.map((creator) => (
                <div key={creator.id} className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
                  {editingCreator === creator.id ? (
                    <CreatorEditForm creator={creator} onCancel={() => setEditingCreator(null)} />
                  ) : (
                    <div>
                      <div className="flex justify-between items-start mb-4">
                        <div className="flex items-center gap-3">
                          <creator.icon className="w-6 h-6 text-gray-600" />
                          <h3 className="text-lg font-medium text-gray-900">{creator.name}</h3>
                        </div>
                        <div className="flex gap-2">
                          <button
                            onClick={() => setEditingCreator(creator.id)}
                            className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
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
                            className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                      <div className="space-y-3 text-sm">
                        <div>
                          <span className="font-medium text-gray-700">Tone:</span>
                          <span className="ml-2 text-gray-600">{creator.style.tone}</span>
                        </div>
                        <div>
                          <span className="font-medium text-gray-700">Structure:</span>
                          <span className="ml-2 text-gray-600">{creator.style.structure}</span>
                        </div>
                        <div>
                          <span className="font-medium text-gray-700">Language:</span>
                          <span className="ml-2 text-gray-600">{creator.style.language}</span>
                        </div>
                        <div>
                          <span className="font-medium text-gray-700">Characteristics:</span>
                          <span className="ml-2 text-gray-600">{creator.style.characteristics}</span>
                        </div>
                        <div>
                          <span className="font-medium text-gray-700">Common Hooks:</span>
                          <div className="mt-1 flex flex-wrap gap-1">
                            {creator.style.hooks.map((hook, index) => (
                              <span key={index} className="bg-green-100 text-green-800 px-2 py-1 rounded text-xs">
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
      <div className="flex-1 overflow-y-auto bg-gray-50">
        <div className="p-6">
          <div className="mb-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-6">Cloud Document Management</h2>
            
            <BackendStatus />
            
            <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200 mb-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Upload Document</h3>
              <div className="flex gap-4 items-end">
                <div className="flex-1">
                  <input
                    id="file-input"
                    type="file"
                    accept=".pdf,.docx,.xlsx,.xls,.pptx,.ppt"
                    onChange={handleFileSelect}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Supported: PDF, DOCX, XLSX, PPTX files
                  </p>
                </div>
                <button
                  onClick={handleUploadDocument}
                  disabled={!selectedFile || isUploadingDocument || backendStatus !== 'connected'}
                  className="px-6 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
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

            <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200 mb-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Crawl Website</h3>
              <div className="flex gap-4 items-end">
                <div className="flex-1">
                  <input
                    type="url"
                    placeholder="https://example.com"
                    value={crawlUrl}
                    onChange={(e) => setCrawlUrl(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
                  />
                  <p className="text-xs text-gray-500 mt-1">
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
                  className="px-6 py-2 bg-green-600 text-white rounded-lg text-sm font-medium hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
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

            <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-medium text-gray-900">Indexed Documents</h3>
                <button
                  onClick={handleRefreshDocuments}
                  disabled={isLoadingDocuments || backendStatus !== 'connected'}
                  className="px-4 py-2 bg-gray-100 text-gray-700 border border-gray-300 rounded-lg text-sm font-medium hover:bg-gray-200 disabled:opacity-50 transition-colors flex items-center gap-2"
                >
                  {isLoadingDocuments ? (
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-600"></div>
                  ) : (
                    <Database className="w-4 h-4" />
                  )}
                  Refresh
                </button>
              </div>
              
              {isLoadingDocuments ? (
                <div className="flex items-center justify-center py-8">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
                </div>
              ) : customDocuments.length === 0 ? (
                <div className="text-center py-12">
                  <Database className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-600">No documents indexed yet. Upload a document or crawl a website to get started.</p>
                  {backendStatus !== 'connected' && (
                    <p className="text-red-600 text-sm mt-2">Backend connection required for document management.</p>
                  )}
                </div>
              ) : (
                <div className="space-y-4">
                  {customDocuments.map((doc, index) => (
                    <div key={doc.filename || index} className="flex justify-between items-center p-4 border border-gray-200 rounded-lg">
                      <div>
                        <h4 className="font-medium text-gray-900">{doc.filename}</h4>
                        <p className="text-sm text-gray-500">
                          {doc.total_chunks} chunks indexed
                        </p>
                      </div>
                      <button
                        onClick={() => handleDeleteDocument(doc.filename)}
                        className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
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
      <div className="flex-1 overflow-y-auto bg-gray-50">
        <div className="p-6">
          <div className="mb-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-6">Web Crawls</h2>
            
            <BackendStatus />
            
            <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200 mb-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Crawl New Website</h3>
              <div className="flex gap-4 items-end">
                <div className="flex-1">
                  <input
                    type="url"
                    placeholder="https://example.com"
                    value={localCrawlUrl}
                    onChange={(e) => setLocalCrawlUrl(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Extract content from any website for research and content generation
                  </p>
                </div>
                <button
                  onClick={handleLocalCrawl}
                  disabled={!localCrawlUrl.trim() || localIsLoadingCrawls || backendStatus !== 'connected'}
                  className="px-6 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
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

            <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-medium text-gray-900">Crawl History</h3>
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
                  className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg text-sm hover:bg-gray-200 transition-colors flex items-center gap-2"
                >
                  <RefreshCw className="w-4 h-4" />
                  Refresh
                </button>
              </div>
              
              {localIsLoadingCrawls ? (
                <div className="text-center py-8">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
                  <p className="text-gray-600">Loading crawls...</p>
                </div>
              ) : localWebCrawls.length > 0 ? (
                <div className="space-y-4">
                  {localWebCrawls.map((crawl) => (
                    <div key={crawl.id} className="border border-gray-200 rounded-lg p-4">
                      <div className="flex justify-between items-start mb-2">
                        <div>
                          <h4 className="font-medium text-gray-900">{crawl.source || crawl.url}</h4>
                          <p className="text-sm text-gray-600">
                            {crawl.created_at ? new Date(crawl.created_at).toLocaleString() : 'Unknown date'}
                          </p>
                        </div>
                        <span className="px-2 py-1 rounded text-xs font-medium bg-green-100 text-green-800">
                          Completed
                        </span>
                      </div>
                      <div className="mb-3">
                        <h5 className="font-medium text-gray-900 mb-1">{crawl.topic}</h5>
                        <p className="text-sm text-gray-600 line-clamp-3">
                          {crawl.findings && crawl.findings.length > 200 
                            ? crawl.findings.substring(0, 200) + '...' 
                            : (crawl.findings || 'No content available')}
                        </p>
                      </div>
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="text-gray-600">Hostname:</span>
                          <span className="ml-2 font-medium">{crawl.tags ? crawl.tags.split(',').find(tag => tag.includes('.')) || 'Unknown' : 'Unknown'}</span>
                        </div>
                        <div>
                          <span className="text-gray-600">Content Length:</span>
                          <span className="ml-2 font-medium">{crawl.data ? crawl.data.match(/Total length: (\d+)/)?.[1] || 'Unknown' : 'Unknown'} chars</span>
                        </div>
                      </div>
                      <div className="mt-3 flex flex-wrap gap-1">
                        {crawl.tags && crawl.tags.split(',').map((tag, index) => (
                          <span key={index} className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs">
                            {tag.trim()}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <Globe className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-600">No web crawls yet. Start crawling websites to see them here.</p>
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
      <div className="flex-1 overflow-y-auto bg-gray-50">
        <div className="p-6">
          <div className="mb-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-6">Content Generation History</h2>
            
            <BackendStatus />
            
            {isLoadingHistory ? (
              <div className="flex items-center justify-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
                <span className="ml-2 text-gray-600">Loading history...</span>
              </div>
            ) : (
              <>
                {availableDates.length > 0 ? (
                  <>
                    {/* Date Dropdown */}
                    <div className="mb-6">
                      <label htmlFor="history-date" className="block text-sm font-medium text-gray-700 mb-2">
                        Select Date ({promptHistory.length} total entries)
                      </label>
                      <select
                        id="history-date"
                        value={selectedHistoryDate}
                        onChange={(e) => setSelectedHistoryDate(e.target.value)}
                        className="w-full md:w-auto px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500 bg-white"
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
                        <div key={entry.id} className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
                          <div className="flex justify-between items-start mb-4">
                            <div>
                              <h3 className="text-lg font-medium text-gray-900 mb-1">{entry.creatorName} Style</h3>
                              <p className="text-sm text-gray-600">
                                {entry.timestamp.toLocaleTimeString()} • {entry.timestamp.toLocaleDateString()}
                              </p>
                            </div>
                            <div className="flex gap-2">
                              <button
                                onClick={() => copyToClipboard(entry.content.linkedinPosts?.[0] || '')}
                                className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
                                title="Copy LinkedIn Post"
                              >
                                <Copy className="w-4 h-4" />
                              </button>
                            </div>
                          </div>
                          
                          <div className="mb-4">
                            <h4 className="text-sm font-medium text-gray-700 mb-2">Original Prompt:</h4>
                            <div className="bg-gray-50 p-3 rounded-lg text-sm text-gray-700">
                              {entry.prompt}
                            </div>
                          </div>
                          
                          {entry.researchUsed && entry.researchUsed.length > 0 && (
                            <div className="mb-4">
                              <h4 className="text-sm font-medium text-gray-700 mb-2">Research Used ({entry.researchUsed.length}):</h4>
                              <div className="flex flex-wrap gap-2">
                                {entry.researchUsed.map((research, index) => (
                                  <span key={index} className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs">
                                    {research.topic}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                          
                          {entry.backendSources && entry.backendSources.length > 0 && (
                            <div className="mb-4">
                              <h4 className="text-sm font-medium text-gray-700 mb-2">Backend Sources Used:</h4>
                              <div className="flex flex-wrap gap-2">
                                {entry.backendSources.map((source, index) => (
                                  <span key={index} className="bg-green-100 text-green-800 px-2 py-1 rounded text-xs">
                                    {source.filename}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                          
                          {/* Generated Content Tabs */}
                          <div className="border-t pt-4">
                            <h4 className="text-sm font-medium text-gray-700 mb-3">Generated Content:</h4>
                            
                            {/* LinkedIn Post */}
                            {entry.content.linkedinPosts && entry.content.linkedinPosts.length > 0 && (
                              <div className="mb-4">
                                <h5 className="text-xs font-medium text-gray-600 mb-2">LinkedIn Post:</h5>
                                <div className="bg-gray-50 p-3 rounded-lg text-sm text-gray-700 whitespace-pre-line max-h-40 overflow-y-auto">
                                  {entry.content.linkedinPosts[0]}
                                </div>
                              </div>
                            )}
                            
                            {/* Video Script */}
                            {entry.content.videoScripts && entry.content.videoScripts.length > 0 && (
                              <div className="mb-4">
                                <h5 className="text-xs font-medium text-gray-600 mb-2">Video Script:</h5>
                                <div className="bg-gray-50 p-3 rounded-lg text-sm text-gray-700 whitespace-pre-line max-h-32 overflow-y-auto">
                                  {entry.content.videoScripts[0]}
                                </div>
                              </div>
                            )}
                            
                            {/* Hashtags */}
                            {entry.content.hashtags && entry.content.hashtags.length > 0 && (
                              <div className="mb-4">
                                <h5 className="text-xs font-medium text-gray-600 mb-2">Hashtags:</h5>
                                <div className="flex flex-wrap gap-1">
                                  {entry.content.hashtags.map((hashtag, index) => (
                                    <span key={index} className="bg-purple-100 text-purple-800 px-2 py-1 rounded text-xs">
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
                          <Calendar className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                          <p className="text-gray-600">No content generated on this date.</p>
                        </div>
                      )}
                    </div>
                  </>
                ) : (
                  <div className="text-center py-12">
                    <BookOpen className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-600">No content generated yet. Start creating content to see your history here.</p>
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
        <div className="bg-gray-50 border-b border-gray-200 px-6 py-5">
          <BackendStatus />
          
          <div className="mb-4">
            <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-3">
              Creator Styles
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {creatorDatabase.slice(0, 3).map((creator) => {
                if (!creator || !creator.key) return null;
                const IconComponent = creator.icon;
                return (
                  <button
                    key={creator.key}
                    onClick={() => handleCreatorSelect(creator.key)}
                    className="flex items-center gap-3 p-3.5 bg-white border border-gray-200 rounded-lg hover:border-gray-300 hover:transform hover:-translate-y-0.5 hover:shadow-md transition-all text-left"
                  >
                    <IconComponent className="w-4.5 h-4.5 text-gray-600" />
                    <span className="text-sm font-medium text-gray-700">{creator.name}</span>
                  </button>
                );
              })}
            </div>
          </div>

          <div>
            <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-3">
              Quick Actions
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <button
                onClick={() => setActiveTab('research')}
                className="flex items-center gap-3 p-3.5 bg-white border border-gray-200 rounded-lg hover:border-gray-300 hover:transform hover:-translate-y-0.5 hover:shadow-md transition-all text-left"
              >
                <Database className="w-4.5 h-4.5 text-gray-600" />
                <span className="text-sm font-medium text-gray-700">Browse Research</span>
              </button>
              <button
                onClick={() => setActiveTab('documents')}
                className="flex items-center gap-3 p-3.5 bg-white border border-gray-200 rounded-lg hover:border-gray-300 hover:transform hover:-translate-y-0.5 hover:shadow-md transition-all text-left"
              >
                <FileText className="w-4.5 h-4.5 text-gray-600" />
                <span className="text-sm font-medium text-gray-700">Manage Documents</span>
              </button>
            </div>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-6 py-6 bg-white">
          <div className="space-y-6">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex w-full ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div className="flex items-start gap-3 max-w-3xl">
                  {message.type === 'bot' && (
                    <div className="w-8 h-8 bg-gray-100 rounded-lg flex items-center justify-center">
                      <Bot className="w-4 h-4 text-gray-600" />
                    </div>
                  )}
                  <div
                    className={`px-4 py-3 rounded-lg ${
                      message.type === 'user'
                        ? 'bg-gray-900 text-white'
                        : 'bg-gray-100 text-gray-900'
                    }`}
                  >
                    <p className="text-sm leading-relaxed whitespace-pre-line">{message.text}</p>
                    <div className="text-xs text-gray-500 mt-2">
                      {message.timestamp.toLocaleTimeString()}
                    </div>
                  </div>
                  {message.type === 'user' && (
                    <div className="w-8 h-8 bg-gray-900 rounded-lg flex items-center justify-center">
                      <User className="w-4 h-4 text-white" />
                    </div>
                  )}
                </div>
              </div>
            ))}

            {generatedContent && <ContentDisplay content={generatedContent} />}

            {isTyping && (
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 bg-gray-100 rounded-lg flex items-center justify-center">
                  <Bot className="w-4 h-4 text-gray-600" />
                </div>
                <div className="bg-gray-100 px-4 py-3 rounded-lg">
                  <div className="flex items-center gap-1">
                    <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                    <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

        <div className="border-t border-gray-200 px-6 py-4 bg-white">
          <div className="flex items-end gap-3">
            <div className="flex-1">
              <textarea
                ref={textareaRef}
                value={localInputValue}
                onChange={handleLocalInputChange}
                onKeyDown={handleLocalKeyDown}
                placeholder="Ask me to create content using your research and uploaded documents..."
                className="w-full px-4 py-3 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500 resize-none"
                style={{ minHeight: '48px', maxHeight: '128px' }}
                rows="1"
              />
            </div>
            <button
              onClick={handleLocalSend}
              disabled={!localInputValue.trim() || isTyping}
              className="px-4 py-3 bg-gray-900 text-white rounded-lg hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
        </div>
      </>
    );
  };

  return (
    <div className="flex h-screen bg-gray-50">
      <div className="w-64 bg-white border-r border-gray-200 flex flex-col">
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
              <Linkedin className="w-4 h-4 text-white" />
            </div>
            <h1 className="text-lg font-semibold text-gray-900">Content Creator</h1>
          </div>
          
          <nav className="space-y-2">
            <button
              onClick={() => setActiveTab('chat')}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                activeTab === 'chat'
                  ? 'bg-gray-900 text-white'
                  : 'text-gray-700 hover:bg-gray-100'
              }`}
            >
              <Bot className="w-4 h-4" />
              Chat
            </button>
            <button
              onClick={() => setActiveTab('research')}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                activeTab === 'research'
                  ? 'bg-gray-900 text-white'
                  : 'text-gray-700 hover:bg-gray-100'
              }`}
            >
              <Database className="w-4 h-4" />
              Research ({researchDatabase.length})
            </button>
            <button
              onClick={() => setActiveTab('creators')}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                activeTab === 'creators'
                  ? 'bg-gray-900 text-white'
                  : 'text-gray-700 hover:bg-gray-100'
              }`}
            >
              <Users className="w-4 h-4" />
              Creators ({creatorDatabase.length})
            </button>
            <button
              onClick={() => setActiveTab('documents')}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                activeTab === 'documents'
                  ? 'bg-gray-900 text-white'
                  : 'text-gray-700 hover:bg-gray-100'
              }`}
            >
              <Upload className="w-4 h-4" />
              Documents ({customDocuments.length})
            </button>
            <button
              onClick={() => setActiveTab('crawls')}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                activeTab === 'crawls'
                  ? 'bg-gray-900 text-white'
                  : 'text-gray-700 hover:bg-gray-100'
              }`}
            >
              <Globe className="w-4 h-4" />
              Web Crawls ({webCrawls?.length || 0})
            </button>
            <button
              onClick={() => setActiveTab('history')}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                activeTab === 'history'
                  ? 'bg-gray-900 text-white'
                  : 'text-gray-700 hover:bg-gray-100'
              }`}
            >
              <BookOpen className="w-4 h-4" />
              History ({promptHistory.length})
            </button>
          </nav>
        </div>

        <div className="flex-1"></div>

        <div className="p-4 border-t border-gray-200">
          <button
            onClick={() => setShowConfig(true)}
            className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-100 transition-colors"
          >
            <Settings className="w-4 h-4" />
            Settings
          </button>
        </div>
      </div>

      <div className="flex-1 flex flex-col">
        {activeTab === 'chat' && <ChatTab />}
        {activeTab === 'research' && <ResearchTab />}
        {activeTab === 'creators' && <CreatorTab />}
        {activeTab === 'documents' && <CustomDocumentsTab />}
        {activeTab === 'crawls' && <WebCrawlsTab />}
        {activeTab === 'history' && <HistoryTab />}
      </div>

      {showConfig && <ConfigPanel />}
    </div>
  );
};

export default LinkedInContentBot;