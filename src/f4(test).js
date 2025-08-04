import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Linkedin, Video, FileText, Database, TrendingUp, Users, Lightbulb, Copy, Settings, Plus, Trash2, Edit3, Save, X, Search, BookOpen, UserPlus, ChevronLeft, ChevronRight, Upload, Globe, AlertCircle, CheckCircle, Clock } from 'lucide-react';

// 🔧 CONFIGURATION - UPDATE THIS WITH YOUR BACKEND URL
const BACKEND_CONFIG = {
  // 🚨 CHANGE THIS TO YOUR DEPLOYED BACKEND URL
  url: 'https://linkedin-content-creator-api-11874687899.asia-south1.run.app', // Replace with your Cloud Run URL
  
  // For local development, use: 'http://localhost:8000'
  // For production, use: 'https://your-service-name-project-id.region.run.app'
  
  // ⚠️ CURRENT ISSUE: The deployed service is a LinkedIn Content Creator API, not a Document Management API
  // The frontend expects document management endpoints (/upload, /documents, /ask, etc.)
  // but the deployed service provides LinkedIn content generation endpoints (/linkedin/creators, /linkedin/research, etc.)
  // 
  // To fix this, you need to either:
  // 1. Deploy the document management backend (backend/server.js or src/SpikedAI-backed-deploy/app.py)
  // 2. Update the frontend to work with the LinkedIn API
  // 3. Use a different backend URL that provides document management functionality
};

// Cloud Document Manager - Updated to match your backend API
const createCloudDocumentManager = () => {
  const BACKEND_URL = BACKEND_CONFIG.url;
  
  return {
    // Upload document to cloud storage
    uploadDocument: async (file) => {
      try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${BACKEND_URL}/upload`, {
          method: 'POST',
          body: formData
        });
        
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.detail || `Upload failed: ${response.status}`);
        }
        
        return await response.json();
      } catch (error) {
        console.error('Cloud upload error:', error);
        throw error;
      }
    },

    // Get all documents from cloud storage
    getDocuments: async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/documents`);
        
        if (!response.ok) {
          throw new Error(`Get documents failed: ${response.status}`);
        }
        
        return await response.json();
      } catch (error) {
        console.error('Cloud get documents error:', error);
        throw error;
      }
    },

    // Delete document from cloud storage
    deleteDocument: async (filename) => {
      try {
        const response = await fetch(`${BACKEND_URL}/documents/${encodeURIComponent(filename)}`, {
          method: 'DELETE'
        });
        
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.detail || `Delete failed: ${response.status}`);
        }
        
        return await response.json();
      } catch (error) {
        console.error('Cloud delete error:', error);
        throw error;
      }
    },

    // Ask questions using the backend's /ask endpoint
    askQuestion: async (question) => {
      try {
        const response = await fetch(`${BACKEND_URL}/ask`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ question })
        });
        
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.detail || `Ask failed: ${response.status}`);
        }
        
        return await response.json();
      } catch (error) {
        console.error('Cloud ask error:', error);
        throw error;
      }
    },

    // Crawl website and store in cloud
    crawlWebsite: async (url) => {
      try {
        const response = await fetch(`${BACKEND_URL}/crawl`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ url })
        });
        
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.detail || `Crawl failed: ${response.status}`);
        }
        
        return await response.json();
      } catch (error) {
        console.error('Cloud crawl error:', error);
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
        const response = await fetch(`${BACKEND_URL}/health`);
        return await response.json();
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
  supabaseUrl: '',
  supabaseKey: ''
};

// Supabase client initialization
const createSupabaseClient = () => {
  const supabaseUrl = configStore.supabaseUrl || '';
  const supabaseKey = configStore.supabaseKey || '';
  
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
  const [config, setConfig] = useState(configStore);
  const [generatedContent, setGeneratedContent] = useState(null);
  const [selectedOptions, setSelectedOptions] = useState({
    linkedinPost: 0,
    videoScript: 0
  });
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
  const [supabaseClient, setSupabaseClient] = useState(null);
  const [documentManager, setDocumentManager] = useState(null);
  const [isLoadingResearch, setIsLoadingResearch] = useState(false);
  const [customDocuments, setCustomDocuments] = useState([]);
  const [isLoadingDocuments, setIsLoadingDocuments] = useState(false);
  const [editingResearch, setEditingResearch] = useState(null);
  const [editingCreator, setEditingCreator] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [isUploadingDocument, setIsUploadingDocument] = useState(false);

  // Test backend connection function
  const testBackendConnection = async () => {
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
  };

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
          <p className="text-xs text-yellow-600 mt-1">
            ⚠️ Connected to LinkedIn Content Creator API (some document management features may not be available)
          </p>
        )}
        {backendError && (
          <div className="text-xs text-red-600 mt-1">
            <p className="font-medium">Error: {backendError}</p>
            <p className="mt-1">
              The backend service at {BACKEND_CONFIG.url} is responding but may not have the expected endpoints. 
              This might be because:
            </p>
            <ul className="list-disc list-inside mt-1 ml-2">
              <li>The service is a LinkedIn Content Creator API, not a Document Management API</li>
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

  // Enhanced Content Generation with Cloud Document Manager
  const generateContentWithCloudDocuments = async (prompt, creatorStyle, relevantResearch) => {
    if (!documentManager) {
      throw new Error('Document manager not initialized');
    }

    try {
      // Use the backend's /ask endpoint to get relevant information
      const askResult = await documentManager.askQuestion(prompt);
      
      // Combine research data with backend search results
      const researchContext = relevantResearch.map((r, index) => `
${index + 1}. Topic: ${r.topic}
   Findings: ${r.findings}
   Data: ${r.data}
   Source: ${r.source}
   Date Added: ${r.dateAdded.toLocaleDateString()}
   ${r.relevanceScore ? `Relevance Score: ${r.relevanceScore.toFixed(1)}` : ''}
`).join('\n');

      // Add backend search results as context
      const backendContext = askResult.answer ? `
Backend Document Search Results:
Answer: ${askResult.answer}

Sources: ${askResult.sources.map(s => s.filename).join(', ')}

Follow-up Suggestions: ${askResult.sales_followups.concat(askResult.client_followups).join(', ')}
` : '';

      // Create enhanced prompt with both research and backend context
      const enhancedPrompt = `Create LinkedIn content in the style of ${creatorStyle.name}.

Creator Style:
- Tone: ${creatorStyle.style.tone}
- Structure: ${creatorStyle.style.structure}
- Language: ${creatorStyle.style.language}
- Length: ${creatorStyle.style.length}
- Common hooks: ${creatorStyle.style.hooks.join(', ')}
- Common endings: ${creatorStyle.style.endings.join(', ')}
- Key characteristics: ${creatorStyle.style.characteristics}

Research Context:
${researchContext}

${backendContext}

CRITICAL INSTRUCTION: Use information from ALL ${relevantResearch.length} research items and the backend search results provided above. Combine insights from multiple sources and reference specific data points.

User Request: ${prompt}

Create TWO DIFFERENT variations each for LinkedIn posts and video scripts. Return in JSON format:
{
  "linkedinPosts": ["first post variation", "second post variation"],
  "videoScripts": ["first video script variation", "second video script variation"],
  "hashtags": ["array", "of", "hashtags"],
  "engagement_tips": ["array", "of", "tips"]
}`;

      // Use OpenAI with enhanced context
      const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${config.openaiApiKey}`
        },
        body: JSON.stringify({
          model: config.model,
          messages: [
            { role: 'system', content: enhancedPrompt },
            { role: 'user', content: prompt }
          ],
          temperature: 0.8,
          max_tokens: 3000
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`OpenAI API error: ${errorData.error?.message || response.status}`);
      }

      const data = await response.json();
      const content = JSON.parse(data.choices[0].message.content);
      
      // Save prompt to history with backend sources
      setPromptHistory(prev => [{
        id: Date.now(),
        prompt,
        creator: creatorStyle.name,
        research: relevantResearch.map(r => r.topic),
        timestamp: new Date(),
        response: content,
        backendSources: askResult.sources || [],
        backendAnswer: askResult.answer || null
      }, ...prev]);

      return content;
    } catch (error) {
      console.error('Enhanced content generation error:', error);
      throw error;
    }
  };

  // OpenAI Integration for multiple variations (fallback)
  const generateContentWithOpenAI = async (prompt, creatorStyle, relevantResearch) => {
    if (!config.openaiApiKey) {
      throw new Error('OpenAI API key not configured');
    }

    const systemPrompt = `You are a LinkedIn content creator that mimics the style of ${creatorStyle.name}. 

Style characteristics:
- Tone: ${creatorStyle.style.tone}
- Structure: ${creatorStyle.style.structure}
- Language: ${creatorStyle.style.language}
- Length: ${creatorStyle.style.length}
- Common hooks: ${creatorStyle.style.hooks.join(', ')}
- Common endings: ${creatorStyle.style.endings.join(', ')}
- Key characteristics: ${creatorStyle.style.characteristics}

Research data to incorporate (prioritized by relevance and recency):
${relevantResearch.map((r, index) => `
${index + 1}. Topic: ${r.topic}
   Findings: ${r.findings}
   Data: ${r.data}
   Source: ${r.source}
   Date Added: ${r.dateAdded.toLocaleDateString()}
   ${r.relevanceScore ? `Relevance Score: ${r.relevanceScore.toFixed(1)}` : ''}
`).join('\n')}

CRITICAL INSTRUCTION: You MUST use information from ALL ${relevantResearch.length} research items provided above. Do NOT focus on just one research item. Your task is to:

1. COMBINE insights from multiple research sources
2. Reference specific data points from different research items
3. Weave together findings from various topics
4. Create comprehensive content that draws from the full research database

For each variation, explicitly reference at least 2-3 different research items by topic or source. Make the content richer by connecting insights across multiple research areas.

Create TWO DIFFERENT variations each for LinkedIn posts and video scripts that incorporate this research data in the creator's authentic style. Make sure each variation is distinct in approach while maintaining the creator's voice.

Return your response in this JSON format:
{
  "linkedinPosts": ["first post variation", "second post variation"],
  "videoScripts": ["first video script variation", "second video script variation"],
  "hashtags": ["array", "of", "hashtags"],
  "engagement_tips": ["array", "of", "tips"]
}`;

    try {
      const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${config.openaiApiKey}`
        },
        body: JSON.stringify({
          model: config.model,
          messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: prompt }
          ],
          temperature: 0.8,
          max_tokens: 3000
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`OpenAI API error: ${errorData.error?.message || response.status}`);
      }

      const data = await response.json();
      const content = JSON.parse(data.choices[0].message.content);
      
      // Save prompt to history
      setPromptHistory(prev => [{
        id: Date.now(),
        prompt,
        creator: creatorStyle.name,
        research: relevantResearch.map(r => r.topic),
        timestamp: new Date(),
        response: content
      }, ...prev]);

      return content;
    } catch (error) {
      console.error('OpenAI API error:', error);
      throw error;
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

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

  const loadResearchFromSupabase = async (client = supabaseClient) => {
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

  const deleteResearch = async (id) => {
    if (supabaseClient) {
      try {
        const { error } = await supabaseClient.from('research').delete().eq('id', id);
        if (error) throw error;
        
        await loadResearchFromSupabase();
      } catch (error) {
        console.error('Error deleting research:', error);
        setResearchDatabase(prev => prev.filter(item => item.id !== id));
      }
    } else {
      setResearchDatabase(prev => prev.filter(item => item.id !== id));
    }
  };

  const updateResearch = async (id, updatedData) => {
    const updated = {
      topic: updatedData.topic,
      findings: updatedData.findings,
      source: updatedData.source,
      data: updatedData.data,
      tags: typeof updatedData.tags === 'string' 
        ? updatedData.tags.split(',').map(tag => tag.trim()).filter(tag => tag).join(',')
        : (Array.isArray(updatedData.tags) ? updatedData.tags.join(',') : updatedData.tags)
    };

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

    if (supabaseClient) {
      try {
        const { error } = await supabaseClient.from('research').update(updated).eq('id', id);
        if (error) {
          console.error('Supabase update error:', error);
          throw error;
        }
        
        await loadResearchFromSupabase();
      } catch (error) {
        console.error('Error updating research in Supabase:', error);
      }
    }
    
    setEditingResearch(null);
  };

  const findRelevantResearch = (text, maxResults = 5) => {
    const lowerText = text.toLowerCase();
    const searchTerms = lowerText.split(' ').filter(term => term.length > 2);
    
    const scoredResearch = researchDatabase.map(item => {
      let score = 0;
      const itemText = `${item.topic} ${item.findings} ${item.data} ${(Array.isArray(item.tags) ? item.tags.join(' ') : '')}`.toLowerCase();
      
      if (itemText.includes(lowerText)) score += 10;
      if (item.topic.toLowerCase().includes(lowerText)) score += 8;
      
      if (Array.isArray(item.tags)) {
        item.tags.forEach(tag => {
          if (lowerText.includes(tag.toLowerCase()) || tag.toLowerCase().includes(lowerText)) {
            score += 6;
          }
        });
      }
      
      searchTerms.forEach(term => {
        if (itemText.includes(term)) score += 2;
        if (item.topic.toLowerCase().includes(term)) score += 3;
      });
      
      const daysSinceAdded = (new Date() - new Date(item.dateAdded)) / (1000 * 60 * 60 * 24);
      if (daysSinceAdded < 7) score += 1;
      if (daysSinceAdded < 30) score += 0.5;
      
      return { ...item, relevanceScore: score };
    });
    
    const sortedByRelevance = scoredResearch
      .sort((a, b) => {
        if (b.relevanceScore !== a.relevanceScore) {
          return b.relevanceScore - a.relevanceScore;
        }
        return new Date(b.dateAdded) - new Date(a.dateAdded);
      });
    
    const minItems = Math.min(3, researchDatabase.length);
    const targetItems = Math.min(maxResults, researchDatabase.length);
    
    let finalResults = sortedByRelevance.slice(0, targetItems);
    
    if (finalResults.length < minItems && researchDatabase.length >= minItems) {
      const usedIds = new Set(finalResults.map(item => item.id));
      const additionalItems = researchDatabase
        .filter(item => !usedIds.has(item.id))
        .sort((a, b) => new Date(b.dateAdded) - new Date(a.dateAdded))
        .slice(0, minItems - finalResults.length);
      
      finalResults = [...finalResults, ...additionalItems];
    }
    
    if (finalResults.length < minItems && researchDatabase.length > 0) {
      finalResults = researchDatabase
        .sort((a, b) => new Date(b.dateAdded) - new Date(a.dateAdded))
        .slice(0, minItems);
    }
    
    return finalResults;
  };

  const handleSendMessage = async (messageText) => {
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
    setSelectedOptions({ linkedinPost: 0, videoScript: 0 });

    try {
      let creatorStyle = creatorDatabase.find(c => c.key === 'gary-v');
      
      const lowerText = text.toLowerCase();
      for (const creator of creatorDatabase) {
        if (lowerText.includes(creator.name.toLowerCase()) || 
            lowerText.includes(creator.key)) {
          creatorStyle = creator;
          break;
        }
      }

      const relevantResearch = findRelevantResearch(text, 5);

      if (!config.openaiApiKey) {
        throw new Error('Please configure OpenAI API key in settings to generate content');
      }

      console.log(`Using ${relevantResearch.length} research items:`, relevantResearch.map(r => r.topic));

      // Try cloud document manager first, fallback to OpenAI
      let content;
      try {
        if (backendStatus === 'connected') {
          content = await generateContentWithCloudDocuments(text, creatorStyle, relevantResearch);
          console.log('Content generated using cloud document manager');
        } else {
          throw new Error('Backend not connected');
        }
      } catch (cloudError) {
        console.log('Cloud document manager failed, using OpenAI fallback:', cloudError.message);
        content = await generateContentWithOpenAI(text, creatorStyle, relevantResearch);
      }

      setGeneratedContent(content);
      
      let responseText = `I've created 2 variations for each content type in ${creatorStyle.name}'s style.`;
      
      if (relevantResearch.length > 0) {
        responseText += ` I analyzed ${relevantResearch.length} research items: ${relevantResearch.map(r => r.topic).join(', ')}.`;
      }
      
      if (backendStatus === 'connected') {
        responseText += ` I also searched through your uploaded documents for relevant context.`;
      }
      
      responseText += ` Use the navigation arrows to choose your preferred options.`;
      
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
  };

  const handleCreatorSelect = (creatorKey) => {
    const creator = creatorDatabase.find(c => c.key === creatorKey);
    const message = `Create a ${creator.name} style LinkedIn post using my latest research and uploaded documents`;
    handleSendMessage(message);
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
  };

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
    if (!documentManager || !url) return;

    try {
      const result = await documentManager.crawlWebsite(url);
      console.log('Website crawled:', result);
      
      const documents = await documentManager.getDocuments();
      setCustomDocuments(documents || []);
      
      const successMessage = {
        id: Date.now(),
        type: 'bot',
        text: `✅ Successfully crawled and indexed content from "${url}". Title: "${result.title || 'Unknown'}". You can now ask questions about this content!`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, successMessage]);
      
    } catch (error) {
      console.error('Crawl failed:', error);
      const errorMessage = {
        id: Date.now(),
        type: 'bot',
        text: `❌ Crawl failed: ${error.message}`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
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
        
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            OpenAI API Key
          </label>
          <input
            type="password"
            value={config.openaiApiKey}
            onChange={(e) => setConfig(prev => ({ ...prev, openaiApiKey: e.target.value }))}
            placeholder="sk-..."
            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500 transition-colors"
          />
        </div>

        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Model
          </label>
          <select
            value={config.model}
            onChange={(e) => setConfig(prev => ({ ...prev, model: e.target.value }))}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500 transition-colors"
          >
            <option value="gpt-4">GPT-4</option>
            <option value="gpt-4-turbo">GPT-4 Turbo</option>
            <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
          </select>
        </div>

        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Supabase URL
          </label>
          <input
            type="text"
            value={config.supabaseUrl}
            onChange={(e) => setConfig(prev => ({ ...prev, supabaseUrl: e.target.value }))}
            placeholder="https://your-project.supabase.co"
            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500 transition-colors"
          />
        </div>

        <div className="mb-8">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Supabase Anon Key
          </label>
          <input
            type="password"
            value={config.supabaseKey}
            onChange={(e) => setConfig(prev => ({ ...prev, supabaseKey: e.target.value }))}
            placeholder="eyJ..."
            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500 transition-colors"
          />
        </div>

        <div className="flex gap-3">
          <button
            onClick={() => {
              configStore = { ...config };
              setShowConfig(false);
            }}
            className="flex-1 px-4 py-2 bg-gray-900 text-white rounded-lg text-sm font-medium hover:bg-gray-800 transition-colors"
          >
            Save
          </button>
          <button
            onClick={() => setShowConfig(false)}
            className="px-4 py-2 bg-gray-100 text-gray-700 border border-gray-300 rounded-lg text-sm font-medium hover:bg-gray-200 transition-colors"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );

  const ContentDisplay = ({ content }) => (
    <div className="bg-white border border-gray-200 rounded-xl p-6 mt-4 shadow-sm">
      <div className="flex items-center gap-2 mb-6 pb-4 border-b border-gray-100">
        <Linkedin className="w-5 h-5 text-blue-600" />
        <h3 className="text-base font-semibold text-gray-900">Generated Content</h3>
      </div>

      <div className="mb-8">
        <div className="flex justify-between items-center mb-3">
          <h4 className="text-sm font-semibold text-gray-900 uppercase tracking-wide">
            LinkedIn Post (Option {selectedOptions.linkedinPost + 1} of 2)
          </h4>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setSelectedOptions(prev => ({ 
                ...prev, 
                linkedinPost: prev.linkedinPost === 0 ? 1 : 0 
              }))}
              className="p-2 bg-gray-100 text-gray-700 border border-gray-300 rounded-lg hover:bg-gray-200 transition-colors"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <button
              onClick={() => setSelectedOptions(prev => ({ 
                ...prev, 
                linkedinPost: prev.linkedinPost === 1 ? 0 : 1 
              }))}
              className="p-2 bg-gray-100 text-gray-700 border border-gray-300 rounded-lg hover:bg-gray-200 transition-colors"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
            <button
              onClick={() => copyToClipboard(content.linkedinPosts[selectedOptions.linkedinPost])}
              className="flex items-center gap-2 px-3 py-1.5 bg-gray-100 text-gray-700 border border-gray-300 rounded-lg text-xs font-medium hover:bg-gray-200 transition-colors"
            >
              <Copy className="w-3.5 h-3.5" />
              Copy
            </button>
          </div>
        </div>
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 text-sm text-gray-700 whitespace-pre-line leading-relaxed">
          {content.linkedinPosts && content.linkedinPosts[selectedOptions.linkedinPost]}
        </div>
      </div>

      <div className="mb-8">
        <div className="flex justify-between items-center mb-3">
          <h4 className="text-sm font-semibold text-gray-900 uppercase tracking-wide flex items-center gap-2">
            <Video className="w-4 h-4" />
            Video Script (Option {selectedOptions.videoScript + 1} of 2)
          </h4>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setSelectedOptions(prev => ({ 
                ...prev, 
                videoScript: prev.videoScript === 0 ? 1 : 0 
              }))}
              className="p-2 bg-gray-100 text-gray-700 border border-gray-300 rounded-lg hover:bg-gray-200 transition-colors"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <button
              onClick={() => setSelectedOptions(prev => ({ 
                ...prev, 
                videoScript: prev.videoScript === 1 ? 0 : 1 
              }))}
              className="p-2 bg-gray-100 text-gray-700 border border-gray-300 rounded-lg hover:bg-gray-200 transition-colors"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
            <button
              onClick={() => copyToClipboard(content.videoScripts[selectedOptions.videoScript])}
              className="flex items-center gap-2 px-3 py-1.5 bg-gray-100 text-gray-700 border border-gray-300 rounded-lg text-xs font-medium hover:bg-gray-200 transition-colors"
            >
              <Copy className="w-3.5 h-3.5" />
              Copy
            </button>
          </div>
        </div>
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 text-xs text-gray-700 whitespace-pre-line leading-relaxed font-mono">
          {content.videoScripts && content.videoScripts[selectedOptions.videoScript]}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h4 className="text-sm font-semibold text-gray-900 uppercase tracking-wide mb-3">
            Hashtags
          </h4>
          <div className="flex flex-wrap gap-2">
            {content.hashtags && content.hashtags.map((tag, index) => (
              <span
                key={index}
                className="bg-gray-100 text-gray-700 px-2 py-1 rounded text-xs font-medium"
              >
                {tag}
              </span>
            ))}
          </div>
        </div>
        <div>
          <h4 className="text-sm font-semibold text-gray-900 uppercase tracking-wide mb-3">
            Tips
          </h4>
          <ul className="text-xs text-gray-600 leading-relaxed space-y-1">
            {content.engagement_tips && content.engagement_tips.map((tip, index) => (
              <li key={index} className="flex items-start">
                <span className="mr-2">•</span>
                {tip}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );

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

    const handleAddResearch = () => {
      if (!localNewResearch.topic.trim()) return;

      const research = {
        topic: localNewResearch.topic,
        findings: localNewResearch.findings,
        source: localNewResearch.source,
        data: localNewResearch.data,
        tags: localNewResearch.tags.split(',').map(tag => tag.trim()).filter(tag => tag).join(','),
        created_at: new Date().toISOString()
      };

      if (supabaseClient) {
        try {
          supabaseClient.from('research').insert([research]).then(async () => {
            await loadResearchFromSupabase();
          });
        } catch (error) {
          console.error('Error adding research:', error);
          setResearchDatabase(prev => [{
            ...research,
            id: Date.now(),
            tags: research.tags.split(',').map(t => t.trim()),
            dateAdded: new Date()
          }, ...prev]);
        }
      } else {
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

            {filteredResearch.length > 0 && (
              <div className="mb-4 text-sm text-gray-600">
                Showing {filteredResearch.length} research item{filteredResearch.length !== 1 ? 's' : ''} 
                {localSearchTerm && ` matching "${localSearchTerm}"`}
                {filteredResearch.length > 0 && ` (sorted by newest first)`}
              </div>
            )}

            <div className="space-y-4">
              {filteredResearch.length === 0 && !isLoadingResearch && (
                <div className="text-center py-12 bg-white rounded-lg border border-gray-200">
                  <Database className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-600">
                    {localSearchTerm ? `No research found matching "${localSearchTerm}"` : 'No research found. Add your first research above!'}
                  </p>
                </div>
              )}
              
              {filteredResearch.map((research, index) => (
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
        </div>
      </div>
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

    const handleAddCreator = () => {
      if (!localNewCreator.name.trim()) return;
      
      const creator = {
        id: Date.now(),
        name: localNewCreator.name,
        key: localNewCreator.name.toLowerCase().replace(/\s+/g, '-'),
        icon: UserPlus,
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
      
      setCreatorDatabase(prev => [creator, ...prev]);
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
                            onClick={() => setCreatorDatabase(prev => prev.filter(item => item.id !== creator.id))}
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

    const handleSave = () => {
      const updatedData = {
        name: editForm.name,
        tone: editForm.tone,
        structure: editForm.structure,
        language: editForm.language,
        length: editForm.length,
        hooks: editForm.hooks.split(',').map(h => h.trim()).filter(h => h),
        endings: editForm.endings.split(',').map(e => e.trim()).filter(e => e),
        characteristics: editForm.characteristics
      };
      
      setCreatorDatabase(prev => prev.map(c => 
        c.id === creator.id 
          ? {
              ...c,
              name: updatedData.name,
              style: {
                tone: updatedData.tone,
                structure: updatedData.structure,
                language: updatedData.language,
                length: updatedData.length,
                hooks: updatedData.hooks,
                endings: updatedData.endings,
                characteristics: updatedData.characteristics
              }
            }
          : c
      ));
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
    const [crawlUrl, setCrawlUrl] = useState('');
    const [isCrawling, setIsCrawling] = useState(false);

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

  const HistoryTab = () => (
    <div className="flex-1 overflow-y-auto bg-gray-50">
      <div className="p-6">
        <div className="mb-8">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Prompt History</h2>
          
          <BackendStatus />
          
          <div className="space-y-4">
            {promptHistory.map((entry) => (
              <div key={entry.id} className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
                <div className="flex justify-between items-start mb-4">
                  <div>
                    <h3 className="text-lg font-medium text-gray-900 mb-1">{entry.creator} Style</h3>
                    <p className="text-sm text-gray-600">
                      {entry.timestamp.toLocaleString()}
                    </p>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => copyToClipboard(entry.response.linkedinPosts[0])}
                      className="p-2 text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
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
                
                <div className="mb-4">
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Research Used:</h4>
                  <div className="flex flex-wrap gap-2">
                    {entry.research.map((topic, index) => (
                      <span key={index} className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs">
                        {topic}
                      </span>
                    ))}
                  </div>
                </div>
                
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
                
                <div className="border-t pt-4">
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Generated Content:</h4>
                  <div className="bg-gray-50 p-3 rounded-lg text-sm text-gray-700 whitespace-pre-line max-h-40 overflow-y-auto">
                    {entry.response.linkedinPosts && entry.response.linkedinPosts[0]}
                  </div>
                </div>
              </div>
            ))}
            
            {promptHistory.length === 0 && (
              <div className="text-center py-12">
                <BookOpen className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-600">No prompts generated yet. Start creating content to see your history here.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );

  const ChatTab = () => {
    const [localInputValue, setLocalInputValue] = useState('');
    const textareaRef = useRef(null);

    const handleLocalInputChange = (e) => {
      setLocalInputValue(e.target.value);
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
      }
    };

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
        {activeTab === 'history' && <HistoryTab />}
      </div>

      {showConfig && <ConfigPanel />}
    </div>
  );
};

export default LinkedInContentBot;