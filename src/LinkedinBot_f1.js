import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, LinkedinIcon, Video, FileText, Database, TrendingUp, Users, Lightbulb, Copy, Settings, Plus, Trash2, Edit3, Save, X, Search, BookOpen, UserPlus, ChevronLeft, ChevronRight, Check } from 'lucide-react';

// Supabase client initialization
const createSupabaseClient = () => {
  const supabaseUrl = localStorage.getItem('supabaseUrl') || '';
  const supabaseKey = localStorage.getItem('supabaseKey') || '';
  
  if (!supabaseUrl || !supabaseKey) {
    return null;
  }

  // Simple Supabase client implementation
  return {
    from: (table) => ({
      select: async (columns = '*') => {
        try {
          const response = await fetch(`${supabaseUrl}/rest/v1/${table}?select=${columns}`, {
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
        eq: (column, value) => {
          return fetch(`${supabaseUrl}/rest/v1/${table}?${column}=eq.${value}`, {
            method: 'PATCH',
            headers: {
              'apikey': supabaseKey,
              'Authorization': `Bearer ${supabaseKey}`,
              'Content-Type': 'application/json',
              'Prefer': 'return=representation'
            },
            body: JSON.stringify(data)
          })
          .then(res => res.json())
          .then(result => ({ data: result, error: null }))
          .catch(error => ({ data: null, error }));
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
      text: "Welcome to the LinkedIn Content Creator Assistant. I help you create professional content in the style of industry leaders using your research data. How can I assist you today?",
      timestamp: new Date()
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [showConfig, setShowConfig] = useState(false);
  const [config, setConfig] = useState({
    openaiApiKey: localStorage.getItem('openaiApiKey') || '',
    model: localStorage.getItem('aiModel') || 'gpt-4',
    supabaseUrl: localStorage.getItem('supabaseUrl') || '',
    supabaseKey: localStorage.getItem('supabaseKey') || ''
  });
  const [generatedContent, setGeneratedContent] = useState(null);
  const [selectedOptions, setSelectedOptions] = useState({
    linkedinPost: 0,
    videoScript: 0
  });
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

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
  const [searchTerm, setSearchTerm] = useState('');
  const [supabaseClient, setSupabaseClient] = useState(null);
  const [isLoadingResearch, setIsLoadingResearch] = useState(false);

  // Form states
  const [newResearch, setNewResearch] = useState({
    topic: '',
    findings: '',
    source: '',
    data: '',
    tags: ''
  });

  const [newCreator, setNewCreator] = useState({
    name: '',
    tone: '',
    structure: '',
    language: '',
    length: '',
    hooks: '',
    endings: '',
    characteristics: ''
  });

  const [editingResearch, setEditingResearch] = useState(null);
  const [editingCreator, setEditingCreator] = useState(null);

  // Initialize Supabase client
  useEffect(() => {
    if (config.supabaseUrl && config.supabaseKey) {
      const client = createSupabaseClient();
      setSupabaseClient(client);
      loadResearchFromSupabase(client);
    }
  }, [config.supabaseUrl, config.supabaseKey]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [inputValue]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Supabase operations
  const loadResearchFromSupabase = async (client = supabaseClient) => {
    if (!client) return;
    
    setIsLoadingResearch(true);
    try {
      const { data, error } = await client.from('research').select('*');
      if (error) throw error;
      
      setResearchDatabase(data.map(item => ({
        ...item,
        tags: typeof item.tags === 'string' ? item.tags.split(',').map(t => t.trim()) : (item.tags || []),
        dateAdded: new Date(item.created_at || item.dateAdded)
      })));
    } catch (error) {
      console.error('Error loading research:', error);
    } finally {
      setIsLoadingResearch(false);
    }
  };

  // Database operations
  const addResearch = async () => {
    if (!newResearch.topic.trim()) return;

    const research = {
      topic: newResearch.topic,
      findings: newResearch.findings,
      source: newResearch.source,
      data: newResearch.data,
      tags: newResearch.tags.split(',').map(tag => tag.trim()).filter(tag => tag).join(','),
      created_at: new Date().toISOString()
    };

    if (supabaseClient) {
      try {
        const { data, error } = await supabaseClient.from('research').insert([research]);
        if (error) throw error;
        
        await loadResearchFromSupabase();
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
    } else {
      // Local storage fallback
      setResearchDatabase(prev => [{
        ...research,
        id: Date.now(),
        tags: research.tags.split(',').map(t => t.trim()),
        dateAdded: new Date()
      }, ...prev]);
    }

    setNewResearch({ topic: '', findings: '', source: '', data: '', tags: '' });
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
      ...updatedData,
      tags: typeof updatedData.tags === 'string' 
        ? updatedData.tags.split(',').map(tag => tag.trim()).filter(tag => tag).join(',')
        : (Array.isArray(updatedData.tags) ? updatedData.tags.join(',') : updatedData.tags)
    };

    if (supabaseClient) {
      try {
        const { error } = await supabaseClient.from('research').update(updated).eq('id', id);
        if (error) throw error;
        
        await loadResearchFromSupabase();
      } catch (error) {
        console.error('Error updating research:', error);
        setResearchDatabase(prev => prev.map(item => 
          item.id === id ? { ...item, ...updated, tags: updated.tags.split(',').map(t => t.trim()) } : item
        ));
      }
    } else {
      setResearchDatabase(prev => prev.map(item => 
        item.id === id ? { ...item, ...updated, tags: updated.tags.split(',').map(t => t.trim()) } : item
      ));
    }
    
    setEditingResearch(null);
  };

  // OpenAI Integration for multiple variations
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

Research data to incorporate:
${relevantResearch.map(r => `
Topic: ${r.topic}
Findings: ${r.findings}
Data: ${r.data}
Source: ${r.source}
`).join('\n')}

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

  const handleSendMessage = async (messageText = null) => {
    const text = messageText || inputValue.trim();
    if (!text) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      text: text,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsTyping(true);
    setSelectedOptions({ linkedinPost: 0, videoScript: 0 });

    try {
      let creatorStyle = creatorDatabase.find(c => c.key === 'gary-v');
      
      // Determine creator from message
      const lowerText = text.toLowerCase();
      for (const creator of creatorDatabase) {
        if (lowerText.includes(creator.name.toLowerCase()) || 
            lowerText.includes(creator.key)) {
          creatorStyle = creator;
          break;
        }
      }

      // Find relevant research
      const relevantResearch = researchDatabase.filter(item => 
        item.topic.toLowerCase().includes(lowerText) ||
        (Array.isArray(item.tags) && item.tags.some(tag => lowerText.includes(tag.toLowerCase()))) ||
        item.findings.toLowerCase().includes(lowerText)
      ).slice(0, 3);

      if (relevantResearch.length === 0 && researchDatabase.length > 0) {
        relevantResearch.push(researchDatabase[0]);
      }

      if (!config.openaiApiKey) {
        throw new Error('Please configure OpenAI API key in settings to generate content');
      }

      const content = await generateContentWithOpenAI(text, creatorStyle, relevantResearch);

      setGeneratedContent(content);
      
      const botResponse = {
        id: Date.now() + 1,
        type: 'bot',
        text: `I've created 2 variations for each content type in ${creatorStyle.name}'s style using your research data. Use the navigation arrows to choose your preferred options.`,
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
    const message = `Create a ${creator.name} style LinkedIn post using my latest research`;
    handleSendMessage(message);
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const filteredResearch = researchDatabase.filter(item =>
    item.topic.toLowerCase().includes(searchTerm.toLowerCase()) ||
    (Array.isArray(item.tags) && item.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase())))
  );

  const ConfigPanel = () => (
    <div className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center z-50" style={{ backdropFilter: 'blur(4px)' }}>
      <div className="bg-white p-8 rounded-xl w-11/12 max-w-md shadow-2xl border border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900 mb-6">Configuration</h3>
        
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
              localStorage.setItem('openaiApiKey', config.openaiApiKey);
              localStorage.setItem('aiModel', config.model);
              localStorage.setItem('supabaseUrl', config.supabaseUrl);
              localStorage.setItem('supabaseKey', config.supabaseKey);
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
        <LinkedinIcon className="w-5 h-5 text-blue-600" />
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

  const ResearchTab = () => (
    <div className="flex-1 overflow-y-auto bg-gray-50">
      <div className="p-6">
        <div className="mb-8">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Research Database</h2>
          
          {/* Supabase Connection Status */}
          <div className="mb-6 p-3 rounded-lg bg-gray-100 border border-gray-200 flex items-center gap-2 text-sm">
            <div className={`w-2 h-2 rounded-full ${supabaseClient ? 'bg-green-500' : 'bg-red-500'}`}></div>
            <span className="text-gray-600">
              {supabaseClient ? 'Connected to Supabase' : 'Using local storage'}
            </span>
          </div>
          
          {/* Search */}
          <div className="relative mb-6">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search research..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
            />
          </div>

          {/* Add New Research */}
          <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200 mb-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Add New Research</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
              <input
                type="text"
                placeholder="Research Topic"
                value={newResearch.topic}
                onChange={(e) => setNewResearch(prev => ({ ...prev, topic: e.target.value }))}
                className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
              />
              <input
                type="text"
                placeholder="Source"
                value={newResearch.source}
                onChange={(e) => setNewResearch(prev => ({ ...prev, source: e.target.value }))}
                className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
              />
            </div>
            <div className="mb-4">
              <textarea
                placeholder="Key Findings"
                value={newResearch.findings}
                onChange={(e) => setNewResearch(prev => ({ ...prev, findings: e.target.value }))}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500 min-h-24 resize-y"
                rows="4"
              />
            </div>
            <div className="mb-4">
              <textarea
                placeholder="Supporting Data/Statistics"
                value={newResearch.data}
                onChange={(e) => setNewResearch(prev => ({ ...prev, data: e.target.value }))}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500 min-h-20 resize-y"
                rows="3"
              />
            </div>
            <div className="flex gap-4">
              <input
                type="text"
                placeholder="Tags (comma separated)"
                value={newResearch.tags}
                onChange={(e) => setNewResearch(prev => ({ ...prev, tags: e.target.value }))}
                className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
              />
              <button
                onClick={addResearch}
                className="px-6 py-2 bg-gray-900 text-white rounded-lg text-sm font-medium hover:bg-gray-800 transition-colors flex items-center gap-2"
              >
                <Plus className="w-4 h-4" />
                Add Research
              </button>
            </div>
          </div>

          {/* Research Preview */}
          {newResearch.topic && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
              <h4 className="text-sm font-medium text-blue-900 mb-2">Preview</h4>
              <div className="space-y-2 text-sm text-blue-800">
                <p><strong>Topic:</strong> {newResearch.topic}</p>
                {newResearch.findings && <p><strong>Findings:</strong> {newResearch.findings}</p>}
                {newResearch.source && <p><strong>Source:</strong> {newResearch.source}</p>}
                {newResearch.data && <p><strong>Data:</strong> {newResearch.data}</p>}
                {newResearch.tags && (
                  <div className="flex items-center gap-2">
                    <strong>Tags:</strong>
                    <div className="flex flex-wrap gap-1">
                      {newResearch.tags.split(',').map((tag, index) => tag.trim()).filter(tag => tag).map((tag, index) => (
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

          {/* Loading State */}
          {isLoadingResearch && (
            <div className="flex items-center justify-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
            </div>
          )}

          {/* Research List */}
          <div className="space-y-4">
            {filteredResearch.length === 0 && !isLoadingResearch && (
              <div className="text-center py-12 bg-white rounded-lg border border-gray-200">
                <Database className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-600">No research found. Add your first research above!</p>
              </div>
            )}
            
            {filteredResearch.map((research) => (
              <div key={research.id} className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
                {editingResearch === research.id ? (
                  <div className="space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <input
                        type="text"
                        value={research.topic}
                        onChange={(e) => setResearchDatabase(prev => prev.map(r => 
                          r.id === research.id ? { ...r, topic: e.target.value } : r
                        ))}
                        className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
                      />
                      <input
                        type="text"
                        value={research.source}
                        onChange={(e) => setResearchDatabase(prev => prev.map(r => 
                          r.id === research.id ? { ...r, source: e.target.value } : r
                        ))}
                        className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
                      />
                    </div>
                    <textarea
                      value={research.findings}
                      onChange={(e) => setResearchDatabase(prev => prev.map(r => 
                        r.id === research.id ? { ...r, findings: e.target.value } : r
                      ))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500 min-h-24 resize-y"
                      rows="4"
                    />
                    <textarea
                      value={research.data}
                      onChange={(e) => setResearchDatabase(prev => prev.map(r => 
                        r.id === research.id ? { ...r, data: e.target.value } : r
                      ))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500 min-h-20 resize-y"
                      rows="3"
                    />
                    <input
                      type="text"
                      value={Array.isArray(research.tags) ? research.tags.join(', ') : research.tags}
                      onChange={(e) => setResearchDatabase(prev => prev.map(r => 
                        r.id === research.id ? { ...r, tags: e.target.value.split(',').map(t => t.trim()) } : r
                      ))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
                    />
                    <div className="flex gap-2">
                      <button
                        onClick={() => updateResearch(research.id, research)}
                        className="px-4 py-2 bg-green-600 text-white rounded-lg text-sm font-medium hover:bg-green-700 transition-colors flex items-center gap-2"
                      >
                        <Save className="w-4 h-4" />
                        Save
                      </button>
                      <button
                        onClick={() => setEditingResearch(null)}
                        className="px-4 py-2 bg-gray-600 text-white rounded-lg text-sm font-medium hover:bg-gray-700 transition-colors flex items-center gap-2"
                      >
                        <X className="w-4 h-4" />
                        Cancel
                      </button>
                    </div>
                  </div>
                ) : (
                  <div>
                    <div className="flex justify-between items-start mb-4">
                      <div>
                        <h3 className="text-lg font-medium text-gray-900 mb-2">{research.topic}</h3>
                        <p className="text-sm text-gray-600 mb-3">{research.source}</p>
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
                    <p className="text-xs text-gray-500">
                      Added: {research.dateAdded.toLocaleDateString()}
                    </p>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );

  const CreatorTab = () => (
    <div className="flex-1 overflow-y-auto bg-gray-50">
      <div className="p-6">
        <div className="mb-8">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Creator Profiles</h2>
          
          {/* Add New Creator */}
          <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200 mb-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Add New Creator</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
              <input
                type="text"
                placeholder="Creator Name"
                value={newCreator.name}
                onChange={(e) => setNewCreator(prev => ({ ...prev, name: e.target.value }))}
                className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
              />
              <input
                type="text"
                placeholder="Tone (e.g., Direct, passionate, no-nonsense)"
                value={newCreator.tone}
                onChange={(e) => setNewCreator(prev => ({ ...prev, tone: e.target.value }))}
                className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
              />
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
              <input
                type="text"
                placeholder="Post Structure"
                value={newCreator.structure}
                onChange={(e) => setNewCreator(prev => ({ ...prev, structure: e.target.value }))}
                className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
              />
              <input
                type="text"
                placeholder="Language Style"
                value={newCreator.language}
                onChange={(e) => setNewCreator(prev => ({ ...prev, language: e.target.value }))}
                className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
              />
            </div>
            <div className="mb-4">
              <input
                type="text"
                placeholder="Typical Post Length"
                value={newCreator.length}
                onChange={(e) => setNewCreator(prev => ({ ...prev, length: e.target.value }))}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
              />
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
              <input
                type="text"
                placeholder="Common Hooks (comma separated)"
                value={newCreator.hooks}
                onChange={(e) => setNewCreator(prev => ({ ...prev, hooks: e.target.value }))}
                className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
              />
              <input
                type="text"
                placeholder="Common Endings (comma separated)"
                value={newCreator.endings}
                onChange={(e) => setNewCreator(prev => ({ ...prev, endings: e.target.value }))}
                className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
              />
            </div>
            <div className="flex gap-4">
              <textarea
                placeholder="Key Characteristics"
                value={newCreator.characteristics}
                onChange={(e) => setNewCreator(prev => ({ ...prev, characteristics: e.target.value }))}
                className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500 min-h-20 resize-y"
                rows="3"
              />
              <button
                onClick={() => {
                  if (!newCreator.name.trim()) return;
                  
                  const creator = {
                    id: Date.now(),
                    name: newCreator.name,
                    key: newCreator.name.toLowerCase().replace(/\s+/g, '-'),
                    icon: UserPlus,
                    style: {
                      tone: newCreator.tone,
                      structure: newCreator.structure,
                      language: newCreator.language,
                      length: newCreator.length,
                      hooks: newCreator.hooks.split(',').map(h => h.trim()).filter(h => h),
                      endings: newCreator.endings.split(',').map(e => e.trim()).filter(e => e),
                      characteristics: newCreator.characteristics
                    }
                  };
                  
                  setCreatorDatabase(prev => [creator, ...prev]);
                  setNewCreator({ name: '', tone: '', structure: '', language: '', length: '', hooks: '', endings: '', characteristics: '' });
                }}
                className="px-6 py-2 bg-gray-900 text-white rounded-lg text-sm font-medium hover:bg-gray-800 transition-colors flex items-center gap-2"
              >
                <Plus className="w-4 h-4" />
                Add Creator
              </button>
            </div>
          </div>

          {/* Creator List */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {creatorDatabase.map((creator) => (
              <div key={creator.id} className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
                {editingCreator === creator.id ? (
                  <div className="space-y-4">
                    <input
                      type="text"
                      value={creator.name}
                      onChange={(e) => setCreatorDatabase(prev => prev.map(c => 
                        c.id === creator.id ? { ...c, name: e.target.value } : c
                      ))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
                    />
                    <input
                      type="text"
                      value={creator.style.tone}
                      onChange={(e) => setCreatorDatabase(prev => prev.map(c => 
                        c.id === creator.id ? { ...c, style: { ...c.style, tone: e.target.value } } : c
                      ))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
                    />
                    <input
                      type="text"
                      value={creator.style.structure}
                      onChange={(e) => setCreatorDatabase(prev => prev.map(c => 
                        c.id === creator.id ? { ...c, style: { ...c.style, structure: e.target.value } } : c
                      ))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
                    />
                    <input
                      type="text"
                      value={creator.style.language}
                      onChange={(e) => setCreatorDatabase(prev => prev.map(c => 
                        c.id === creator.id ? { ...c, style: { ...c.style, language: e.target.value } } : c
                      ))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
                    />
                    <input
                      type="text"
                      value={creator.style.hooks.join(', ')}
                      onChange={(e) => setCreatorDatabase(prev => prev.map(c => 
                        c.id === creator.id ? { ...c, style: { ...c.style, hooks: e.target.value.split(',').map(h => h.trim()) } } : c
                      ))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
                    />
                    <input
                      type="text"
                      value={creator.style.endings.join(', ')}
                      onChange={(e) => setCreatorDatabase(prev => prev.map(c => 
                        c.id === creator.id ? { ...c, style: { ...c.style, endings: e.target.value.split(',').map(e => e.trim()) } } : c
                      ))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500"
                    />
                    <textarea
                      value={creator.style.characteristics}
                      onChange={(e) => setCreatorDatabase(prev => prev.map(c => 
                        c.id === creator.id ? { ...c, style: { ...c.style, characteristics: e.target.value } } : c
                      ))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500 min-h-20 resize-y"
                      rows="3"
                    />
                    <div className="flex gap-2">
                      <button
                        onClick={() => {
                          const updatedData = {
                            name: creator.name,
                            tone: creator.style.tone,
                            structure: creator.style.structure,
                            language: creator.style.language,
                            length: creator.style.length,
                            hooks: creator.style.hooks.join(', '),
                            endings: creator.style.endings.join(', '),
                            characteristics: creator.style.characteristics
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
                                    hooks: updatedData.hooks.split(',').map(h => h.trim()).filter(h => h),
                                    endings: updatedData.endings.split(',').map(e => e.trim()).filter(e => e),
                                    characteristics: updatedData.characteristics
                                  }
                                }
                              : c
                          ));
                          setEditingCreator(null);
                        }}
                        className="px-4 py-2 bg-green-600 text-white rounded-lg text-sm font-medium hover:bg-green-700 transition-colors flex items-center gap-2"
                      >
                        <Save className="w-4 h-4" />
                        Save
                      </button>
                      <button
                        onClick={() => setEditingCreator(null)}
                        className="px-4 py-2 bg-gray-600 text-white rounded-lg text-sm font-medium hover:bg-gray-700 transition-colors flex items-center gap-2"
                      >
                        <X className="w-4 h-4" />
                        Cancel
                      </button>
                    </div>
                  </div>
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

  const HistoryTab = () => (
    <div className="flex-1 overflow-y-auto bg-gray-50">
      <div className="p-6">
        <div className="mb-8">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Prompt History</h2>
          
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
                  <p className="text-sm text-gray-600 bg-gray-50 p-3 rounded-lg">{entry.prompt}</p>
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

  const ChatTab = () => (
    <>
      {/* Creator Selection */}
      <div className="bg-gray-50 border-b border-gray-200 px-6 py-5">
        <div className="mb-4">
          <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-3">
            Creator Styles
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {creatorDatabase.slice(0, 6).map((creator) => {
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
              onClick={() => {
                const message = "Create a video script for a LinkedIn reel about my recent findings";
                handleSendMessage(message);
              }}
              className="flex items-center gap-3 p-3.5 bg-white border border-gray-200 rounded-lg hover:border-gray-300 hover:transform hover:-translate-y-0.5 hover:shadow-md transition-all text-left"
            >
              <Video className="w-4.5 h-4.5 text-gray-600" />
              <span className="text-sm font-medium text-gray-700">Video Script</span>
            </button>
            <button
              onClick={() => setActiveTab('research')}
              className="flex items-center gap-3 p-3.5 bg-white border border-gray-200 rounded-lg hover:border-gray-300 hover:transform hover:-translate-y-0.5 hover:shadow-md transition-all text-left"
            >
              <Database className="w-4.5 h-4.5 text-gray-600" />
              <span className="text-sm font-medium text-gray-700">Browse Research</span>
            </button>
            <button
              onClick={() => setActiveTab('creators')}
              className="flex items-center gap-3 p-3.5 bg-white border border-gray-200 rounded-lg hover:border-gray-300 hover:transform hover:-translate-y-0.5 hover:shadow-md transition-all text-left"
            >
              <FileText className="w-4.5 h-4.5 text-gray-600" />
              <span className="text-sm font-medium text-gray-700">Manage Creators</span>
            </button>
          </div>
        </div>
      </div>

      {/* Messages */}
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

          {/* Generated Content Display */}
          {generatedContent && <ContentDisplay content={generatedContent} />}

          {/* Typing Indicator */}
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

      {/* Input Area */}
      <div className="border-t border-gray-200 px-6 py-4 bg-white">
        <div className="flex items-end gap-3">
          <div className="flex-1">
            <textarea
              ref={textareaRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask me to create content in any creator's style using your research..."
              className="w-full px-4 py-3 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500 resize-none min-h-12 max-h-32"
              rows="1"
            />
          </div>
          <button
            onClick={() => handleSendMessage()}
            disabled={!inputValue.trim() || isTyping}
            className="px-4 py-3 bg-gray-900 text-white rounded-lg hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </>
  );

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <div className="w-64 bg-white border-r border-gray-200 flex flex-col">
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
              <LinkedinIcon className="w-4 h-4 text-white" />
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

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {activeTab === 'chat' && <ChatTab />}
        {activeTab === 'research' && <ResearchTab />}
        {activeTab === 'creators' && <CreatorTab />}
        {activeTab === 'history' && <HistoryTab />}
      </div>

      {/* Config Modal */}
      {showConfig && <ConfigPanel />}
    </div>
  );
};

export default LinkedInContentBot;