import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, LinkedinIcon, Video, FileText, Database, TrendingUp, Users, Lightbulb, Copy, Settings, Plus, Trash2, Edit3, Save, X, Search, BookOpen, UserPlus } from 'lucide-react';

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
    openaiApiKey: '',
    model: 'gpt-4'
  });
  const [generatedContent, setGeneratedContent] = useState(null);
  const messagesEndRef = useRef(null);

  // Database states
  const [researchDatabase, setResearchDatabase] = useState([
    {
      id: 1,
      topic: 'AI in Content Creation',
      findings: 'AI-generated content is 3x faster to produce but requires human oversight for authenticity',
      source: 'Content Marketing Institute 2024',
      data: '67% of marketers now use AI tools regularly',
      tags: ['AI', 'Content', 'Marketing'],
      dateAdded: new Date('2024-01-15')
    },
    {
      id: 2,
      topic: 'LinkedIn Engagement Trends',
      findings: 'Video content gets 5x more engagement than text posts',
      source: 'LinkedIn Analytics Report 2024',
      data: 'Posts with personal stories get 30% more comments',
      tags: ['LinkedIn', 'Engagement', 'Video'],
      dateAdded: new Date('2024-01-10')
    },
    {
      id: 3,
      topic: 'Remote Work Productivity',
      findings: 'Hybrid workers report 23% higher job satisfaction',
      source: 'Future Work Institute Study',
      data: '89% of companies plan to continue hybrid models',
      tags: ['Remote Work', 'Productivity', 'Hybrid'],
      dateAdded: new Date('2024-01-05')
    }
  ]);

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

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Database operations
  const addResearch = () => {
    if (!newResearch.topic.trim()) return;

    const research = {
      id: Date.now(),
      ...newResearch,
      tags: newResearch.tags.split(',').map(tag => tag.trim()).filter(tag => tag),
      dateAdded: new Date()
    };

    setResearchDatabase(prev => [research, ...prev]);
    setNewResearch({ topic: '', findings: '', source: '', data: '', tags: '' });
  };

  const addCreator = () => {
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
  };

  const deleteResearch = (id) => {
    setResearchDatabase(prev => prev.filter(item => item.id !== id));
  };

  const deleteCreator = (id) => {
    setCreatorDatabase(prev => prev.filter(item => item.id !== id));
  };

  const updateResearch = (id, updatedData) => {
    setResearchDatabase(prev => prev.map(item => 
      item.id === id 
        ? { ...item, ...updatedData, tags: updatedData.tags.split(',').map(tag => tag.trim()).filter(tag => tag) }
        : item
    ));
    setEditingResearch(null);
  };

  const updateCreator = (id, updatedData) => {
    setCreatorDatabase(prev => prev.map(item => 
      item.id === id 
        ? {
            ...item,
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
        : item
    ));
    setEditingCreator(null);
  };

  // OpenAI Integration
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

Create both a LinkedIn post and a video script that incorporates this research data in the creator's authentic style.

Return your response in this JSON format:
{
  "linkedinPost": "the post content",
  "videoScript": "detailed video script with timestamps",
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
          temperature: 0.7,
          max_tokens: 2000
        })
      });

      if (!response.ok) {
        throw new Error(`OpenAI API error: ${response.status}`);
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

  const getFallbackContent = (prompt, researchData, creatorStyle) => {
    const research = researchData[0] || {
      topic: 'Business Innovation',
      findings: 'Companies that embrace change are 2x more likely to succeed',
      data: '78% of successful businesses pivoted during challenges'
    };

    let linkedinPost = '';
    let videoScript = '';

    if (creatorStyle.name === 'Gary Vaynerchuk') {
      linkedinPost = `Look, here's the thing about ${research.topic.toLowerCase()}...

I see so many people talking about this, but they're missing the REAL point.

${research.findings}

You know what this means? It means while everyone else is debating, the smart ones are DOING.

${research.data}

Stop overthinking. Start executing.

The market rewards action, not perfection.

What do you think? Drop your thoughts below 👇`;

      videoScript = `[0-3s] HOOK: "Why most people get ${research.topic.toLowerCase()} completely wrong..."
[4-8s] PERSONAL INTRO: "Gary V here, and I need to tell you something important"
[9-20s] THE INSIGHT: "${research.findings}"
[21-35s] THE DATA: "Look at this - ${research.data}"
[36-45s] THE CHALLENGE: "So here's what you need to do RIGHT NOW..."
[46-60s] CALL TO ACTION: "Comment below with your biggest challenge in this area"

Visual cues:
- High energy throughout
- Direct eye contact
- Hand gestures for emphasis
- Quick cuts between points
- End with pointing at camera`;
    } else {
      linkedinPost = `Here's what I learned about ${research.topic.toLowerCase()}:

${research.findings}

The thing is, most people see the data (${research.data}) and think it's about the numbers.

It's not.

It's about understanding that change isn't just inevitable - it's the only constant that matters.

Worth considering.`;

      videoScript = `[0-5s] HOOK: "Here's what everyone gets wrong about ${research.topic.toLowerCase()}"
[6-20s] THE INSIGHT: Share the key finding
[21-40s] THE DEEPER MEANING: Explain the real implication
[41-55s] THE BROADER CONTEXT: Connect to bigger picture
[56-60s] CLOSING THOUGHT: "Just saying."`;
    }

    return {
      linkedinPost,
      videoScript,
      hashtags: ['#LinkedIn', '#Content', '#Business', '#Innovation', '#Leadership'],
      engagement_tips: [
        'Post during peak hours (8-10 AM or 12-1 PM)',
        'Engage with comments within first 2 hours',
        'Ask a question to encourage responses'
      ]
    };
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
        item.tags.some(tag => lowerText.includes(tag.toLowerCase())) ||
        item.findings.toLowerCase().includes(lowerText)
      ).slice(0, 3);

      if (relevantResearch.length === 0) {
        relevantResearch.push(researchDatabase[0]);
      }

      let content;
      
      if (config.openaiApiKey && (text.includes('post') || text.includes('content') || text.includes('script'))) {
        try {
          content = await generateContentWithOpenAI(text, creatorStyle, relevantResearch);
        } catch (error) {
          console.error('OpenAI error, using fallback:', error);
          content = getFallbackContent(text, relevantResearch, creatorStyle);
        }
      } else {
        content = getFallbackContent(text, relevantResearch, creatorStyle);
      }

      setGeneratedContent(content);
      
      const botResponse = {
        id: Date.now() + 1,
        type: 'bot',
        text: `Content generated successfully in ${creatorStyle.name}'s style using your research data.`,
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

  const handleActionSelect = (actionKey) => {
    const actionMessages = {
      'video-script': "Create a video script for a LinkedIn reel about my recent findings",
      'browse-research': "Show me my research database and trending topics",
      'custom-style': "Analyze a custom creator's style and create similar content"
    };
    
    handleSendMessage(actionMessages[actionKey]);
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
    item.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()))
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

        <div className="mb-8">
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

        <div className="flex gap-3">
          <button
            onClick={() => setShowConfig(false)}
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
            LinkedIn Post
          </h4>
          <button
            onClick={() => copyToClipboard(content.linkedinPost)}
            className="flex items-center gap-2 px-3 py-1.5 bg-gray-100 text-gray-700 border border-gray-300 rounded-lg text-xs font-medium hover:bg-gray-200 transition-colors"
          >
            <Copy className="w-3.5 h-3.5" />
            Copy
          </button>
        </div>
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 text-sm text-gray-700 whitespace-pre-line leading-relaxed">
          {content.linkedinPost}
        </div>
      </div>

      <div className="mb-8">
        <div className="flex justify-between items-center mb-3">
          <h4 className="text-sm font-semibold text-gray-900 uppercase tracking-wide flex items-center gap-2">
            <Video className="w-4 h-4" />
            Video Script
          </h4>
          <button
            onClick={() => copyToClipboard(content.videoScript)}
            className="flex items-center gap-2 px-3 py-1.5 bg-gray-100 text-gray-700 border border-gray-300 rounded-lg text-xs font-medium hover:bg-gray-200 transition-colors"
          >
            <Copy className="w-3.5 h-3.5" />
            Copy
          </button>
        </div>
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 text-xs text-gray-700 whitespace-pre-line leading-relaxed font-mono">
          {content.videoScript}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h4 className="text-sm font-semibold text-gray-900 uppercase tracking-wide mb-3">
            Hashtags
          </h4>
          <div className="flex flex-wrap gap-2">
            {content.hashtags.map((tag, index) => (
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
            {content.engagement_tips.map((tip, index) => (
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
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500 min-h-20"
              />
            </div>
            <div className="mb-4">
              <textarea
                placeholder="Supporting Data/Statistics"
                value={newResearch.data}
                onChange={(e) => setNewResearch(prev => ({ ...prev, data: e.target.value }))}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500 min-h-16"
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

          {/* Research List */}
          <div className="space-y-4">
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
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500 min-h-20"
                    />
                    <textarea
                      value={research.data}
                      onChange={(e) => setResearchDatabase(prev => prev.map(r => 
                        r.id === research.id ? { ...r, data: e.target.value } : r
                      ))}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500 min-h-16"
                    />
                    <input
                      type="text"
                      value={research.tags.join(', ')}
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
                      {research.tags.map((tag, index) => (
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
                className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500 min-h-16"
              />
              <button
                onClick={addCreator}
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
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:border-gray-500 min-h-16"
                    />
                    <div className="flex gap-2">
                      <button
                        onClick={() => updateCreator(creator.id, {
                          name: creator.name,
                          tone: creator.style.tone,
                          structure: creator.style.structure,
                          language: creator.style.language,
                          length: creator.style.length,
                          hooks: creator.style.hooks.join(', '),
                          endings: creator.style.endings.join(', '),
                          characteristics: creator.style.characteristics
                        })}
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
                          onClick={() => deleteCreator(creator.id)}
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
                      onClick={() => copyToClipboard(entry.response.linkedinPost)}
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
                    {entry.response.linkedinPost}
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
              onClick={() => handleActionSelect('video-script')}
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
                  <div className="p-3 rounded-lg bg-gray-100 border border-gray-200 flex items-center justify-center flex-shrink-0">
                    <Bot className="w-4.5 h-4.5 text-gray-600" />
                  </div>
                )}
                
                <div className={`px-5 py-4 rounded-xl ${
                  message.type === 'user' 
                    ? 'bg-gray-900 text-white' 
                    : 'bg-gray-50 text-gray-700 border border-gray-200'
                } ${message.type === 'user' ? 'rounded-br-sm' : 'rounded-bl-sm'}`}>
                  <div className="text-sm leading-relaxed whitespace-pre-line">
                    {message.text}
                  </div>
                  <div className={`text-xs mt-3 ${
                    message.type === 'user' ? 'text-gray-300' : 'text-gray-500'
                  }`}>
                    {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </div>
                </div>
                
                {message.type === 'user' && (
                  <div className="p-3 rounded-lg bg-gray-100 border border-gray-200 flex items-center justify-center flex-shrink-0">
                    <User className="w-4.5 h-4.5 text-gray-600" />
                  </div>
                )}
              </div>
            </div>
          ))}

          {generatedContent && <ContentDisplay content={generatedContent} />}
          
          {isTyping && (
            <div className="flex w-full justify-start">
              <div className="flex items-start gap-3 max-w-3xl">
                <div className="p-3 rounded-lg bg-gray-100 border border-gray-200 flex items-center justify-center">
                  <Bot className="w-4.5 h-4.5 text-gray-600" />
                </div>
                <div className="px-5 py-4 rounded-xl bg-gray-50 border border-gray-200 rounded-bl-sm">
                  <div className="flex gap-1 items-center">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse"></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse" style={{ animationDelay: '0.4s' }}></div>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input */}
      <div className="bg-white border-t border-gray-200 px-6 py-6">
        <div className="flex gap-4 items-end">
          <div className="flex-1 relative">
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Describe the content you'd like to create..."
              className="w-full border border-gray-300 rounded-xl px-4 py-3 resize-none text-sm leading-relaxed min-h-12 max-h-32 focus:outline-none focus:border-gray-500 transition-colors bg-white"
              rows="1"
            />
          </div>
          <button
            onClick={() => handleSendMessage()}
            disabled={!inputValue.trim()}
            className={`p-4 rounded-xl min-w-12 min-h-12 flex items-center justify-center transition-colors ${
              inputValue.trim() 
                ? 'bg-gray-900 text-white hover:bg-gray-800' 
                : 'bg-gray-100 text-gray-400 cursor-not-allowed'
            }`}
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
        <p className="text-xs text-gray-500 text-center mt-4">
          Create professional LinkedIn content with AI assistance
        </p>
      </div>
    </>
  );

  return (
    <div className="flex flex-col h-screen max-w-6xl mx-auto bg-white font-system">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="bg-gray-900 p-3 rounded-lg flex items-center justify-center">
              <LinkedinIcon className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-semibold text-gray-900 tracking-tight">
                Content Creator Assistant
              </h1>
              <p className="text-sm text-gray-600">
                Professional content generation with AI
              </p>
            </div>
          </div>
          <button
            onClick={() => setShowConfig(true)}
            className="bg-gray-100 border border-gray-300 rounded-lg p-3 hover:bg-gray-200 transition-colors"
          >
            <Settings className="w-4.5 h-4.5 text-gray-600" />
          </button>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="bg-white border-b border-gray-200 px-6">
        <div className="flex space-x-8">
          {[
            { id: 'chat', label: 'Chat', icon: Bot },
            { id: 'research', label: 'Research', icon: Database },
            { id: 'creators', label: 'Creators', icon: Users },
            { id: 'history', label: 'History', icon: BookOpen }
          ].map((tab) => {
            const IconComponent = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 py-4 px-1 border-b-2 text-sm font-medium transition-colors ${
                  activeTab === tab.id
                    ? 'border-gray-900 text-gray-900'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <IconComponent className="w-4 h-4" />
                {tab.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Tab Content */}
      {activeTab === 'chat' && <ChatTab />}
      {activeTab === 'research' && <ResearchTab />}
      {activeTab === 'creators' && <CreatorTab />}
      {activeTab === 'history' && <HistoryTab />}

      {showConfig && <ConfigPanel />}
    </div>
  );
};

export default LinkedInContentBot;