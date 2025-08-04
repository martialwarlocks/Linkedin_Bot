// SYNTAX FIX for the end of your React component
// The issue is that your code ends abruptly with an incomplete JSX element

// Your code ends with:
/*
                <span className="text-sm font-medium text-gray-700">Video Script</span>
*/

// But it should complete the JSX structure like this:

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