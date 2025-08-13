-- Supabase Database Schema for LinkedIn Content Creator Bot

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Research table
CREATE TABLE IF NOT EXISTS research (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    topic TEXT NOT NULL,
    findings TEXT NOT NULL,
    data TEXT,
    source TEXT,
    tags TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Creator styles table
CREATE TABLE IF NOT EXISTS creator_styles (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    key TEXT NOT NULL UNIQUE,
    style JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    filename TEXT NOT NULL,
    gcs_url TEXT NOT NULL,
    file_type TEXT,
    file_size INTEGER,
    chunks JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Content generation history table
CREATE TABLE IF NOT EXISTS content_history (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    prompt TEXT NOT NULL,
    creator_style TEXT NOT NULL,
    creator_name TEXT NOT NULL,
    generated_content JSONB NOT NULL,
    research_used JSONB,
    backend_sources JSONB,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_research_topic ON research(topic);
CREATE INDEX IF NOT EXISTS idx_research_tags ON research USING GIN(to_tsvector('english', tags));
CREATE INDEX IF NOT EXISTS idx_creator_styles_key ON creator_styles(key);
CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename);
CREATE INDEX IF NOT EXISTS idx_content_history_creator ON content_history(creator_style);
CREATE INDEX IF NOT EXISTS idx_content_history_date ON content_history(created_at);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_research_updated_at BEFORE UPDATE ON research
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_creator_styles_updated_at BEFORE UPDATE ON creator_styles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_content_history_updated_at BEFORE UPDATE ON content_history
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert default creator styles
INSERT INTO creator_styles (name, key, style) VALUES
(
    'Gary Vaynerchuk',
    'gary-v',
    '{
        "tone": "Direct, energetic, and motivational",
        "structure": "Hook, story, value, call-to-action",
        "language": "Casual, conversational, uses emojis",
        "length": "Medium to long",
        "hooks": ["Think about this", "Listen up", "Here''s the truth"],
        "endings": ["What do you think?", "Drop a comment below", "Let''s discuss"],
        "characteristics": "High energy, personal stories, actionable advice"
    }'
),
(
    'Simon Sinek',
    'simon-sinek',
    '{
        "tone": "Thoughtful, inspiring, and philosophical",
        "structure": "Why, how, what framework",
        "language": "Professional, clear, purpose-driven",
        "length": "Medium",
        "hooks": ["Why do we do what we do?", "The question is", "Consider this"],
        "endings": ["What''s your why?", "Find your purpose", "Start with why"],
        "characteristics": "Purpose-driven, leadership focus, inspiring questions"
    }'
),
(
    'Seth Godin',
    'seth-godin',
    '{
        "tone": "Insightful, challenging, and educational",
        "structure": "Observation, insight, lesson",
        "language": "Clear, concise, thought-provoking",
        "length": "Short to medium",
        "hooks": ["Here''s what I noticed", "The thing about", "Consider this"],
        "endings": ["What do you think?", "The choice is yours", "Your turn"],
        "characteristics": "Short insights, challenging assumptions, educational"
    }'
)
ON CONFLICT (key) DO NOTHING;

-- Enable Row Level Security (RLS)
ALTER TABLE research ENABLE ROW LEVEL SECURITY;
ALTER TABLE creator_styles ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- Create policies for public read access
CREATE POLICY "Allow public read access to research" ON research
    FOR SELECT USING (true);

CREATE POLICY "Allow public read access to creator_styles" ON creator_styles
    FOR SELECT USING (true);

CREATE POLICY "Allow public read access to documents" ON documents
    FOR SELECT USING (true);

-- Create policies for authenticated users to insert/update/delete
CREATE POLICY "Allow authenticated users to manage research" ON research
    FOR ALL USING (auth.role() = 'authenticated');

CREATE POLICY "Allow authenticated users to manage creator_styles" ON creator_styles
    FOR ALL USING (auth.role() = 'authenticated');

CREATE POLICY "Allow authenticated users to manage documents" ON documents
    FOR ALL USING (auth.role() = 'authenticated');

-- Enable RLS for content_history
ALTER TABLE content_history ENABLE ROW LEVEL SECURITY;

-- Create policies for content_history
CREATE POLICY "Allow public read access to content_history" ON content_history
    FOR SELECT USING (true);

CREATE POLICY "Allow authenticated users to manage content_history" ON content_history
    FOR ALL USING (auth.role() = 'authenticated'); 