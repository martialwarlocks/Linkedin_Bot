// Supabase Configuration
import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.REACT_APP_SUPABASE_URL || 'https://qgyqkgmdnwfcnzzuzict.supabase.co'
const supabaseAnonKey = process.env.REACT_APP_SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFneXFrZ21kbndmY256enV6aWN0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTM3NzkyOTcsImV4cCI6MjA2OTM1NTI5N30.d9VFOHZsWDxhqY8UM0jvx5pGJVVOSkgHVFODL16Nc6s'

export const supabase = createClient(supabaseUrl, supabaseAnonKey)

// Database table schemas
export const TABLES = {
  RESEARCH: 'research',
  CREATOR_STYLES: 'creator_styles',
  DOCUMENTS: 'documents'
}

// Research item structure
export const RESEARCH_SCHEMA = {
  id: 'uuid',
  topic: 'text',
  findings: 'text',
  data: 'text',
  source: 'text',
  tags: 'text',
  created_at: 'timestamp',
  updated_at: 'timestamp'
}

// Creator style structure
export const CREATOR_STYLE_SCHEMA = {
  id: 'uuid',
  name: 'text',
  key: 'text',
  style: 'jsonb',
  created_at: 'timestamp',
  updated_at: 'timestamp'
}

// Document structure
export const DOCUMENT_SCHEMA = {
  id: 'uuid',
  filename: 'text',
  gcs_url: 'text',
  file_type: 'text',
  file_size: 'integer',
  chunks: 'jsonb',
  created_at: 'timestamp',
  updated_at: 'timestamp'
} 