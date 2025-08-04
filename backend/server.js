const express = require('express');
const multer = require('multer');
const { Storage } = require('@google-cloud/storage');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Initialize Google Cloud Storage
const storage = new Storage({
  keyFilename: process.env.GOOGLE_CLOUD_KEY_FILE || './google-cloud-key.json',
  projectId: process.env.GOOGLE_CLOUD_PROJECT_ID || 'your-project-id'
});

// LinkedIn Bot specific bucket
const LINKEDIN_BOT_BUCKET = process.env.LINKEDIN_BOT_BUCKET || 'linkedin-bot-documents';
const bucket = storage.bucket(LINKEDIN_BOT_BUCKET);

// Ensure bucket exists
async function ensureBucketExists() {
  try {
    const [exists] = await bucket.exists();
    if (!exists) {
      await bucket.create();
      console.log(`Bucket ${LINKEDIN_BOT_BUCKET} created successfully`);
    }
  } catch (error) {
    console.error('Error ensuring bucket exists:', error);
  }
}

ensureBucketExists();

// Multer configuration for file uploads
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit
  },
  fileFilter: (req, file, cb) => {
    const allowedTypes = [
      'application/pdf',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      'application/vnd.openxmlformats-officedocument.presentationml.presentation',
      'text/plain',
      'text/html'
    ];
    
    if (allowedTypes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type'), false);
    }
  }
});

// Helper function to extract text from different file types
async function extractTextFromFile(fileBuffer, mimeType) {
  // This is a simplified text extraction
  // In production, you'd use proper libraries like pdf-parse, mammoth, etc.
  
  if (mimeType === 'text/plain' || mimeType === 'text/html') {
    return fileBuffer.toString('utf-8');
  }
  
  // For other file types, return a placeholder
  // In production, implement proper text extraction
  return `Document content extracted from ${mimeType} file. This is a placeholder for the actual text extraction.`;
}

// Helper function to chunk text
function chunkText(text, chunkSize = 1000, overlap = 200) {
  const chunks = [];
  let start = 0;
  
  while (start < text.length) {
    const end = Math.min(start + chunkSize, text.length);
    const chunk = text.substring(start, end);
    
    chunks.push({
      id: uuidv4(),
      text: chunk,
      start: start,
      end: end
    });
    
    start = end - overlap;
  }
  
  return chunks;
}

// Routes

// Root endpoint (redirects to health check)
app.get('/', (req, res) => {
  res.json({ 
    status: 'online', 
    service: 'LinkedIn Bot Document Manager',
    message: 'Document management API is running',
    endpoints: {
      health: '/health',
      upload: '/upload',
      documents: '/documents',
      search: '/search',
      ask: '/ask',
      crawl: '/crawl',
      chunks: '/chunks',
      refresh: '/refresh'
    }
  });
});

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'OK', service: 'LinkedIn Bot Document Manager' });
});

// Upload document
app.post('/upload', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    const file = req.file;
    const documentId = uuidv4();
    const fileName = `${documentId}_${file.originalname}`;
    
    // Upload file to Google Cloud Storage
    const blob = bucket.file(fileName);
    await blob.save(file.buffer, {
      metadata: {
        contentType: file.mimetype,
        metadata: {
          originalName: file.originalname,
          documentId: documentId,
          uploadedAt: new Date().toISOString()
        }
      }
    });

    // Extract text from file
    const extractedText = await extractTextFromFile(file.buffer, file.mimetype);
    
    // Chunk the text
    const chunks = chunkText(extractedText);
    
    // Store document metadata and chunks
    const documentData = {
      id: documentId,
      filename: file.originalname,
      fileType: file.mimetype,
      fileSize: file.size,
      uploadDate: new Date().toISOString(),
      gcsFileName: fileName,
      chunks: chunks,
      text: extractedText
    };

    // Store document metadata in a separate file
    const metadataBlob = bucket.file(`metadata/${documentId}.json`);
    await metadataBlob.save(JSON.stringify(documentData), {
      metadata: {
        contentType: 'application/json'
      }
    });

    res.json({
      success: true,
      document: {
        id: documentId,
        filename: file.originalname,
        fileType: file.mimetype,
        fileSize: file.size,
        uploadDate: documentData.uploadDate,
        chunkCount: chunks.length
      },
      chunksAdded: chunks.length
    });

  } catch (error) {
    console.error('Upload error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get all documents
app.get('/documents', async (req, res) => {
  try {
    const [files] = await bucket.getFiles({ prefix: 'metadata/' });
    
    const documents = [];
    for (const file of files) {
      try {
        const [content] = await file.download();
        const documentData = JSON.parse(content.toString());
        documents.push({
          id: documentData.id,
          filename: documentData.filename,
          fileType: documentData.fileType,
          fileSize: documentData.fileSize,
          uploadDate: documentData.uploadDate,
          chunkCount: documentData.chunks.length
        });
      } catch (error) {
        console.error(`Error reading document ${file.name}:`, error);
      }
    }

    res.json(documents);
  } catch (error) {
    console.error('Error fetching documents:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get document content
app.get('/documents/:id', async (req, res) => {
  try {
    const documentId = req.params.id;
    const metadataBlob = bucket.file(`metadata/${documentId}.json`);
    
    const [content] = await metadataBlob.download();
    const documentData = JSON.parse(content.toString());
    
    res.json(documentData);
  } catch (error) {
    console.error('Error fetching document:', error);
    res.status(404).json({ error: 'Document not found' });
  }
});

// Delete document by ID
app.delete('/documents/:id', async (req, res) => {
  try {
    const documentId = req.params.id;
    
    // Get document metadata first
    const metadataBlob = bucket.file(`metadata/${documentId}.json`);
    const [content] = await metadataBlob.download();
    const documentData = JSON.parse(content.toString());
    
    // Delete the original file
    const fileBlob = bucket.file(documentData.gcsFileName);
    await fileBlob.delete();
    
    // Delete metadata file
    await metadataBlob.delete();
    
    res.json({ success: true, message: 'Document deleted successfully' });
  } catch (error) {
    console.error('Error deleting document:', error);
    res.status(500).json({ error: error.message });
  }
});

// Delete document by filename (alternative endpoint)
app.delete('/documents', async (req, res) => {
  try {
    const { filename } = req.body;
    
    if (!filename) {
      return res.status(400).json({ error: 'Filename is required' });
    }
    
    // Find document by filename
    const [files] = await bucket.getFiles({ prefix: 'metadata/' });
    let documentId = null;
    
    for (const file of files) {
      try {
        const [content] = await file.download();
        const documentData = JSON.parse(content.toString());
        
        if (documentData.filename === filename) {
          documentId = documentData.id;
          break;
        }
      } catch (error) {
        console.error(`Error reading document ${file.name}:`, error);
      }
    }
    
    if (!documentId) {
      return res.status(404).json({ error: 'Document not found' });
    }
    
    // Get document metadata
    const metadataBlob = bucket.file(`metadata/${documentId}.json`);
    const [content] = await metadataBlob.download();
    const documentData = JSON.parse(content.toString());
    
    // Delete the original file
    const fileBlob = bucket.file(documentData.gcsFileName);
    await fileBlob.delete();
    
    // Delete metadata file
    await metadataBlob.delete();
    
    res.json({ success: true, message: 'Document deleted successfully' });
  } catch (error) {
    console.error('Error deleting document:', error);
    res.status(500).json({ error: error.message });
  }
});

// Search documents
app.post('/search', async (req, res) => {
  try {
    const { query, maxResults = 5 } = req.body;
    
    if (!query) {
      return res.status(400).json({ error: 'Query is required' });
    }

    // Get all documents
    const [files] = await bucket.getFiles({ prefix: 'metadata/' });
    
    const results = [];
    for (const file of files) {
      try {
        const [content] = await file.download();
        const documentData = JSON.parse(content.toString());
        
        // Search through chunks
        documentData.chunks.forEach(chunk => {
          const similarity = calculateSimilarity(query.toLowerCase(), chunk.text.toLowerCase());
          if (similarity > 0.1) {
            results.push({
              chunk: chunk,
              similarity: similarity,
              document: {
                id: documentData.id,
                filename: documentData.filename
              }
            });
          }
        });
      } catch (error) {
        console.error(`Error searching document ${file.name}:`, error);
      }
    }

    // Sort by similarity and return top results
    const sortedResults = results
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, maxResults);

    res.json(sortedResults);
  } catch (error) {
    console.error('Error searching documents:', error);
    res.status(500).json({ error: error.message });
  }
});

// Simple text similarity function
function calculateSimilarity(text1, text2) {
  const words1 = text1.split(/\s+/);
  const words2 = text2.split(/\s+/);
  
  const wordSet = new Set([...words1, ...words2]);
  const vector1 = Array.from(wordSet).map(word => words1.filter(w => w === word).length);
  const vector2 = Array.from(wordSet).map(word => words2.filter(w => w === word).length);
  
  const dotProduct = vector1.reduce((sum, val, i) => sum + val * vector2[i], 0);
  const magnitude1 = Math.sqrt(vector1.reduce((sum, val) => sum + val * val, 0));
  const magnitude2 = Math.sqrt(vector2.reduce((sum, val) => sum + val * val, 0));
  
  return dotProduct / (magnitude1 * magnitude2);
}

// Crawl website
app.post('/crawl', async (req, res) => {
  try {
    const { url } = req.body;
    
    if (!url) {
      return res.status(400).json({ error: 'URL is required' });
    }

    // Simple web scraping using a proxy service
    const response = await fetch(`https://api.allorigins.win/get?url=${encodeURIComponent(url)}`);
    const data = await response.json();
    
    if (!data.contents) {
      throw new Error('Failed to fetch website content');
    }

    // Extract text from HTML
    const text = extractTextFromHTML(data.contents);
    
    // Create document record
    const documentId = uuidv4();
    const fileName = `web_${documentId}.html`;
    
    // Upload HTML content to GCS
    const blob = bucket.file(fileName);
    await blob.save(data.contents, {
      metadata: {
        contentType: 'text/html',
        metadata: {
          originalUrl: url,
          documentId: documentId,
          uploadedAt: new Date().toISOString()
        }
      }
    });

    // Chunk the text
    const chunks = chunkText(text);
    
    // Store document metadata
    const documentData = {
      id: documentId,
      filename: `Web Content - ${url}`,
      fileType: 'text/html',
      fileSize: text.length,
      uploadDate: new Date().toISOString(),
      gcsFileName: fileName,
      url: url,
      chunks: chunks,
      text: text
    };

    const metadataBlob = bucket.file(`metadata/${documentId}.json`);
    await metadataBlob.save(JSON.stringify(documentData), {
      metadata: {
        contentType: 'application/json'
      }
    });

    res.json({
      success: true,
      document: {
        id: documentId,
        filename: documentData.filename,
        fileType: documentData.fileType,
        fileSize: documentData.fileSize,
        uploadDate: documentData.uploadDate,
        chunkCount: chunks.length
      },
      chunksAdded: chunks.length
    });

  } catch (error) {
    console.error('Crawl error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Ask questions about documents (similar to search but with different response format)
app.post('/ask', async (req, res) => {
  try {
    const { question } = req.body;
    
    if (!question) {
      return res.status(400).json({ error: 'Question is required' });
    }

    // Get all documents
    const [files] = await bucket.getFiles({ prefix: 'metadata/' });
    
    const results = [];
    for (const file of files) {
      try {
        const [content] = await file.download();
        const documentData = JSON.parse(content.toString());
        
        // Search through chunks
        documentData.chunks.forEach(chunk => {
          const similarity = calculateSimilarity(question.toLowerCase(), chunk.text.toLowerCase());
          if (similarity > 0.1) {
            results.push({
              chunk: chunk,
              similarity: similarity,
              document: {
                id: documentData.id,
                filename: documentData.filename
              }
            });
          }
        });
      } catch (error) {
        console.error(`Error searching document ${file.name}:`, error);
      }
    }

    // Sort by similarity and return top results
    const sortedResults = results
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, 5);

    // Format response to match frontend expectations
    const answer = sortedResults.length > 0 
      ? `Based on the documents, here's what I found: ${sortedResults[0].chunk.text.substring(0, 200)}...`
      : "I couldn't find any relevant information in the uploaded documents.";

    const sources = sortedResults.map(result => ({
      filename: result.document.filename,
      similarity: result.similarity
    }));

    res.json({
      answer: answer,
      sources: sources,
      sales_followups: ["Would you like me to search for more specific information?", "Should I look for related documents?"],
      client_followups: ["Is there anything else you'd like to know about this topic?", "Would you like me to analyze other aspects of the documents?"]
    });

  } catch (error) {
    console.error('Error asking question:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get all chunks from all documents
app.get('/chunks', async (req, res) => {
  try {
    const [files] = await bucket.getFiles({ prefix: 'metadata/' });
    
    const allChunks = [];
    for (const file of files) {
      try {
        const [content] = await file.download();
        const documentData = JSON.parse(content.toString());
        
        documentData.chunks.forEach(chunk => {
          allChunks.push({
            ...chunk,
            documentId: documentData.id,
            filename: documentData.filename
          });
        });
      } catch (error) {
        console.error(`Error reading document ${file.name}:`, error);
      }
    }

    res.json(allChunks);
  } catch (error) {
    console.error('Error fetching chunks:', error);
    res.status(500).json({ error: error.message });
  }
});

// Refresh index (reprocess all documents)
app.post('/refresh', async (req, res) => {
  try {
    // This is a simple refresh that just returns success
    // In a more sophisticated implementation, you might reprocess all documents
    res.json({ 
      success: true, 
      message: 'Index refreshed successfully',
      documentsProcessed: 0 // Placeholder
    });
  } catch (error) {
    console.error('Error refreshing index:', error);
    res.status(500).json({ error: error.message });
  }
});

// Extract text from HTML
function extractTextFromHTML(html) {
  // Simple HTML to text conversion
  const text = html
    .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, '')
    .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '')
    .replace(/<[^>]+>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  
  return text.substring(0, 50000); // Limit text length
}

// Start server
app.listen(PORT, () => {
  console.log(`LinkedIn Bot Document Manager running on port ${PORT}`);
  console.log(`Using bucket: ${LINKEDIN_BOT_BUCKET}`);
}); 