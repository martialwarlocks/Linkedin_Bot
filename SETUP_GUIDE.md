# 🚀 Quick Setup Guide - Fix File Upload Issues

## 🎯 **Step 1: Start Your Backend**

### **Option A: Using the startup script (Recommended)**
```bash
# From the project root directory
./start-backend.sh
```

### **Option B: Manual setup**
```bash
# Navigate to backend directory
cd backend

# Install dependencies
npm install

# Create environment file
cp env.example .env

# Edit .env file with your Google Cloud settings
# GOOGLE_CLOUD_PROJECT_ID=your-project-id
# GOOGLE_CLOUD_KEY_FILE=./google-cloud-key.json
# LINKEDIN_BOT_BUCKET=linkedin-bot-documents

# Start the backend
npm start
```

## 🎯 **Step 2: Test Backend**

```bash
# In a new terminal, run the test script
cd backend
node test-backend.js
```

You should see:
```
🧪 Testing Backend Endpoints...

1️⃣ Testing Health Check...
✅ Health Check: { status: 'OK', service: 'LinkedIn Bot Document Manager' }

2️⃣ Testing Root Endpoint...
✅ Root Endpoint: { status: 'online', service: 'LinkedIn Bot Document Manager', ... }

3️⃣ Testing Get Documents...
✅ Documents: []

4️⃣ Testing Get Chunks...
✅ Chunks: []

🎉 All tests passed! Backend is working correctly.
```

## 🎯 **Step 3: Start Your Frontend**

```bash
# In another terminal, start the frontend
npm start
```

## 🎯 **Step 4: Test File Upload**

1. Open your app in the browser
2. Go to the "Documents" tab
3. Click "Choose File" and select a PDF
4. Click "Upload"
5. Check the browser console for detailed logs

## 🔧 **Troubleshooting**

### **If backend won't start:**
- Check if port 8000 is available
- Verify Google Cloud credentials
- Check `.env` file configuration

### **If upload fails:**
- Check browser console for error messages
- Verify backend is running on `http://localhost:8000`
- Check file size (max 10MB)
- Check file type (PDF, DOCX, etc.)

### **If files don't appear:**
- Check browser console logs
- Verify Google Cloud Storage bucket exists
- Check backend logs for errors

## 📝 **Common Issues & Solutions**

### **Issue: "Backend connection failed"**
**Solution:** Make sure backend is running on port 8000

### **Issue: "Google Cloud credentials not found"**
**Solution:** Download service account key and place in `backend/google-cloud-key.json`

### **Issue: "Bucket not found"**
**Solution:** Create bucket: `gsutil mb gs://linkedin-bot-documents`

### **Issue: "Upload failed: 413"**
**Solution:** File too large, reduce file size (max 10MB)

### **Issue: "Invalid file type"**
**Solution:** Only PDF, DOCX, XLSX, PPTX, TXT, HTML files are supported 