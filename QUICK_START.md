# Quick Start Guide

## 🚀 Start the Application

### Option 1: Using the Start Script (Recommended)

**On macOS/Linux:**
```bash
./start-app.sh
```

**On Windows:**
```cmd
start-app.bat
```

### Option 2: Manual Start

**Terminal 1 - Start Backend:**
```bash
python3 local-fastapi-server.py
```

**Terminal 2 - Start Frontend:**
```bash
npm start
```

## 📱 Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 🛠️ Script Commands

### Using start-app.sh (macOS/Linux):

```bash
# Start both servers
./start-app.sh start

# Check server status
./start-app.sh status

# Stop both servers
./start-app.sh stop

# Start only backend
./start-app.sh backend

# Start only frontend
./start-app.sh frontend
```

## 🔧 Troubleshooting

### If ports are already in use:
```bash
# Check what's using the ports
lsof -i :3000
lsof -i :8000

# Kill processes if needed
pkill -f "python3 local-fastapi-server.py"
pkill -f "react-scripts start"
```

### If dependencies are missing:
```bash
# Install Python dependencies
pip install -r local-requirements.txt

# Install Node.js dependencies
npm install
```

## 📋 Prerequisites

- **Python 3.9+**
- **Node.js 18+**
- **npm** (comes with Node.js)

## 🎯 What's Running

1. **FastAPI Backend** (`local-fastapi-server.py`)
   - Document processing
   - Web crawling
   - AI content generation
   - Research management
   - Vector database

2. **React Frontend** (`npm start`)
   - User interface
   - Content creation
   - Research management
   - Document upload
   - Chat interface

## 🔄 Restart the Application

If you need to restart:

1. **Stop current servers:**
   ```bash
   ./start-app.sh stop
   ```

2. **Start again:**
   ```bash
   ./start-app.sh start
   ```

## 📊 Monitor Logs

The backend will show logs in the terminal where you started it. You can see:
- API requests
- Content generation
- Web crawling
- Database operations
- Error messages

## 🆘 Need Help?

- Check the full setup guide: `SETUP_GUIDE.md`
- Review API documentation: http://localhost:8000/docs
- Check the logs in the terminal 