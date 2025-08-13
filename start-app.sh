#!/bin/bash

# LinkedIn Content Creator - Quick Start Script
echo "🚀 Starting LinkedIn Content Creator..."

# Function to check if a port is in use
check_port() {
    lsof -i :$1 >/dev/null 2>&1
}

# Function to start backend
start_backend() {
    echo "🐍 Starting FastAPI Backend..."
    if check_port 8000; then
        echo "✅ Backend already running on port 8000"
    else
        python3 local-fastapi-server.py &
        BACKEND_PID=$!
        echo "✅ Backend started with PID: $BACKEND_PID"
        
        # Wait for backend to be ready
        echo "⏳ Waiting for backend to be ready..."
        for i in {1..30}; do
            if curl -s http://localhost:8000/ > /dev/null 2>&1; then
                echo "✅ Backend is ready!"
                break
            fi
            sleep 1
        done
    fi
}

# Function to start frontend
start_frontend() {
    echo "⚛️  Starting React Frontend..."
    if check_port 3000; then
        echo "✅ Frontend already running on port 3000"
    else
        npm start &
        FRONTEND_PID=$!
        echo "✅ Frontend started with PID: $FRONTEND_PID"
    fi
}

# Function to stop servers
stop_servers() {
    echo "🛑 Stopping servers..."
    pkill -f "python3 local-fastapi-server.py"
    pkill -f "react-scripts start"
    echo "✅ Servers stopped"
}

# Function to show status
show_status() {
    echo "📊 Server Status:"
    if check_port 8000; then
        echo "✅ Backend: Running on http://localhost:8000"
    else
        echo "❌ Backend: Not running"
    fi
    
    if check_port 3000; then
        echo "✅ Frontend: Running on http://localhost:3000"
    else
        echo "❌ Frontend: Not running"
    fi
}

# Main menu
case "${1:-start}" in
    "start")
        start_backend
        start_frontend
        echo ""
        echo "🎉 LinkedIn Content Creator is starting!"
        echo "📱 Frontend: http://localhost:3000"
        echo "🔧 Backend API: http://localhost:8000"
        echo "📚 API Documentation: http://localhost:8000/docs"
        echo ""
        echo "Press Ctrl+C to stop both servers"
        echo "Use './start-app.sh stop' to stop servers"
        echo "Use './start-app.sh status' to check status"
        
        # Wait for user to stop
        wait
        ;;
    "stop")
        stop_servers
        ;;
    "status")
        show_status
        ;;
    "backend")
        start_backend
        ;;
    "frontend")
        start_frontend
        ;;
    *)
        echo "Usage: $0 {start|stop|status|backend|frontend}"
        echo "  start   - Start both backend and frontend (default)"
        echo "  stop    - Stop both servers"
        echo "  status  - Show server status"
        echo "  backend - Start only backend"
        echo "  frontend- Start only frontend"
        exit 1
        ;;
esac 