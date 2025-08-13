#!/bin/bash

# Fix Dockerfile port
sed -i '' 's/EXPOSE 8000/EXPOSE 8080/' Dockerfile
sed -i '' 's/port", "8000"/port", "8080"/' Dockerfile

# Fix app.py port default
sed -i '' 's/PORT", 8000/PORT", 8080/' app.py

echo "✅ Fixed port configurations"
echo "Dockerfile:"
tail -2 Dockerfile
echo "app.py end:"
tail -3 app.py
