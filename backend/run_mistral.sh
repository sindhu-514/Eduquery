#!/bin/bash

echo "🚀 Setting up Optimized Mistral EduQuery..."

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed. Please install Ollama first:"
    echo "   Visit: https://ollama.ai/download"
    exit 1
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "🔄 Starting Ollama server..."
    ollama serve &
    sleep 3
fi

# Check if Mistral model is available
echo "🔍 Checking for Mistral model..."
if ! ollama list | grep -q "mistral:7b"; then
    echo "📥 Pulling Mistral model (this may take a few minutes)..."
    ollama pull mistral:7b
else
    echo "✅ Mistral model already available"
fi

# Check if bge-m3 embedding model is available
echo "🔍 Checking for bge-m3 embedding model..."
if ! ollama list | grep -q "bge-m3"; then
    echo "📥 Pulling bge-m3 embedding model..."
    ollama pull bge-m3:latest
else
    echo "✅ bge-m3 embedding model already available"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p ./books
mkdir -p ./data
mkdir -p ./cache

# Check Python dependencies
echo "🔍 Checking Python dependencies..."
python3 -c "import fastapi, uvicorn, chromadb, langchain, requests, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing Python dependencies..."
    pip install fastapi uvicorn chromadb langchain langchain-community requests numpy python-dotenv
fi

echo "✅ Setup complete!"
echo ""
echo "🌐 Starting Optimized Mistral EduQuery on port 8007..."
echo "📖 API Documentation: http://localhost:8007/docs"
echo "🔗 Health Check: http://localhost:8007/"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the application
python3 mistral.py 