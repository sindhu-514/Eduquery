#!/bin/bash

# Set environment variables to reduce Ollama logging
export OLLAMA_DEBUG=false
export OLLAMA_LOG_LEVEL=warn

# Start Ollama with reduced logging
ollama serve 2>&1 | grep -E "(ERROR|WARN|FATAL)" || true 