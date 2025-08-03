#!/usr/bin/env python3
"""
Test script to verify Mistral model availability and functionality
"""

import requests
import json
import logging

logging.basicConfig(level=logging.INFO)

def test_ollama_connection():
    """Test if Ollama is running and accessible"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        model_names = [model.get("name", "") for model in models]
        
        print(f"✅ Ollama is running")
        print(f"📋 Available models: {model_names}")
        
        return model_names
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        return []

def test_mistral_model():
    """Test if Mistral model is working"""
    try:
        payload = {
            "model": "mistral:7b",
            "prompt": "Hello, this is a test. Please respond with 'Test successful'",
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 50
            }
        }
        
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        print(f"✅ Mistral model is working")
        print(f"📝 Response: {result.get('response', 'No response')}")
        
        return True
    except Exception as e:
        print(f"❌ Mistral model test failed: {e}")
        return False

def main():
    print("🔍 Testing Mistral Setup...")
    print("=" * 50)
    
    # Test Ollama connection
    models = test_ollama_connection()
    
    if not models:
        print("\n❌ Ollama is not running or not accessible")
        print("💡 Please run: ollama serve")
        return
    
    # Check if Mistral is available
    if "mistral:7b" not in models:
        print(f"\n❌ Mistral model not found in available models")
        print("💡 Please run: ollama pull mistral:7b")
        return
    
    print(f"\n✅ Mistral model is available")
    
    # Test Mistral functionality
    if test_mistral_model():
        print("\n🎉 Everything is working! You can now run the main application.")
        print("💡 Run: python mistral.py")
    else:
        print("\n❌ Mistral model is not working properly")
        print("💡 Try restarting Ollama: pkill ollama && ollama serve")

if __name__ == "__main__":
    main() 