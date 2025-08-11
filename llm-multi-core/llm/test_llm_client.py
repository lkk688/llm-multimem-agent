#!/usr/bin/env python3
"""
Test script for the universal LLM client.

This script tests the call_llm function with different backends.
To run this script, you need to have the required packages installed
and the corresponding services running (if applicable).
"""

import os
import sys
import argparse
from typing import List, Dict, Any

# Add the parent directory to the path so we can import the llm module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from .llm.llm_client import call_llm, call_vision_llm, call_llm_with_tools

# Test messages
TEST_MESSAGES = [
    {"role": "user", "content": "What is the capital of France?"}
]

# Test vision messages (with a placeholder image URL)
TEST_VISION_MESSAGES = [
    {"role": "user", "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": "https://upload.wikimedia.org/wikipedia/commons/4/4f/Eiffel_Tower_from_Champ_de_Mars%2C_Paris%2C_France.jpg"}
    ]}
]

# Test tools
TEST_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# Test messages for tool calling
TEST_TOOL_MESSAGES = [
    {"role": "user", "content": "What's the weather like in Paris?"}
]

def test_openai():
    """Test the OpenAI backend."""
    print("\n=== Testing OpenAI Backend ===\n")
    try:
        response = call_llm(
            messages=TEST_MESSAGES,
            backend="openai",
            model="gpt-3.5-turbo"
        )
        print(f"Response: {response}")
        
        # Test vision if available
        try:
            vision_response = call_vision_llm(
                messages=TEST_VISION_MESSAGES,
                backend="openai",
                model="gpt-4-vision-preview"
            )
            print(f"Vision Response: {vision_response}")
        except Exception as e:
            print(f"Vision test failed: {e}")
        
        # Test tool calling
        try:
            tool_response = call_llm_with_tools(
                messages=TEST_TOOL_MESSAGES,
                tools=TEST_TOOLS,
                backend="openai",
                model="gpt-4-turbo"
            )
            print(f"Tool Response: {tool_response}")
        except Exception as e:
            print(f"Tool calling test failed: {e}")
            
    except Exception as e:
        print(f"Test failed: {e}")

def test_litellm():
    """Test the LiteLLM backend."""
    print("\n=== Testing LiteLLM Backend ===\n")
    try:
        response = call_llm(
            messages=TEST_MESSAGES,
            backend="litellm",
            model="gpt-3.5-turbo"
        )
        print(f"Response: {response}")
        
        # Test vision if available
        try:
            vision_response = call_vision_llm(
                messages=TEST_VISION_MESSAGES,
                backend="litellm",
                model="gpt-4-vision-preview"
            )
            print(f"Vision Response: {vision_response}")
        except Exception as e:
            print(f"Vision test failed: {e}")
        
        # Test tool calling
        try:
            tool_response = call_llm_with_tools(
                messages=TEST_TOOL_MESSAGES,
                tools=TEST_TOOLS,
                backend="litellm",
                model="gpt-4-turbo"
            )
            print(f"Tool Response: {tool_response}")
        except Exception as e:
            print(f"Tool calling test failed: {e}")
            
    except Exception as e:
        print(f"Test failed: {e}")

def test_hf():
    """Test the Hugging Face backend."""
    print("\n=== Testing Hugging Face Backend ===\n")
    try:
        response = call_llm(
            messages=TEST_MESSAGES,
            backend="hf",
            model="google/flan-t5-base",  # A small model for testing
            device="cpu"
        )
        print(f"Response: {response}")
        
        # Test vision if available
        try:
            vision_response = call_vision_llm(
                messages=TEST_VISION_MESSAGES,
                backend="hf",
                model="Salesforce/blip2-opt-2.7b",  # A vision model
                device="cpu"
            )
            print(f"Vision Response: {vision_response}")
        except Exception as e:
            print(f"Vision test failed: {e}")
            
    except Exception as e:
        print(f"Test failed: {e}")

def test_llamacpp():
    """Test the llama.cpp backend."""
    print("\n=== Testing llama.cpp Backend ===\n")
    try:
        response = call_llm(
            messages=TEST_MESSAGES,
            backend="llamacpp",
            model="llama2",  # Model name doesn't matter for llama.cpp server
            api_url="http://localhost:8000/v1/completions",
            chat_api_url="http://localhost:8000/v1/chat/completions"
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"Test failed: {e}")

def test_ollama():
    """Test the Ollama backend."""
    print("\n=== Testing Ollama Backend ===\n")
    try:
        response = call_llm(
            messages=TEST_MESSAGES,
            backend="ollama",
            model="llama2"
        )
        print(f"Response: {response}")
        
        # Test vision if available
        try:
            vision_response = call_vision_llm(
                messages=TEST_VISION_MESSAGES,
                backend="ollama",
                model="llava"
            )
            print(f"Vision Response: {vision_response}")
        except Exception as e:
            print(f"Vision test failed: {e}")
            
    except Exception as e:
        print(f"Test failed: {e}")

def test_vllm():
    """Test the vLLM backend."""
    print("\n=== Testing vLLM Backend ===\n")
    try:
        response = call_llm(
            messages=TEST_MESSAGES,
            backend="vllm",
            model="meta-llama/Llama-2-7b-chat-hf"
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"Test failed: {e}")

def main():
    """Run the tests based on command line arguments."""
    parser = argparse.ArgumentParser(description="Test the universal LLM client.")
    parser.add_argument("--backend", choices=["all", "openai", "litellm", "hf", "llamacpp", "ollama", "vllm"], 
                        default="all", help="The backend to test")
    args = parser.parse_args()
    
    if args.backend == "all" or args.backend == "openai":
        test_openai()
    
    if args.backend == "all" or args.backend == "litellm":
        test_litellm()
    
    if args.backend == "all" or args.backend == "hf":
        test_hf()
    
    if args.backend == "all" or args.backend == "llamacpp":
        test_llamacpp()
    
    if args.backend == "all" or args.backend == "ollama":
        test_ollama()
    
    if args.backend == "all" or args.backend == "vllm":
        test_vllm()

if __name__ == "__main__":
    main()