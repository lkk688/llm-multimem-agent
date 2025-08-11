#!/usr/bin/env python3
"""
Ollama Integration Examples with gpt-oss models

This script demonstrates various ways to use Ollama with gpt-oss models:
1. Basic chat completions with the OpenAI SDK
2. Function calling
3. Using the Responses API
4. Integration with OpenAI's Agents SDK

Based on: https://cookbook.openai.com/articles/gpt-oss/run-locally-ollama
"""

import asyncio
import json
import os
import sys
from typing import Dict, List, Optional, Any

# For basic OpenAI SDK usage
from openai import OpenAI

# For OpenAI Responses API example (you'll need the latest OpenAI SDK)
# pip install openai>=1.2.3
try:
    # Check if OpenAI SDK is available with Responses API support
    import openai
    # Initialize with a dummy key to avoid errors when checking for attributes
    # This is only used for checking if the API is available, not for actual calls
    openai.api_key = "dummy_key_for_attribute_check"
    RESPONSES_API_AVAILABLE = hasattr(openai, 'responses')
except ImportError:
    RESPONSES_API_AVAILABLE = False
    print("OpenAI SDK not available or outdated. Install with: pip install openai>=1.2.3")

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434/v1"
DUMMY_API_KEY = "ollama"  # Ollama doesn't need a real API key
MODEL_NAME = "gpt-oss:20b"  # Use "gpt-oss:120b" for the larger model


def setup_client():
    """Set up and return an OpenAI client configured for Ollama"""
    return OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key=DUMMY_API_KEY
    )


def basic_chat_example():
    """Basic example of using the Chat Completions API with Ollama"""
    print("\n=== Basic Chat Example ===")
    client = setup_client()
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain what MXFP4 quantization is in simple terms."}
        ]
    )
    
    print(f"Response: {response.choices[0].message.content}")


def function_calling_example():
    """Example of using function calling with Ollama"""
    print("\n=== Function Calling Example ===")
    client = setup_client()
    
    # Define the tools/functions the model can use
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather in a given city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                },
            },
        }
    ]
    
    # First message from user
    messages = [{"role": "user", "content": "What's the weather in Tokyo right now?"}]
    
    # First API call - model should request to call the function
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    assistant_message = response.choices[0].message
    messages.append(assistant_message)
    
    # Extract reasoning if available in model_extra
    reasoning = ""
    if hasattr(assistant_message, 'model_extra') and 'reasoning' in assistant_message.model_extra:
        reasoning = assistant_message.model_extra['reasoning']
        print(f"Reasoning: {reasoning}")
    
    print(f"Assistant: {assistant_message.content if assistant_message.content else 'Thinking...'}")
    
    
    # Check if the model wants to call a function
    if assistant_message.tool_calls:
        # Process each tool call
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            # Get the function result by calling a real weather API
            def get_weather_data(city):
                try:
                    # Using OpenWeatherMap API (you would need to set your API key in a real app)
                    # For demo purposes, we'll return simulated data based on the city
                    weather_data = {
                        "Tokyo": {"condition": "sunny", "temp": 25},
                        "New York": {"condition": "cloudy", "temp": 18},
                        "London": {"condition": "rainy", "temp": 15},
                        "Sydney": {"condition": "clear", "temp": 28},
                    }
                    
                    # Get data for the requested city or provide a default response
                    city_data = weather_data.get(city, {"condition": "unknown", "temp": 20})
                    return f"It's {city_data['condition']} and {city_data['temp']}°C in {city} today."
                except Exception as e:
                    return f"Error getting weather data: {str(e)}"
            
            # Call the weather function with the city argument
            function_response = get_weather_data(function_args.get('city', 'unknown location'))
            print(f"Called function: {function_name}({function_args})")
            print(f"Function returned: {function_response}")
            
            # Append the function response to messages
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            })
        
        # Second API call with function results
        second_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
        )
        
        # Extract reasoning from second response if available
        final_message = second_response.choices[0].message
        if hasattr(final_message, 'model_extra') and 'reasoning' in final_message.model_extra:
            reasoning = final_message.model_extra['reasoning']
            print(f"Final reasoning: {reasoning}")
        
        print(f"Final response: {final_message.content}")


def main():
    """Run all examples"""
    print(f"Using model: {MODEL_NAME}")
    print("Make sure Ollama is running with the model pulled:")
    print(f"  ollama pull {MODEL_NAME}")
    print("\nPress Enter to continue or Ctrl+C to exit...")
    input()
    
    try:
        # Run the examples
        basic_chat_example()
        function_calling_example()
            
        print("\nAll examples completed!")
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure Ollama is running and the model is pulled.")
        return 1
    
    return 0


# Note: The assistants_api_example function has been removed as it was deprecated
# and replaced with the responses_api_example_with_function function.


if __name__ == "__main__":
    sys.exit(main())