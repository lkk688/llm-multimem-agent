import os
import sys
import json
from PIL import Image

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the litellm utilities
from VisionLangAnnotateModels.VLM.litellm_utils import (
    process_with_litellm_text,
    process_with_litellm_vision,
    text_stream_callback,
    LITELLM_AVAILABLE
)

def main():
    """Test the litellm utilities with different providers"""
    if not LITELLM_AVAILABLE:
        print("LiteLLM is not available. Please install it with 'pip install litellm'")
        return
    
    # Get API keys from environment variables
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    
    # Dictionary to store API keys
    api_keys = {}
    if openai_api_key:
        api_keys["openai"] = openai_api_key
    if anthropic_api_key:
        api_keys["anthropic"] = anthropic_api_key
    
    # Example 1: Text processing with OpenAI
    if "openai" in api_keys:
        print("\n=== Text processing example with OpenAI ===\n")
        text_result = process_with_litellm_text(
            prompt="What are the main features of Python?",
            model="openai/gpt-3.5-turbo",
            api_keys=api_keys,
            max_tokens=100
        )
        
        if text_result["success"]:
            print(f"Response: {text_result['response']}")
        else:
            print(f"Error: {text_result.get('error', 'Unknown error')}")
    
    # Example 2: Text processing with Anthropic
    if "anthropic" in api_keys:
        print("\n=== Text processing example with Anthropic ===\n")
        text_result = process_with_litellm_text(
            prompt="Explain the concept of recursion in programming.",
            model="anthropic/claude-instant-1",
            api_keys=api_keys,
            max_tokens=100
        )
        
        if text_result["success"]:
            print(f"Response: {text_result['response']}")
        else:
            print(f"Error: {text_result.get('error', 'Unknown error')}")
    
    # Example 3: Text processing with local Ollama model
    print("\n=== Text processing example with Ollama ===\n")
    text_result = process_with_litellm_text(
        prompt="What is the capital of France?",
        model="ollama/llama2",  # Assumes llama2 is available in Ollama
        max_tokens=50
    )
    
    if text_result["success"]:
        print(f"Response: {text_result['response']}")
    else:
        print(f"Error: {text_result.get('error', 'Unknown error')}")
    
    # Example 4: Streaming example with Ollama
    print("\n=== Streaming example with Ollama ===\n")
    print("Response: ", end="")
    stream_result = process_with_litellm_text(
        prompt="List the first 5 prime numbers and explain why they are prime.",
        model="ollama/llama2",
        max_tokens=150,
        stream=True,
        stream_callback=text_stream_callback
    )
    
    print(f"\nStreaming success: {stream_result['success']}")
    
    # Example 5: Vision processing (if OpenAI API key is available)
    if "openai" in api_keys:
        # Find a sample image
        sample_image_paths = [
            "sample_image.jpg",
            "test_images/sample_image.jpg",
            "../test_images/sample_image.jpg",
        ]
        
        image_path = None
        for path in sample_image_paths:
            if os.path.exists(path):
                image_path = path
                break
        
        if image_path:
            print(f"\n=== Vision processing example using image: {image_path} ===\n")
            image = Image.open(image_path).convert("RGB")
            
            vision_result = process_with_litellm_vision(
                images=[image],
                prompts=["Describe what you see in this image."],
                model="openai/gpt-4-vision-preview",
                api_keys=api_keys,
                max_tokens=100
            )
            
            if vision_result["success"]:
                print(f"Response: {vision_result['response']}")
            else:
                print(f"Error: {vision_result.get('error', 'Unknown error')}")
        else:
            print("No sample image found. Skipping vision examples.")
    
    # Example 6: Tool use with OpenAI
    if "openai" in api_keys:
        print("\n=== Tool use example with OpenAI ===\n")
        
        # Define a calculator tool
        calculator_tool = {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "A calculator that can perform basic arithmetic operations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["add", "subtract", "multiply", "divide"],
                            "description": "The arithmetic operation to perform"
                        },
                        "a": {
                            "type": "number",
                            "description": "The first operand"
                        },
                        "b": {
                            "type": "number",
                            "description": "The second operand"
                        }
                    },
                    "required": ["operation", "a", "b"]
                }
            }
        }
        
        # Call the model with the tool
        tool_result = process_with_litellm_text(
            prompt="What is 25 multiplied by 13?",
            model="openai/gpt-4",
            api_keys=api_keys,
            tools=[calculator_tool],
            tool_choice="auto"
        )
        
        if tool_result["success"] and "tool_calls" in tool_result:
            print("Model requested to use a tool:")
            tool_calls = tool_result["tool_calls"]
            for tool_call in tool_calls:
                print(f"Tool: {tool_call.function.name}")
                print(f"Arguments: {tool_call.function.arguments}")
                
                # Parse the arguments
                args = json.loads(tool_call.function.arguments)
                
                # Execute the calculator function
                result = None
                if args["operation"] == "add":
                    result = args["a"] + args["b"]
                elif args["operation"] == "subtract":
                    result = args["a"] - args["b"]
                elif args["operation"] == "multiply":
                    result = args["a"] * args["b"]
                elif args["operation"] == "divide":
                    result = args["a"] / args["b"]
                
                print(f"Result: {result}")
                
                # Send the result back to the model
                tool_outputs = [
                    {
                        "tool_call_id": tool_call.id,
                        "output": str(result)
                    }
                ]
                
                # Get the final response
                final_result = process_with_litellm_text(
                    prompt="",  # No new prompt needed
                    model="openai/gpt-4",
                    api_keys=api_keys,
                    tools=[calculator_tool],
                    tool_choice="auto",
                    tool_call_outputs=tool_outputs
                )
                
                if final_result["success"]:
                    print(f"Final response: {final_result['response']}")
                else:
                    print(f"Error in final response: {final_result.get('error', 'Unknown error')}")
        else:
            print(f"Error or no tool calls: {tool_result.get('error', 'No tool calls made')}")
            if tool_result["success"]:
                print(f"Response: {tool_result['response']}")

if __name__ == "__main__":
    main()