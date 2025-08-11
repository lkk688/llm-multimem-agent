import json
import re
import requests
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
from PIL import Image
import time
import io
import base64

# Check if OpenAI package is available
try:
    import openai
    OPENAI_AVAILABLE = True
    
    # Check if Responses API is available
    try:
        from openai import OpenAI
        client = OpenAI(api_key="test")
        # Just check if the attribute exists
        if hasattr(client, 'responses'):
            RESPONSES_API_AVAILABLE = True
        else:
            RESPONSES_API_AVAILABLE = False
    except (ImportError, AttributeError):
        RESPONSES_API_AVAILABLE = False
except ImportError:
    OPENAI_AVAILABLE = False
    RESPONSES_API_AVAILABLE = False

# Default OpenAI API endpoint
DEFAULT_OPENAI_API_URL = "https://api.openai.com/v1"

class OpenAIClient:
    """
    Client for interacting with OpenAI API for text-only, vision LLMs, and the Responses API.
    
    This class provides interfaces for:
    1. Text-only LLMs (process_text)
    2. Vision LLMs (process_vision)
    3. Responses API (process_responses)
    
    It handles API communication, error handling, response parsing, and streaming.
    """
    
    def __init__(self, api_key: str, api_url: str = DEFAULT_OPENAI_API_URL, timeout: int = 60, stream_callback: Optional[Callable[[str], None]] = None):
        """
        Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key
            api_url: URL of the OpenAI API endpoint
            timeout: Timeout in seconds for API requests
            stream_callback: Optional callback function for streaming responses. 
                             The callback should accept a string chunk and return None.
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package is not available. Please install it with 'pip install openai'.")
        
        self.api_url = api_url
        self.timeout = timeout
        self.stream_callback = stream_callback
        self.client = openai.OpenAI(api_key=api_key)
    
    def process_text(self, 
                     prompt: str, 
                     model: str = "gpt-4o", 
                     system_prompt: Optional[str] = None,
                     max_tokens: int = 1000,
                     stream: bool = False) -> Dict[str, Any]:
        """
        Process text using an OpenAI text-only LLM.
        
        Args:
            prompt: The text prompt to send to the model
            model: Name of the OpenAI model to use (e.g., "gpt-4o", "gpt-3.5-turbo")
            system_prompt: Optional system prompt to set context
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response (requires stream_callback to be set)
            
        Returns:
            Dictionary containing the processed result or error information
        """
        # Check if streaming is requested but no callback is set
        if stream and self.stream_callback is None:
            return {
                "success": False,
                "error": "Stream callback must be set to use streaming"
            }
            
        try:
            # Prepare the messages
            messages = []
            
            # Add system prompt if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add user prompt
            messages.append({"role": "user", "content": prompt})
            
            # For streaming responses
            if stream:
                return self._handle_streaming_request(model, messages, max_tokens)
            
            # For non-streaming responses
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                stream=False
            )
            
            # Extract response text
            response_text = response.choices[0].message.content
            
            return {
                "success": True,
                "response": response_text,
                "raw_response": response
            }
                
        except openai.APITimeoutError:
            # Handle timeout
            error_msg = f"Timeout while calling OpenAI API ({self.timeout}s)"
            return {
                "success": False,
                "error": error_msg
            }
            
        except Exception as e:
            # Handle other errors
            error_msg = f"Error calling OpenAI: {str(e)}"
            return {
                "success": False,
                "error": error_msg
            }
    
    def _handle_streaming_request(self, model: str, messages: List[Dict[str, Any]], max_tokens: int) -> Dict[str, Any]:
        """
        Handle streaming requests to OpenAI API.
        
        Args:
            model: The model to use
            messages: The messages to send
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Dictionary containing the processed result or error information
        """
        try:
            # Make the streaming request
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                stream=True
            )
            
            # Process the streaming response
            full_response = ""
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    # Extract the text chunk
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content is not None:
                        text_chunk = delta.content
                        
                        # Call the callback with the chunk
                        if self.stream_callback and text_chunk:
                            self.stream_callback(text_chunk)
                        
                        # Append to full response
                        full_response += text_chunk
            
            # Return the full response
            return {
                "success": True,
                "response": full_response,
                "raw_response": {"response": full_response}  # Simplified raw response
            }
                
        except openai.APITimeoutError:
            # Handle timeout
            error_msg = f"Timeout while streaming from OpenAI API ({self.timeout}s)"
            return {
                "success": False,
                "error": error_msg
            }
            
        except Exception as e:
            # Handle other errors
            error_msg = f"Error streaming from OpenAI: {str(e)}"
            return {
                "success": False,
                "error": error_msg
            }
    
    def process_vision(self,
                      images: List[Image.Image],
                      prompts: List[str],
                      model: str = "gpt-4-vision-preview",
                      combined_prompt: Optional[str] = None,
                      max_tokens: int = 1000,
                      stream: bool = False) -> Dict[str, Any]:
        """
        Process images using an OpenAI vision-capable LLM (e.g., GPT-4V).
        
        This method supports both single and multiple images with corresponding prompts.
        For multiple images, it can either process them individually or as a batch with a
        combined prompt.
        
        Args:
            images: List of PIL Image objects
            prompts: List of text prompts corresponding to each image
            model: Name of the OpenAI vision model to use (e.g., "gpt-4-vision-preview")
            combined_prompt: Optional combined prompt for batch processing
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response (requires stream_callback to be set)
            
        Returns:
            Dictionary containing the processed results or error information
        """
        # Validate inputs
        if not images or len(images) != len(prompts):
            return {
                "success": False,
                "error": "Number of images must match number of prompts"
            }
        
        # Check if streaming is requested but no callback is set
        if stream and self.stream_callback is None:
            return {
                "success": False,
                "error": "Stream callback must be set to use streaming"
            }
        
        try:
            # For single image processing
            if len(images) == 1:
                return self._process_single_vision(images[0], prompts[0], model, max_tokens, stream)
            
            # For multiple images
            if combined_prompt:
                # Process as batch with combined prompt
                return self._process_batch_vision(images, combined_prompt, model, max_tokens, stream)
            else:
                # Process each image individually and combine results
                results = []
                for i, (image, prompt) in enumerate(zip(images, prompts)):
                    result = self._process_single_vision(image, prompt, model, max_tokens, stream)
                    results.append(result)
                
                # Check if all were successful
                all_successful = all(r.get("success", False) for r in results)
                
                if all_successful:
                    return {
                        "success": True,
                        "responses": [r.get("response", "") for r in results],
                        "raw_responses": [r.get("raw_response", {}) for r in results]
                    }
                else:
                    # Return the first error encountered
                    for result in results:
                        if not result.get("success", False):
                            return result
                    
                    # Fallback error if none found but success is still False
                    return {
                        "success": False,
                        "error": "Unknown error in batch processing"
                    }
                
        except Exception as e:
            # Handle other errors
            error_msg = f"Error in vision processing: {str(e)}"
            return {
                "success": False,
                "error": error_msg
            }
    
    def _process_single_vision(self, image: Image.Image, prompt: str, model: str, max_tokens: int, stream: bool = False) -> Dict[str, Any]:
        """
        Process a single image with an OpenAI vision-capable model.
        
        Args:
            image: PIL Image object
            prompt: Text prompt for the image
            model: Name of the OpenAI vision model
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response (requires stream_callback to be set)
            
        Returns:
            Dictionary containing the processed result or error information
        """
        try:
            # Convert image to base64
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="PNG")
            img_byte_arr = img_byte_arr.getvalue()
            img_base64 = base64.b64encode(img_byte_arr).decode("utf-8")
            
            # Prepare the messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                    ]
                }
            ]
            
            # For streaming responses
            if stream:
                return self._handle_streaming_request(model, messages, max_tokens)
            
            # For non-streaming responses
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                stream=False
            )
            
            # Extract response text
            response_text = response.choices[0].message.content
            
            return {
                "success": True,
                "response": response_text,
                "raw_response": response
            }
                
        except openai.APITimeoutError:
            # Handle timeout
            error_msg = f"Timeout while calling OpenAI API ({self.timeout}s)"
            return {
                "success": False,
                "error": error_msg
            }
            
        except Exception as e:
            # Handle other errors
            error_msg = f"Error calling OpenAI vision: {str(e)}"
            return {
                "success": False,
                "error": error_msg
            }
    
    def _process_batch_vision(self, images: List[Image.Image], combined_prompt: str, model: str, max_tokens: int, stream: bool = False) -> Dict[str, Any]:
        """
        Process multiple images as a batch with a combined prompt.
        
        Args:
            images: List of PIL Image objects
            combined_prompt: Combined prompt for all images
            model: Name of the OpenAI vision model
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response (requires stream_callback to be set)
            
        Returns:
            Dictionary containing the processed result or error information
        """
        try:
            # Convert all images to base64
            content = [{"type": "text", "text": combined_prompt}]
            
            for image in images:
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format="PNG")
                img_byte_arr = img_byte_arr.getvalue()
                img_base64 = base64.b64encode(img_byte_arr).decode("utf-8")
                
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}})
            
            # Prepare the messages
            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]
            
            # For streaming responses
            if stream:
                return self._handle_streaming_request(model, messages, max_tokens)
            
            # For non-streaming responses
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                stream=False
            )
            
            # Extract response text
            response_text = response.choices[0].message.content
            
            return {
                "success": True,
                "response": response_text,
                "raw_response": response
            }
                
        except openai.APITimeoutError:
            # Handle timeout
            error_msg = f"Timeout while calling OpenAI API ({self.timeout}s)"
            return {
                "success": False,
                "error": error_msg
            }
            
        except Exception as e:
            # Handle other errors
            error_msg = f"Error calling OpenAI batch vision: {str(e)}"
            return {
                "success": False,
                "error": error_msg
            }
            
    def process_responses(self,
                         input_text: str,
                         model: str = "gpt-4o",
                         instructions: Optional[str] = None,
                         tools: Optional[List[Dict[str, Any]]] = None,
                         tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
                         max_tokens: Optional[int] = None,
                         temperature: float = 1.0,
                         top_p: float = 1.0,
                         stream: bool = False,
                         previous_response_id: Optional[str] = None,
                         tool_call_outputs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Process text using OpenAI's Responses API.
        
        The Responses API provides a simplified interface for chat completions with additional
        features like function calling, tool use, and structured outputs.
        
        Args:
            input_text: The text input to send to the model
            model: Name of the OpenAI model to use (e.g., "gpt-4o")
            instructions: Optional system instructions to set context
            tools: Optional list of tools the model can use
            tool_choice: Optional specification for which tool to use
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter (0-1)
            stream: Whether to stream the response (requires stream_callback to be set)
            previous_response_id: Optional ID of a previous response for conversation continuity
            tool_call_outputs: Optional list of outputs from previously called tools
            
        Returns:
            Dictionary containing the processed result or error information
        """
        if not RESPONSES_API_AVAILABLE:
            return {
                "success": False,
                "error": "Responses API is not available. Please update your OpenAI Python package."
            }
            
        # Check if streaming is requested but no callback is set
        if stream and self.stream_callback is None:
            return {
                "success": False,
                "error": "Stream callback must be set to use streaming"
            }
            
        try:
            # Prepare the request parameters
            params = {
                "model": model,
                "input": input_text,
            }
            
            # Add optional parameters if provided
            if instructions:
                params["instructions"] = instructions
                
            if tools:
                params["tools"] = tools
                
            if tool_choice:
                params["tool_choice"] = tool_choice
                
            if max_tokens:
                params["max_output_tokens"] = max_tokens
                
            if temperature != 1.0:
                params["temperature"] = temperature
                
            if top_p != 1.0:
                params["top_p"] = top_p
                
            if previous_response_id:
                params["previous_response_id"] = previous_response_id
                
            if tool_call_outputs:
                params["tool_call_outputs"] = tool_call_outputs
            
            # For streaming responses
            if stream:
                # TODO: Implement streaming for Responses API
                # This will require a different approach than the chat completions API
                return {
                    "success": False,
                    "error": "Streaming is not yet implemented for the Responses API"
                }
            
            # For non-streaming responses
            response = self.client.responses.create(**params)
            
            # Extract response text
            response_text = response.output_text if hasattr(response, 'output_text') else None
            
            # Check for tool calls
            tool_calls = response.tool_calls if hasattr(response, 'tool_calls') else None
            
            return {
                "success": True,
                "response": response_text,
                "tool_calls": tool_calls,
                "response_id": response.id if hasattr(response, 'id') else None,
                "raw_response": response
            }
                
        except openai.APITimeoutError:
            # Handle timeout
            error_msg = f"Timeout while calling OpenAI Responses API ({self.timeout}s)"
            return {
                "success": False,
                "error": error_msg
            }
            
        except Exception as e:
            # Handle other errors
            error_msg = f"Error calling OpenAI Responses API: {str(e)}"
            return {
                "success": False,
                "error": error_msg
            }


def process_with_openai_text(prompt: str, 
                           api_key: str,
                           model: str = "gpt-4o", 
                           system_prompt: Optional[str] = None,
                           max_tokens: int = 1000,
                           api_url: str = DEFAULT_OPENAI_API_URL,
                           timeout: int = 60,
                           stream: bool = False,
                           stream_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """
    Process text using an OpenAI text-only LLM.
    
    Args:
        prompt: The text prompt to send to the model
        api_key: OpenAI API key
        model: Name of the OpenAI model to use (e.g., "gpt-4o", "gpt-3.5-turbo")
        system_prompt: Optional system prompt to set context
        max_tokens: Maximum number of tokens to generate
        api_url: URL of the OpenAI API endpoint
        timeout: Timeout in seconds for API requests
        stream: Whether to stream the response
        stream_callback: Callback function for streaming responses
        
    Returns:
        Dictionary containing the processed result or error information
    """
    client = OpenAIClient(api_key=api_key, api_url=api_url, timeout=timeout, stream_callback=stream_callback)
    return client.process_text(prompt, model, system_prompt, max_tokens, stream=stream)


def process_with_openai_vision(images: List[Image.Image], 
                             prompts: List[str], 
                             api_key: str,
                             model: str = "gpt-4-vision-preview",
                             combined_prompt: Optional[str] = None,
                             max_tokens: int = 1000,
                             api_url: str = DEFAULT_OPENAI_API_URL,
                             timeout: int = 60,
                             stream: bool = False,
                             stream_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """
    Process images using an OpenAI vision-capable LLM.
    
    Args:
        images: List of PIL Image objects
        prompts: List of text prompts corresponding to each image
        api_key: OpenAI API key
        model: Name of the OpenAI vision model to use (e.g., "gpt-4-vision-preview")
        combined_prompt: Optional combined prompt for batch processing
        max_tokens: Maximum number of tokens to generate
        api_url: URL of the OpenAI API endpoint
        timeout: Timeout in seconds for API requests
        stream: Whether to stream the response
        stream_callback: Callback function for streaming responses
        
    Returns:
        Dictionary containing the processed results or error information
    """
    client = OpenAIClient(api_key=api_key, api_url=api_url, timeout=timeout, stream_callback=stream_callback)
    return client.process_vision(images, prompts, model, combined_prompt, max_tokens, stream=stream)


def process_with_openai_responses(input_text: str,
                                api_key: str,
                                model: str = "gpt-4o", 
                                instructions: Optional[str] = None,
                                tools: Optional[List[Dict[str, Any]]] = None, 
                                tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
                                max_tokens: Optional[int] = None, 
                                temperature: float = 1.0, 
                                top_p: float = 1.0,
                                api_url: str = DEFAULT_OPENAI_API_URL,
                                timeout: int = 60,
                                stream: bool = False,
                                stream_callback: Optional[Callable[[str], None]] = None,
                                previous_response_id: Optional[str] = None, 
                                tool_call_outputs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Process text using OpenAI's Responses API.
    
    Args:
        input_text: The text input to send to the model
        api_key: OpenAI API key
        model: Name of the OpenAI model to use (e.g., "gpt-4o")
        instructions: Optional system instructions to set context
        tools: Optional list of tools the model can use
        tool_choice: Optional specification for which tool to use
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0-2)
        top_p: Nucleus sampling parameter (0-1)
        api_url: URL of the OpenAI API endpoint
        timeout: Timeout in seconds for API requests
        stream: Whether to stream the response
        stream_callback: Callback function for streaming responses
        previous_response_id: Optional ID of a previous response for conversation continuity
        tool_call_outputs: Optional list of outputs from previously called tools
        
    Returns:
        Dictionary containing the processed result or error information
    """
    client = OpenAIClient(api_key=api_key, api_url=api_url, timeout=timeout, stream_callback=stream_callback)
    return client.process_responses(input_text, model, instructions, tools, tool_choice, 
                                  max_tokens, temperature, top_p, stream, 
                                  previous_response_id, tool_call_outputs)


class OpenAIVLMInterface:
    """
    Interface for using OpenAI's vision models with the VisionLangAnnotate framework.
    
    This class provides a standardized interface for using OpenAI's vision models
    with the RegionCaptioner and VLMClassifier classes.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4-vision-preview", api_url: str = DEFAULT_OPENAI_API_URL):
        """
        Initialize the OpenAI VLM interface.
        
        Args:
            api_key: OpenAI API key
            model: Name of the OpenAI vision model to use
            api_url: URL of the OpenAI API endpoint
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package is not available. Please install it with 'pip install openai'.")
        
        self.api_key = api_key
        self.model = model
        self.api_url = api_url
        self.client = openai.OpenAI(api_key=api_key, base_url=api_url)
    
    def process_image(self, image: Image.Image, prompt: str, max_tokens: int = 100) -> str:
        """
        Process a single image with a prompt.
        
        Args:
            image: PIL Image object
            prompt: Text prompt for the image
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text response
        """
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(img_byte_arr).decode('utf-8')}"}}
                        ]
                    }
                ],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    def process_images(self, images: List[Image.Image], prompts: List[str], max_tokens: int = 100) -> List[str]:
        """
        Process multiple images with corresponding prompts.
        
        Args:
            images: List of PIL Image objects
            prompts: List of text prompts corresponding to each image
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            List of generated text responses
        """
        results = []
        for image, prompt in zip(images, prompts):
            results.append(self.process_image(image, prompt, max_tokens))
        return results


if __name__ == "__main__":
    # Example usage
    import os
    import sys
    import json
    from PIL import Image
    
    # Get API key from environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        exit(1)
    
    # Example 1: Text processing
    def text_stream_callback(chunk):
        print(chunk, end="", flush=True)
    
    print("\nText processing example:")
    text_result = process_with_openai_text(
        prompt="Explain quantum computing in simple terms.",
        api_key=api_key,
        model="gpt-3.5-turbo",  # Using a cheaper model for testing
        max_tokens=100,
        stream=False
    )
    
    if text_result["success"]:
        print(f"Response: {text_result['response']}")
    else:
        print(f"Error: {text_result.get('error', 'Unknown error')}")
    
    # Example 2: Text streaming
    print("\nText streaming example:")
    stream_text_result = process_with_openai_text(
        prompt="List the top 5 programming languages in 2024.",
        api_key=api_key,
        model="gpt-3.5-turbo",  # Using a cheaper model for testing
        max_tokens=100,
        stream=True,
        stream_callback=text_stream_callback
    )
    
    print(f"\nStreaming success: {stream_text_result['success']}")
    
    # Example 3: Vision processing
    # Try to find a sample image in the project
    sample_image_paths = [
        "/home/lkk/Developer/VisionLangAnnotate/VisionLangAnnotateModels/sampledata/bus.jpg",
        "/home/lkk/Developer/VisionLangAnnotate/VisionLangAnnotateModels/sampledata/064224_000100original.jpg",
        "/home/lkk/Developer/VisionLangAnnotate/static/uploads/test.jpg"
    ]
    
    image_path = None
    for path in sample_image_paths:
        if os.path.exists(path):
            image_path = path
            break
    
    if image_path:
        print(f"\nVision processing example using image: {image_path}")
        image = Image.open(image_path).convert("RGB")
        
        vision_result = process_with_openai_vision(
            images=[image],
            prompts=["Describe what you see in this image."],
            api_key=api_key,
            model="gpt-4-vision-preview",
            max_tokens=100,
            stream=False
        )
        
        if vision_result["success"]:
            print(f"Response: {vision_result['response']}")
        else:
            print(f"Error: {vision_result.get('error', 'Unknown error')}")
        
        # Example 4: Vision streaming
        print("\nVision streaming example:")
        stream_vision_result = process_with_openai_vision(
            images=[image],
            prompts=["What objects can you identify in this image?"],
            api_key=api_key,
            model="gpt-4-vision-preview",
            max_tokens=100,
            stream=True,
            stream_callback=text_stream_callback
        )
        
        print(f"\nStreaming success: {stream_vision_result['success']}")
    else:
        print("No sample image found. Skipping vision examples.")
        
    # Example 5: Responses API
    if RESPONSES_API_AVAILABLE:
        print("\nResponses API example:")
        responses_result = process_with_openai_responses(
            input_text="What are the main features of Python?",
            api_key=api_key,
            model="gpt-4o",
            instructions="You are a helpful programming assistant.",
            max_tokens=150
        )
        
        if responses_result["success"]:
            print(f"Response: {responses_result['response']}")
        else:
            print(f"Error: {responses_result.get('error', 'Unknown error')}")
            
        # Example 6: Responses API with tools
        print("\nResponses API with tools example:")
        
        # Define a calculator tool
        calculator_tool = {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Perform a mathematical calculation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The mathematical expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }
        
        # First call to get tool calls
        tool_result = process_with_openai_responses(
            input_text="Calculate 24 * 15 - 7",
            api_key=api_key,
            model="gpt-4o",
            tools=[calculator_tool],
            tool_choice="auto"
        )
        
        if tool_result["success"] and tool_result.get("tool_calls"):
            print(f"Tool called: {tool_result['tool_calls'][0].function.name}")
            print(f"With arguments: {tool_result['tool_calls'][0].function.arguments}")
            
            # In a real app, you would execute the tool here
            # For this example, we'll calculate the result directly
            expression = json.loads(tool_result['tool_calls'][0].function.arguments)["expression"]
            calculation_result = eval(expression)  # Note: eval is used for demonstration only
            
            # Second call with tool outputs
            tool_outputs = [{
                "tool_call_id": tool_result["tool_calls"][0].id,
                "output": str(calculation_result)
            }]
            
            final_result = process_with_openai_responses(
                input_text="Calculate 24 * 15 - 7",  # Same input
                api_key=api_key,
                model="gpt-4o",
                previous_response_id=tool_result["response_id"],
                tool_call_outputs=tool_outputs
            )
            
            if final_result["success"]:
                print(f"Final response: {final_result['response']}")
            else:
                print(f"Error: {final_result.get('error', 'Unknown error')}")
    else:
        print("\nResponses API is not available. Please update your OpenAI Python package.")