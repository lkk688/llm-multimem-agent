import json
import re
import requests
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
from PIL import Image
import time
import io
import base64

# Try to import litellm, but don't fail if it's not available
try:
    import litellm
    from litellm import completion as litellm_completion
    from litellm import acompletion as litellm_acompletion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    print("Warning: litellm package not available. Install with 'pip install litellm'")

class LiteLLMClient:
    """
    Client for interacting with various LLM providers through litellm.
    
    This class provides a unified interface for:
    1. Text-only LLMs (process_text)
    2. Vision LLMs (process_vision) - for providers that support it
    3. Streaming responses
    4. Async processing
    
    It handles API communication, error handling, response parsing, and streaming.
    """
    
    def __init__(self, 
                 api_keys: Optional[Dict[str, str]] = None, 
                 timeout: int = 60, 
                 stream_callback: Optional[Callable[[str], None]] = None):
        """
        Initialize the LiteLLM client.
        
        Args:
            api_keys: Dictionary mapping provider names to API keys (e.g., {"openai": "sk-..."})
            timeout: Timeout in seconds for API requests
            stream_callback: Optional callback function for streaming responses. 
                             The callback should accept a string chunk and return None.
        """
        if not LITELLM_AVAILABLE:
            raise ImportError("litellm package is not available. Please install it with 'pip install litellm'")
        
        self.timeout = timeout
        self.stream_callback = stream_callback
        
        # Set API keys if provided
        if api_keys:
            for provider, key in api_keys.items():
                env_var = f"{provider.upper()}_API_KEY"
                litellm.api_keys[env_var] = key
    
    def process_text(self, 
                     prompt: str, 
                     model: str,
                     system_prompt: Optional[str] = None,
                     max_tokens: Optional[int] = None, 
                     temperature: float = 0.7, 
                     top_p: float = 1.0,
                     stream: bool = False,
                     tools: Optional[List[Dict[str, Any]]] = None,
                     tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
                     tool_call_outputs: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Process text using any LLM provider supported by litellm.
        
        Args:
            prompt: The text prompt to send to the model
            model: Model identifier in litellm format (e.g., "openai/gpt-4", "anthropic/claude-3-opus")
            system_prompt: Optional system prompt to set context
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter (0-1)
            stream: Whether to stream the response
            tools: Optional list of tools the model can use
            tool_choice: Optional specification for which tool to use
            tool_call_outputs: Optional list of tool outputs from previous tool calls, each containing tool_call_id and output
            
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
            # Prepare messages
            messages = []
            
            # Add system message if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add user message if prompt is not empty
            if prompt:
                messages.append({"role": "user", "content": prompt})
                
            # Add tool outputs if provided
            if tool_call_outputs:
                # We need to add an assistant message with the tool calls first
                # This is typically from a previous response
                if len(messages) == 0 or messages[-1]["role"] != "assistant":
                    # If there's no previous assistant message, we need to create one
                    # This is a placeholder and might need adjustment based on actual use
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": []
                    })
                
                # Add the tool outputs as tool messages
                for tool_output in tool_call_outputs:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_output["tool_call_id"],
                        "content": tool_output["output"]
                    })
            
            # Prepare parameters
            params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "stream": stream,
            }
            
            # Add optional parameters if provided
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
                
            if tools:
                params["tools"] = tools
                
            if tool_choice:
                params["tool_choice"] = tool_choice
            
            # For streaming responses
            if stream:
                return self._handle_streaming_request(params)
            
            # For non-streaming responses
            response = litellm_completion(**params)
            
            # Extract response text
            response_text = response.choices[0].message.content
            
            # Check if there are tool calls
            tool_calls = None
            if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
                tool_calls = response.choices[0].message.tool_calls
            
            result = {
                "success": True,
                "response": response_text,
                "raw_response": response
            }
            
            # Add tool calls if present
            if tool_calls:
                result["tool_calls"] = tool_calls
                
            return result
                
        except Exception as e:
            # Handle errors
            error_msg = f"Error calling LLM via litellm: {str(e)}"
            return {
                "success": False,
                "error": error_msg
            }
    
    def _handle_streaming_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle streaming requests to LLM providers.
        
        Args:
            params: Parameters for the LLM request
            
        Returns:
            Dictionary containing the processed result or error information
        """
        try:
            # Get streaming response
            response_stream = litellm_completion(**params)
            
            # Process the stream
            full_response = ""
            for chunk in response_stream:
                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content is not None:
                        content = delta.content
                        full_response += content
                        
                        # Call the callback if set
                        if self.stream_callback:
                            self.stream_callback(content)
            
            # Return the full response
            return {
                "success": True,
                "response": full_response,
                "raw_response": {"response": full_response}  # Simplified raw response
            }
                
        except Exception as e:
            # Handle errors
            error_msg = f"Error streaming from LLM via litellm: {str(e)}"
            return {
                "success": False,
                "error": error_msg
            }
    
    def process_vision(self, 
                      images: List[Image.Image], 
                      prompts: List[str], 
                      model: str,
                      combined_prompt: Optional[str] = None,
                      max_tokens: int = 1000,
                      stream: bool = False) -> Dict[str, Any]:
        """
        Process images using a vision-capable LLM provider.
        
        Args:
            images: List of PIL Image objects
            prompts: List of text prompts corresponding to each image
            model: Model identifier in litellm format (e.g., "openai/gpt-4-vision-preview")
            combined_prompt: Optional combined prompt for batch processing
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Dictionary containing the processed results or error information
        """
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
    
    def _process_single_vision(self, 
                              image: Image.Image, 
                              prompt: str, 
                              model: str,
                              max_tokens: int = 1000,
                              stream: bool = False) -> Dict[str, Any]:
        """
        Process a single image with a vision-capable LLM.
        
        Args:
            image: PIL Image object
            prompt: Text prompt for the image
            model: Model identifier in litellm format
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Dictionary containing the processed result or error information
        """
        try:
            # Convert image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Prepare messages with image content
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}
                        }
                    ]
                }
            ]
            
            # Prepare parameters
            params = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "stream": stream
            }
            
            # For streaming responses
            if stream:
                return self._handle_streaming_request(params)
            
            # For non-streaming responses
            response = litellm_completion(**params)
            
            # Extract response text
            response_text = response.choices[0].message.content
            
            return {
                "success": True,
                "response": response_text,
                "raw_response": response
            }
                
        except Exception as e:
            # Handle errors
            error_msg = f"Error in vision processing: {str(e)}"
            return {
                "success": False,
                "error": error_msg
            }
    
    def _process_batch_vision(self, 
                             images: List[Image.Image], 
                             combined_prompt: str, 
                             model: str,
                             max_tokens: int = 1000,
                             stream: bool = False) -> Dict[str, Any]:
        """
        Process multiple images with a single combined prompt.
        
        Args:
            images: List of PIL Image objects
            combined_prompt: Combined text prompt for all images
            model: Model identifier in litellm format
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Dictionary containing the processed result or error information
        """
        try:
            # Convert all images to base64
            image_contents = []
            for img in images:
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                image_contents.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}
                })
            
            # Prepare message content with text and all images
            content = [{"type": "text", "text": combined_prompt}]
            content.extend(image_contents)
            
            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]
            
            # Prepare parameters
            params = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "stream": stream
            }
            
            # For streaming responses
            if stream:
                return self._handle_streaming_request(params)
            
            # For non-streaming responses
            response = litellm_completion(**params)
            
            # Extract response text
            response_text = response.choices[0].message.content
            
            return {
                "success": True,
                "response": response_text,
                "raw_response": response
            }
                
        except Exception as e:
            # Handle errors
            error_msg = f"Error in batch vision processing: {str(e)}"
            return {
                "success": False,
                "error": error_msg
            }
    
    async def process_text_async(self, 
                              prompt: str, 
                              model: str,
                              system_prompt: Optional[str] = None,
                              max_tokens: Optional[int] = None, 
                              temperature: float = 0.7, 
                              top_p: float = 1.0) -> Dict[str, Any]:
        """
        Process text asynchronously using any LLM provider supported by litellm.
        
        Args:
            prompt: The text prompt to send to the model
            model: Model identifier in litellm format
            system_prompt: Optional system prompt to set context
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter (0-1)
            
        Returns:
            Dictionary containing the processed result or error information
        """
        try:
            # Prepare messages
            messages = []
            
            # Add system message if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add user message
            messages.append({"role": "user", "content": prompt})
            
            # Prepare parameters
            params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
            }
            
            # Add optional parameters if provided
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
            
            # Call the async completion function
            response = await litellm_acompletion(**params)
            
            # Extract response text
            response_text = response.choices[0].message.content
            
            return {
                "success": True,
                "response": response_text,
                "raw_response": response
            }
                
        except Exception as e:
            # Handle errors
            error_msg = f"Error calling LLM via litellm async: {str(e)}"
            return {
                "success": False,
                "error": error_msg
            }


def process_with_litellm_text(prompt: str, 
                            model: str,
                            api_keys: Optional[Dict[str, str]] = None,
                            system_prompt: Optional[str] = None,
                            max_tokens: Optional[int] = None, 
                            temperature: float = 0.7, 
                            top_p: float = 1.0,
                            timeout: int = 60,
                            stream: bool = False,
                            stream_callback: Optional[Callable[[str], None]] = None,
                            tools: Optional[List[Dict[str, Any]]] = None,
                            tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
                            tool_call_outputs: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """
    Process text using any LLM provider supported by litellm.
    
    Args:
        prompt: The text prompt to send to the model
        model: Model identifier in litellm format (e.g., "openai/gpt-4", "anthropic/claude-3-opus")
        api_keys: Dictionary mapping provider names to API keys (e.g., {"openai": "sk-..."})
        system_prompt: Optional system prompt to set context
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0-2)
        top_p: Nucleus sampling parameter (0-1)
        timeout: Timeout in seconds for API requests
        stream: Whether to stream the response
        stream_callback: Callback function for streaming responses
        tools: Optional list of tools the model can use
        tool_choice: Optional specification for which tool to use
        tool_call_outputs: Optional list of tool outputs from previous tool calls, each containing tool_call_id and output
        
    Returns:
        Dictionary containing the processed result or error information
    """
    client = LiteLLMClient(api_keys=api_keys, timeout=timeout, stream_callback=stream_callback)
    return client.process_text(prompt, model, system_prompt, max_tokens, temperature, top_p, stream, tools, tool_choice, tool_call_outputs)


def process_with_litellm_vision(images: List[Image.Image], 
                              prompts: List[str], 
                              model: str,
                              api_keys: Optional[Dict[str, str]] = None,
                              combined_prompt: Optional[str] = None,
                              max_tokens: int = 1000,
                              timeout: int = 60,
                              stream: bool = False,
                              stream_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """
    Process images using a vision-capable LLM provider through litellm.
    
    Args:
        images: List of PIL Image objects
        prompts: List of text prompts corresponding to each image
        model: Model identifier in litellm format (e.g., "openai/gpt-4-vision-preview")
        api_keys: Dictionary mapping provider names to API keys (e.g., {"openai": "sk-..."})
        combined_prompt: Optional combined prompt for batch processing
        max_tokens: Maximum number of tokens to generate
        timeout: Timeout in seconds for API requests
        stream: Whether to stream the response
        stream_callback: Callback function for streaming responses
        
    Returns:
        Dictionary containing the processed results or error information
    """
    client = LiteLLMClient(api_keys=api_keys, timeout=timeout, stream_callback=stream_callback)
    return client.process_vision(images, prompts, model, combined_prompt, max_tokens, stream)


async def process_with_litellm_text_async(prompt: str, 
                                        model: str,
                                        api_keys: Optional[Dict[str, str]] = None,
                                        system_prompt: Optional[str] = None,
                                        max_tokens: Optional[int] = None, 
                                        temperature: float = 0.7, 
                                        top_p: float = 1.0,
                                        timeout: int = 60) -> Dict[str, Any]:
    """
    Process text asynchronously using any LLM provider supported by litellm.
    
    Args:
        prompt: The text prompt to send to the model
        model: Model identifier in litellm format (e.g., "openai/gpt-4", "anthropic/claude-3-opus")
        api_keys: Dictionary mapping provider names to API keys (e.g., {"openai": "sk-..."})
        system_prompt: Optional system prompt to set context
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0-2)
        top_p: Nucleus sampling parameter (0-1)
        timeout: Timeout in seconds for API requests
        
    Returns:
        Dictionary containing the processed result or error information
    """
    client = LiteLLMClient(api_keys=api_keys, timeout=timeout)
    return await client.process_text_async(prompt, model, system_prompt, max_tokens, temperature, top_p)


def text_stream_callback(chunk: str) -> None:
    """
    Simple callback function for streaming text responses.
    
    Args:
        chunk: Text chunk from the streaming response
    """
    print(chunk, end="", flush=True)


if __name__ == "__main__":
    # Example usage
    import os
    import sys
    import json
    from PIL import Image
    
    # Example 1: Text processing with OpenAI
    print("\nText processing example with OpenAI:")
    text_result = process_with_litellm_text(
        prompt="What are the main features of Python?",
        model="openai/gpt-3.5-turbo",
        api_keys={"openai": os.environ.get("OPENAI_API_KEY", "")},
        max_tokens=100
    )
    
    if text_result["success"]:
        print(f"Response: {text_result['response']}")
    else:
        print(f"Error: {text_result.get('error', 'Unknown error')}")
    
    # Example 2: Text processing with Anthropic
    print("\nText processing example with Anthropic:")
    text_result = process_with_litellm_text(
        prompt="Explain the concept of recursion in programming.",
        model="anthropic/claude-instant-1",
        api_keys={"anthropic": os.environ.get("ANTHROPIC_API_KEY", "")},
        max_tokens=100
    )
    
    if text_result["success"]:
        print(f"Response: {text_result['response']}")
    else:
        print(f"Error: {text_result.get('error', 'Unknown error')}")
    
    # Example 3: Text processing with local Ollama model
    print("\nText processing example with Ollama:")
    text_result = process_with_litellm_text(
        prompt="What is the capital of France?",
        model="ollama/llama2",
        max_tokens=50
    )
    
    if text_result["success"]:
        print(f"Response: {text_result['response']}")
    else:
        print(f"Error: {text_result.get('error', 'Unknown error')}")
    
    # Example 4: Vision processing (if OpenAI API key is available)
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        # Find a sample image
        sample_image_paths = [
            "sample_image.jpg",
            "../sample_image.jpg",
            "../../sample_image.jpg",
        ]
        
        image_path = None
        for path in sample_image_paths:
            if os.path.exists(path):
                image_path = path
                break
        
        if image_path:
            print(f"\nVision processing example using image: {image_path}")
            image = Image.open(image_path).convert("RGB")
            
            vision_result = process_with_litellm_vision(
                images=[image],
                prompts=["Describe what you see in this image."],
                model="openai/gpt-4-vision-preview",
                api_keys={"openai": openai_api_key},
                max_tokens=100
            )
            
            if vision_result["success"]:
                print(f"Response: {vision_result['response']}")
            else:
                print(f"Error: {vision_result.get('error', 'Unknown error')}")
            
            # Example 5: Vision streaming
            print("\nVision streaming example:")
            stream_vision_result = process_with_litellm_vision(
                images=[image],
                prompts=["What objects can you identify in this image?"],
                model="openai/gpt-4-vision-preview",
                api_keys={"openai": openai_api_key},
                max_tokens=100,
                stream=True,
                stream_callback=text_stream_callback
            )
            
            print(f"\nStreaming success: {stream_vision_result['success']}")
        else:
            print("No sample image found. Skipping vision examples.")
    else:
        print("OpenAI API key not found. Skipping vision examples.")