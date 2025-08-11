import os
import json
import requests
from typing import List, Dict, Any, Optional, Union, Callable

# Import and check availability of each backend

# OpenAI
try:
    from .openai_utils import OpenAIClient, OPENAI_AVAILABLE
except ImportError:
    OPENAI_AVAILABLE = False

# LiteLLM
try:
    from .litellm_utils import LiteLLMClient, LITELLM_AVAILABLE
except ImportError:
    LITELLM_AVAILABLE = False

# Hugging Face
try:
    from .hfllm_utils import HuggingFaceClient, TRANSFORMERS_AVAILABLE as HFLLM_AVAILABLE
except ImportError:
    HFLLM_AVAILABLE = False

# llama.cpp
try:
    from .llamacpp_utils import LlamaCppClient
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False

# Ollama
try:
    from .ollama_utils import OllamaClient, extract_ollama_response_text
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# vLLM
try:
    from .vllm_utils import VLLMClient, VLLM_AVAILABLE
except ImportError:
    VLLM_AVAILABLE = False

# Cache for client instances
_client_cache = {}

def get_client(backend: str, model: str, **kwargs) -> Any:
    """
    Get or create a client instance for the specified backend.
    
    Args:
        backend: The backend to use ('openai', 'litellm', 'hf', 'llamacpp', 'ollama', 'vllm')
        model: The model name to use
        **kwargs: Additional arguments to pass to the client constructor
        
    Returns:
        A client instance for the specified backend
    """
    cache_key = f"{backend}:{model}:{json.dumps(kwargs, sort_keys=True)}"
    
    if cache_key in _client_cache:
        return _client_cache[cache_key]
    
    client = None
    
    if backend == "openai":
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package is not available. Install with 'pip install openai'")
        api_key = kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required. Provide it as a parameter or set OPENAI_API_KEY environment variable.")
        client = OpenAIClient(api_key=api_key, **kwargs)
    
    elif backend == "litellm":
        if not LITELLM_AVAILABLE:
            raise ImportError("LiteLLM package is not available. Install with 'pip install litellm'")
        client = LiteLLMClient(**kwargs)
    
    elif backend == "hf":
        if not HFLLM_AVAILABLE:
            raise ImportError("Transformers package is not available. Install with 'pip install transformers'")
        client = HuggingFaceClient(model_name=model, **kwargs)
    
    elif backend == "llamacpp":
        if not LLAMACPP_AVAILABLE:
            raise ImportError("llama.cpp utils are not available")
        client = LlamaCppClient(**kwargs)
    
    elif backend == "ollama":
        if not OLLAMA_AVAILABLE:
            raise ImportError("Ollama utils are not available")
        client = OllamaClient(**kwargs)
    
    elif backend == "vllm":
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM package is not available. Install with 'pip install vllm'")
        client = VLLMClient(model_name=model, **kwargs)
    
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    
    _client_cache[cache_key] = client
    return client

def call_llm(messages: List[Dict[str, str]], 
             backend: str = "openai", 
             model: str = "gpt-4", 
             system_prompt: Optional[str] = None,
             temperature: float = 0.7,
             max_tokens: Optional[int] = None,
             stream: bool = False,
             stream_callback: Optional[Callable[[str], None]] = None,
             **kwargs) -> str:
    """
    Universal LLM interface for text processing.
    
    Supports six backends:
    - 'openai': OpenAI API (requires API key)
    - 'litellm': LiteLLM (supports multiple providers)
    - 'hf': Hugging Face models
    - 'llamacpp': llama.cpp server
    - 'ollama': Ollama API
    - 'vllm': vLLM
    
    Args:
        messages: List of message dicts [{"role": "user", "content": "..."}, ...]
        backend: The backend to use
        model: The model name to use
        system_prompt: Optional system prompt to prepend
        temperature: Temperature for sampling (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        stream: Whether to stream the response
        stream_callback: Callback function for streaming responses
        **kwargs: Additional arguments to pass to the backend
        
    Returns:
        The generated text response
    """
    # Prepare messages with system prompt if provided
    processed_messages = messages.copy()
    if system_prompt and not any(msg.get("role") == "system" for msg in processed_messages):
        processed_messages.insert(0, {"role": "system", "content": system_prompt})
    
    try:
        # Handle each backend
        if backend == "openai":
            client = get_client(backend, model, stream_callback=stream_callback, **kwargs)
            return client.process_text(
                messages=processed_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
        
        elif backend == "litellm":
            client = get_client(backend, model, stream_callback=stream_callback, **kwargs)
            return client.process_text(
                messages=processed_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
        
        elif backend == "hf":
            client = get_client(backend, model, stream_callback=stream_callback, **kwargs)
            # Extract the last user message for text-only processing
            last_user_msg = next((msg["content"] for msg in reversed(processed_messages) 
                                if msg["role"] == "user"), "")
            return client.process_text(
                prompt=last_user_msg,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        elif backend == "llamacpp":
            client = get_client(backend, model, stream_callback=stream_callback, **kwargs)
            return client.process_text(
                messages=processed_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
        
        elif backend == "ollama":
            client = get_client(backend, model, stream_callback=stream_callback, **kwargs)
            return client.process_text(
                messages=processed_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
        
        elif backend == "vllm":
            client = get_client(backend, model, stream_callback=stream_callback, **kwargs)
            # Extract the last user message for text-only processing
            last_user_msg = next((msg["content"] for msg in reversed(processed_messages) 
                                if msg["role"] == "user"), "")
            return client.process_text(
                prompt=last_user_msg,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        else:
            return f"[Unsupported backend: {backend}]"
    
    except Exception as e:
        return f"[Error with {backend} backend: {str(e)}]"

def call_vision_llm(messages: List[Dict[str, Any]], 
                   backend: str = "openai", 
                   model: str = "gpt-4-vision-preview", 
                   system_prompt: Optional[str] = None,
                   temperature: float = 0.7,
                   max_tokens: Optional[int] = None,
                   stream: bool = False,
                   stream_callback: Optional[Callable[[str], None]] = None,
                   **kwargs) -> str:
    """
    Universal interface for vision LLMs.
    
    Supports backends that have vision capabilities:
    - 'openai': OpenAI API (requires API key)
    - 'litellm': LiteLLM (supports multiple providers)
    - 'hf': Hugging Face vision models
    - 'ollama': Ollama API with vision models
    
    Args:
        messages: List of message dicts with text and image content
                 [{"role": "user", "content": [{"type": "text", "text": "..."}, 
                                           {"type": "image_url", "image_url": "..."}]}]
        backend: The backend to use
        model: The model name to use
        system_prompt: Optional system prompt to prepend
        temperature: Temperature for sampling (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        stream: Whether to stream the response
        stream_callback: Callback function for streaming responses
        **kwargs: Additional arguments to pass to the backend
        
    Returns:
        The generated text response
    """
    # Prepare messages with system prompt if provided
    processed_messages = messages.copy()
    if system_prompt and not any(msg.get("role") == "system" for msg in processed_messages):
        processed_messages.insert(0, {"role": "system", "content": system_prompt})
    
    try:
        # Handle each backend with vision capabilities
        if backend == "openai":
            client = get_client(backend, model, stream_callback=stream_callback, **kwargs)
            return client.process_vision(
                messages=processed_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
        
        elif backend == "litellm":
            client = get_client(backend, model, stream_callback=stream_callback, **kwargs)
            return client.process_vision(
                messages=processed_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
        
        elif backend == "hf":
            client = get_client(backend, model, stream_callback=stream_callback, **kwargs)
            # Extract text and image from the last user message
            last_user_msg = next((msg for msg in reversed(processed_messages) 
                                if msg["role"] == "user"), None)
            
            if last_user_msg and isinstance(last_user_msg["content"], list):
                text_parts = [part["text"] for part in last_user_msg["content"] 
                             if part["type"] == "text"]
                image_parts = [part["image_url"] for part in last_user_msg["content"] 
                              if part["type"] == "image_url"]
                
                text = " ".join(text_parts) if text_parts else ""
                image_url = image_parts[0] if image_parts else None
                
                if image_url:
                    return client.process_vision(
                        prompt=text,
                        image=image_url,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
            
            return "[Error: No valid image found in the message]"
        
        elif backend == "ollama":
            client = get_client(backend, model, stream_callback=stream_callback, **kwargs)
            return client.process_vision(
                messages=processed_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
        
        else:
            return f"[Unsupported vision backend: {backend}]"
    
    except Exception as e:
        return f"[Error with {backend} vision backend: {str(e)}]"

def call_llm_with_tools(messages: List[Dict[str, str]],
                       tools: List[Dict[str, Any]],
                       backend: str = "openai",
                       model: str = "gpt-4-turbo",
                       system_prompt: Optional[str] = None,
                       temperature: float = 0.7,
                       max_tokens: Optional[int] = None,
                       **kwargs) -> Dict[str, Any]:
    """
    Universal interface for LLMs with tool/function calling capabilities.
    
    Currently supports:
    - 'openai': OpenAI API (requires API key)
    - 'litellm': LiteLLM (supports multiple providers with function calling)
    
    Args:
        messages: List of message dicts [{"role": "user", "content": "..."}, ...]
        tools: List of tool definitions in OpenAI format
        backend: The backend to use
        model: The model name to use
        system_prompt: Optional system prompt to prepend
        temperature: Temperature for sampling (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        **kwargs: Additional arguments to pass to the backend
        
    Returns:
        The complete response including tool calls if any
    """
    # Prepare messages with system prompt if provided
    processed_messages = messages.copy()
    if system_prompt and not any(msg.get("role") == "system" for msg in processed_messages):
        processed_messages.insert(0, {"role": "system", "content": system_prompt})
    
    try:
        # Handle backends with tool/function calling capabilities
        if backend == "openai":
            client = get_client(backend, model, **kwargs)
            return client.process_with_tools(
                messages=processed_messages,
                tools=tools,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        elif backend == "litellm":
            client = get_client(backend, model, **kwargs)
            return client.process_with_tools(
                messages=processed_messages,
                tools=tools,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        else:
            return {"error": f"Unsupported backend for tool calling: {backend}"}
    
    except Exception as e:
        return {"error": f"Error with {backend} tool calling: {str(e)}"}