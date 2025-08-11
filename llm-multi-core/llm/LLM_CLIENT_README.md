# Universal LLM Client

This module provides a universal interface for interacting with various Large Language Model (LLM) backends. It supports text processing, vision processing, and tool/function calling capabilities across multiple LLM providers.

## Supported Backends

- **OpenAI API** (`openai_utils.py`): OpenAI's GPT models, including text, vision, and function calling
- **LiteLLM** (`litellm_utils.py`): A unified API for multiple LLM providers
- **Hugging Face** (`hfllm_utils.py`): Local or remote Hugging Face models
- **llama.cpp** (`llamacpp_utils.py`): Local models via llama.cpp server
- **Ollama** (`ollama_utils.py`): Local models via Ollama
- **vLLM** (`vllm_utils.py`): High-performance inference for open-source LLMs

## Installation

Each backend requires specific packages to be installed. The client will automatically detect which backends are available based on installed packages.

```bash
# For OpenAI
pip install openai

# For LiteLLM
pip install litellm

# For Hugging Face
pip install transformers torch pillow

# For Ollama and llama.cpp
# No additional packages required, but services must be running

# For vLLM
pip install vllm
```

## Usage

### Basic Text Processing

```python
from llmmultimem.llm.llm_client import call_llm

# Using OpenAI
response = call_llm(
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    backend="openai",
    model="gpt-3.5-turbo"
)

# Using Ollama
response = call_llm(
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    backend="ollama",
    model="llama2"
)

# Using Hugging Face
response = call_llm(
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    backend="hf",
    model="google/flan-t5-base",
    device="cpu"
)
```

### Vision Processing

```python
from llmmultimem.llm.llm_client import call_vision_llm

# Using OpenAI
response = call_vision_llm(
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": "https://example.com/image.jpg"}
        ]
    }],
    backend="openai",
    model="gpt-4-vision-preview"
)

# Using Ollama with LLaVA
response = call_vision_llm(
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": "https://example.com/image.jpg"}
        ]
    }],
    backend="ollama",
    model="llava"
)
```

### Tool/Function Calling

```python
from llmmultimem.llm.llm_client import call_llm_with_tools

# Define tools in OpenAI format
tools = [
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

# Using OpenAI
response = call_llm_with_tools(
    messages=[{"role": "user", "content": "What's the weather like in Paris?"}],
    tools=tools,
    backend="openai",
    model="gpt-4-turbo"
)

# Using LiteLLM
response = call_llm_with_tools(
    messages=[{"role": "user", "content": "What's the weather like in Paris?"}],
    tools=tools,
    backend="litellm",
    model="gpt-4-turbo"
)
```

### Streaming Responses

```python
from llmmultimem.llm.llm_client import call_llm

# Define a callback function for streaming
def stream_callback(chunk):
    print(chunk, end="", flush=True)

# Using OpenAI with streaming
response = call_llm(
    messages=[{"role": "user", "content": "Write a short poem about Paris."}],
    backend="openai",
    model="gpt-3.5-turbo",
    stream=True,
    stream_callback=stream_callback
)
```

## Advanced Configuration

Each backend supports additional configuration options that can be passed as keyword arguments:

```python
# OpenAI with custom API URL and timeout
response = call_llm(
    messages=[{"role": "user", "content": "Hello"}],
    backend="openai",
    model="gpt-3.5-turbo",
    api_url="https://custom-openai-endpoint.com/v1",
    timeout=120
)

# Hugging Face with specific device and model parameters
response = call_llm(
    messages=[{"role": "user", "content": "Hello"}],
    backend="hf",
    model="facebook/opt-1.3b",
    device="cuda:0",
    max_length=100,
    temperature=0.8
)
```

## Client Caching

The client automatically caches instances for each backend to improve performance when making multiple calls with the same configuration.

## Error Handling

Errors from the backends are caught and returned as error messages in the response:

```python
response = call_llm(
    messages=[{"role": "user", "content": "Hello"}],
    backend="nonexistent"
)
# Returns: "[Unsupported backend: nonexistent]"
```

## Running Tests

A test script is provided to verify that the client works correctly with all backends:

```bash
# Test all backends
python -m llmmultimem.llm.test_llm_client

# Test a specific backend
python -m llmmultimem.llm.test_llm_client --backend openai
```

## Backend Selection Guide

| Backend | Best For | Advantages | Limitations |
|---------|----------|------------|-------------|
| OpenAI | Production applications requiring high reliability | Best performance, supports all features | Requires API key, costs money |
| LiteLLM | Multi-provider applications | Unified API for many providers | Adds a layer of abstraction |
| Hugging Face | Local inference, customization | Full control, privacy | Requires more resources |
| llama.cpp | Efficient local inference | Low resource usage | Limited to llama-based models |
| Ollama | Easy local deployment | Simple setup, good performance | Limited model selection |
| vLLM | High-throughput applications | Optimized for performance | More complex setup |