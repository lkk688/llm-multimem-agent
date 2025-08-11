# LLM Backends for llm-multi-core

This module provides a unified interface for interacting with various LLM providers through multiple backend implementations. It supports text-only LLMs, vision-capable LLMs, streaming responses, and tool usage across different providers and deployment options.

## Features

- **Unified API**: Interact with multiple LLM providers using a consistent interface
- **Multiple Backend Support**: 
  - OpenAI API (`openai_utils.py`)
  - LiteLLM (`litellm_utils.py`) - Supports OpenAI, Anthropic, and many more providers
  - Hugging Face models (`hfllm_utils.py`)
  - llama.cpp (`llamacpp_utils.py`)
  - Ollama (`ollama_utils.py`)
  - vLLM (`vllm_utils.py`)
- **Capabilities**:
  - Text processing
  - Vision processing (for supported models)
  - Streaming responses
  - Asynchronous processing
  - Tool usage and function calling

## Installation

```bash
pip install -r requirements.txt
```

## Universal LLM Client

The `llm_client.py` module provides a unified interface for interacting with all supported LLM backends through a simple function call. This makes it easy to switch between different backends without changing your code structure.

```python
from llmmultimem.llm.llm_client import call_llm

# Create a messages list in ChatML format
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]

# Call different backends with the same interface
response_openai = call_llm(messages, backend="openai", model="gpt-4")
response_ollama = call_llm(messages, backend="ollama", model="llama2")
response_vllm = call_llm(messages, backend="vllm", model="llama2")
response_litellm = call_llm(messages, backend="litellm", model="openai/gpt-3.5-turbo")
```

This universal client is ideal for applications that need to support multiple LLM providers or want to easily switch between local and cloud-based models.

## Usage Examples

### Universal LLM Client

```python
from llmmultimem.llm.llm_client import call_llm

# Create a messages list
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What are the main features of Python?"}
]

# With OpenAI
response = call_llm(messages, backend="openai", model="gpt-4")

# With Ollama
response = call_llm(messages, backend="ollama", model="llama2")

# With vLLM
response = call_llm(messages, backend="vllm", model="llama2")

# With LiteLLM
response = call_llm(messages, backend="litellm", model="openai/gpt-3.5-turbo")
```

### OpenAI Backend

```python
from llmmultimem.llm.openai_utils import OpenAIClient

# Initialize the client
client = OpenAIClient(api_key="your-api-key")

# Process text
result = client.process_text(
    prompt="What are the main features of Python?",
    model="gpt-4o",
    max_tokens=100
)
```

### LiteLLM Backend

```python
from llmmultimem.llm.litellm_utils import LiteLLMClient

# Initialize the client
client = LiteLLMClient(api_keys={"openai": "your-api-key"})

# Process text with OpenAI through LiteLLM
result = client.process_text(
    prompt="What are the main features of Python?",
    model="openai/gpt-3.5-turbo",
    max_tokens=100
)

# With Anthropic through LiteLLM
result = client.process_text(
    prompt="Explain the concept of recursion in programming.",
    model="anthropic/claude-3-opus",
    api_keys={"anthropic": "your-api-key"},
    max_tokens=100
)
```

### Hugging Face Backend

```python
from llmmultimem.llm.hfllm_utils import HuggingFaceClient

# Initialize the client with a text model
client = HuggingFaceClient(model_name="meta-llama/Llama-2-7b-chat-hf")

# Process text
result = client.process_text(
    prompt="What are the main features of Python?",
    max_tokens=100
)
```

### llama.cpp Backend

```python
from llmmultimem.llm.llamacpp_utils import LlamaCppClient

# Initialize the client (assuming llama.cpp server is running)
client = LlamaCppClient()

# Process text
result = client.process_text(
    prompt="What are the main features of Python?",
    model="llama2",  # Model loaded in llama.cpp server
    max_tokens=100
)
```

### Ollama Backend

```python
from llmmultimem.llm.ollama_utils import OllamaClient

# Initialize the client (assuming Ollama is running)
client = OllamaClient()

# Process text
result = client.process_text(
    prompt="What are the main features of Python?",
    model="llama2",  # Model available in Ollama
    system_prompt="You are a helpful assistant."
)
```

### vLLM Backend

```python
from llmmultimem.llm.vllm_utils import VLLMClient

# Initialize the client with a model
client = VLLMClient(model_name="meta-llama/Llama-2-7b-chat-hf")

# Process text
result = client.process_text(
    prompt="What are the main features of Python?",
    max_tokens=100
)
```

### Streaming Responses

```python
from llmmultimem.llm.openai_utils import OpenAIClient

# Define a custom callback
def my_callback(chunk):
    print(chunk, end="", flush=True)

# Initialize client with stream callback
client = OpenAIClient(api_key="your-api-key", stream_callback=my_callback)

# Stream with the callback
result = client.process_text(
    prompt="List the first 5 prime numbers and explain why they are prime.",
    model="gpt-4o",
    max_tokens=150,
    stream=True
)
```

### Vision Processing

```python
from PIL import Image
from llmmultimem.llm.openai_utils import OpenAIClient

# Initialize the client
client = OpenAIClient(api_key="your-api-key")

# Load an image
image = Image.open("sample_image.jpg").convert("RGB")

# Process with a vision-capable model
result = client.process_vision(
    images=[image],
    prompt="Describe what you see in this image.",
    model="gpt-4o",
    max_tokens=100
)
```

#### Vision Processing with Other Backends

Many of the backends support vision models:

```python
# With Ollama
from llmmultimem.llm.ollama_utils import OllamaClient

client = OllamaClient()
result = client.process_vision(
    images=[image],
    prompt="Describe what you see in this image.",
    model="llava",  # Requires llava model in Ollama
)

# With Hugging Face
from llmmultimem.llm.hfllm_utils import HuggingFaceClient

client = HuggingFaceClient(model_name="llava-hf/llava-1.5-7b-hf")
result = client.process_vision(
    images=[image],
    prompt="Describe what you see in this image.",
)
```

### Tool Usage / Function Calling

```python
import json
from llmmultimem.llm.openai_utils import OpenAIClient

# Initialize the client
client = OpenAIClient(api_key="your-api-key")

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
tool_result = client.process_text_with_tools(
    prompt="What is 25 multiplied by 13?",
    model="gpt-4o",
    tools=[calculator_tool],
    tool_choice="auto"
)

# Process tool calls and get final response
if tool_result["success"] and "tool_calls" in tool_result:
    tool_calls = tool_result["tool_calls"]
    for tool_call in tool_calls:
        # Parse the arguments
        args = json.loads(tool_call["function"]["arguments"])
        
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
        
        # Send the result back to the model
        tool_outputs = [
            {
                "tool_call_id": tool_call["id"],
                "output": str(result)
            }
        ]
        
        # Get the final response
        final_result = client.process_text_with_tools(
            prompt="",  # No new prompt needed
            model="gpt-4o",
            tools=[calculator_tool],
            tool_choice="auto",
            tool_call_outputs=tool_outputs
        )
```

#### Tool Usage with LiteLLM

LiteLLM also supports tool usage for compatible models:

```python
from llmmultimem.llm.litellm_utils import LiteLLMClient

# Initialize the client
client = LiteLLMClient(api_keys={"openai": "your-api-key"})

# Call the model with the tool
tool_result = client.process_text(
    prompt="What is 25 multiplied by 13?",
    model="openai/gpt-4",
    tools=[calculator_tool],
    tool_choice="auto"
)

# Process tool calls as shown in the previous example
```

## Running the Test Scripts

Test scripts are provided to demonstrate the functionality of each backend:

```bash
# Test the universal client
python -m llmmultimem.llm.tests.test_llm_client

# Test specific backends
python -m llmmultimem.llm.tests.test_openai
python -m llmmultimem.llm.tests.test_litellm
python -m llmmultimem.llm.tests.test_hfllm
python -m llmmultimem.llm.tests.test_llamacpp
python -m llmmultimem.llm.tests.test_ollama
python -m llmmultimem.llm.tests.test_vllm
```

Make sure to set the appropriate API keys as environment variables before running the test scripts:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

