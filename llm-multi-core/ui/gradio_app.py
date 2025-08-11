import time
import subprocess
import json
import os
import threading
import sys
import argparse
import platform
import requests
import gradio as gr
from gradio.themes.soft import Soft  # Explicit and clean
from typing import List, Dict, Any, Optional

# Import our modularized system monitoring utilities
# Import system monitoring utilities from local directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.performance_monitor import system_monitor # pylint: disable=import-error

# Import LLM client
try:
    from llm.llm_client import call_llm, get_client
    from llm.llm_client import OPENAI_AVAILABLE, LITELLM_AVAILABLE, HFLLM_AVAILABLE, LLAMACPP_AVAILABLE, OLLAMA_AVAILABLE, VLLM_AVAILABLE
    print("✅ Successfully imported LLM client")
except ImportError:
    # Fallback to relative import
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from llm_multi_core.llm.llm_client import call_llm, get_client
        from llm_multi_core.llm.llm_client import OPENAI_AVAILABLE, LITELLM_AVAILABLE, HFLLM_AVAILABLE, LLAMACPP_AVAILABLE, OLLAMA_AVAILABLE, VLLM_AVAILABLE
        print("✅ Successfully imported LLM client from llm_multi_core")
    except ImportError as e:
        print(f"⚠️ Error importing LLM client: {e}")
        # Define fallback values if import fails
        OPENAI_AVAILABLE = False
        LITELLM_AVAILABLE = False
        HFLLM_AVAILABLE = False
        LLAMACPP_AVAILABLE = True  # Assume llama.cpp is available as fallback
        OLLAMA_AVAILABLE = True    # Assume Ollama is available as fallback
        VLLM_AVAILABLE = False
        
        # Define dummy functions if import fails
        def call_llm(*args, **kwargs):
            raise ImportError("LLM client not available")
        
        def get_client(*args, **kwargs):
            return None

# Settings
# Dynamically build the list of available backends
BACKENDS = []
if OPENAI_AVAILABLE:
    BACKENDS.append("openai")
if LITELLM_AVAILABLE:
    BACKENDS.append("litellm")
if HFLLM_AVAILABLE:
    BACKENDS.append("hf")
if LLAMACPP_AVAILABLE:
    BACKENDS.append("llamacpp")
if OLLAMA_AVAILABLE:
    BACKENDS.append("ollama")
if VLLM_AVAILABLE:
    BACKENDS.append("vllm")

# Fallback to default backends if none are available
if not BACKENDS:
    BACKENDS = ["ollama", "llama.cpp"]
    print("⚠️ No LLM backends available, falling back to default backends")

# API URLs (for backward compatibility)
OLLAMA_API_URL = "http://localhost:11434/api/generate"
LLAMACPP_API_URL = "http://localhost:8000/completion"
CHAT_LOG_PATH = "./chat_logs"
os.makedirs(CHAT_LOG_PATH, exist_ok=True)

# Initialize system monitoring
platform_info = system_monitor.init_monitoring()
IS_JETSON = platform_info["is_jetson"]
IS_APPLE_SILICON = platform_info["is_apple_silicon"]
HAS_NVIDIA_GPU = platform_info["has_nvidia_gpu"]

# Start the appropriate monitoring thread and get the system_info dictionary
system_info = system_monitor.start_monitoring()

# Fetch available models
def list_models(backend="ollama"):
    try:
        # Use the original implementation for backward compatibility
        if backend == "ollama" and OLLAMA_AVAILABLE:
            try:
                r = requests.get("http://localhost:11434/api/tags", timeout=5)
                if r.ok:
                    models = [m["name"] for m in r.json().get("models", [])]
                    if models:
                        return models
                    else:
                        print(f"No models found for Ollama. Pull a model with 'ollama pull <model>'")
                        return ["llama3", "mistral", "gemma", "phi3"], "No models found. Pull a model with 'ollama pull <model>'"
            except Exception as e:
                print(f"Error connecting to Ollama: {e}")
                return ["llama3", "mistral", "gemma", "phi3"], f"Error connecting to Ollama: {str(e)}"
        
        # Default models for each backend
        if backend == "openai":
            models = ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
            return models, f"Available OpenAI models: {len(models)}"
        elif backend == "litellm":
            models = ["gpt-3.5-turbo", "gpt-4", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307", "gemini-pro"]
            return models, f"Available LiteLLM models: {len(models)}"
        elif backend == "hf":
            models = ["meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf", "mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mixtral-8x7B-Instruct-v0.1", "microsoft/phi-2"]
            return models, f"Available Hugging Face models: {len(models)}"
        elif backend == "llamacpp":
            try:
                r = requests.get("http://localhost:8000/models", timeout=5)
                if r.ok and r.json().get("models"):
                    models = r.json().get("models")
                    return models, f"Found {len(models)} llama.cpp models"
                else:
                    return ["llama.cpp-default", "llama-3-8b", "mistral-7b"], "Could not get models from llama.cpp server"
            except Exception as e:
                return ["llama.cpp-default", "llama-3-8b", "mistral-7b"], f"Error connecting to llama.cpp: {str(e)}"
        elif backend == "ollama":
            models = ["llama3", "mistral", "gemma", "phi3"]
            return models, f"Default Ollama models: {len(models)}"
        elif backend == "vllm":
            models = ["meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf", "mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mixtral-8x7B-Instruct-v0.1"]
            return models, f"Available vLLM models: {len(models)}"
        else:
            return [], f"Backend {backend} not available or not supported"
    except Exception as e:
        print(f"Error listing models for {backend}: {e}")
        return [], f"Error listing models: {str(e)}"
    
    return []

# Chat streaming
def chat_with_backend_stream(prompt, model, backend, history=None):
    if history is None:
        history = []
    
    start = time.time()
    response = ""
    tokens = 0
    error_msg = None
    
    # Add user message to history immediately for better UX
    history.append({"role": "user", "content": prompt})
    # Create a placeholder for assistant response
    history.append({"role": "assistant", "content": "Thinking..."})
    
    # Convert history to the format expected by llm_client
    messages = []
    for msg in history[:-1]:  # Exclude the placeholder
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    try:
        # Use llm_client.py for all backends if available
        if backend in BACKENDS:
            try:
                # Call the LLM using the unified client
                response = call_llm(
                    messages=messages,
                    backend=backend,
                    model=model,
                    temperature=0.7,
                    max_tokens=1024,
                    stream=False
                )
                
                # Update the assistant's message in history
                history[-1]["content"] = response
                # Calculate tokens (approximate)
                tokens = len(response.split())
                
            except Exception as e:
                error_msg = f"[ERROR] Failed to process with {backend} backend: {str(e)}"
                history[-1]["content"] = error_msg
                return history, history, "N/A"
        
        # Fallback to original implementation for backward compatibility
        elif backend == "ollama":
            # Check if Ollama server is running
            try:
                server_check = requests.get("http://localhost:11434/", timeout=2)
                if not server_check.ok:
                    error_msg = "[ERROR] Ollama server is not responding. Make sure it's running."
                    history[-1]["content"] = error_msg
                    return history, history, "N/A"
                
                # Check if the model exists
                models_response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if not models_response.ok:
                    error_msg = f"[ERROR] Failed to get models list: {models_response.status_code} - {models_response.text}"
                    history[-1]["content"] = error_msg
                    return history, history, "N/A"
                
                models_data = models_response.json()
                available_models = [m["name"] for m in models_data.get("models", [])]
                
                if not available_models:
                    error_msg = "[ERROR] No models available in Ollama. Please pull a model first using 'ollama pull <model>'"
                    history[-1]["content"] = error_msg
                    return history, history, "N/A"
                
                if model not in available_models:
                    error_msg = f"[ERROR] Selected model '{model}' not found. Available models: {', '.join(available_models)}"
                    history[-1]["content"] = error_msg
                    return history, history, "N/A"
                
                # Make the API request
                payload = {"model": model, "prompt": prompt, "stream": False}
                with requests.post(OLLAMA_API_URL, json=payload, timeout=60) as r:
                    if not r.ok:
                        error_msg = f"[ERROR] API returned status code {r.status_code}: {r.text}"
                        history[-1]["content"] = error_msg
                        return history, history, "N/A"
                    
                    data = r.json()
                    response = data.get("response", "")
                    history[-1]["content"] = response
                    tokens = len(response.split())
            except Exception as e:
                error_msg = f"[ERROR] Error with Ollama: {str(e)}"
                history[-1]["content"] = error_msg
                return history, history, "N/A"
        
        elif backend == "llama.cpp":
            try:
                # Make the API request
                payload = {"prompt": prompt, "n_predict": 128, "stream": False}
                with requests.post(LLAMACPP_API_URL, json=payload, timeout=60) as r:
                    if not r.ok:
                        error_msg = f"[ERROR] API returned status code {r.status_code}: {r.text}"
                        history[-1]["content"] = error_msg
                        return history, history, "N/A"
                    
                    data = r.json()
                    response = data.get("content", "")
                    history[-1]["content"] = response
                    tokens = len(response.split())
            except Exception as e:
                error_msg = f"[ERROR] Error with llama.cpp: {str(e)}"
                history[-1]["content"] = error_msg
                return history, history, "N/A"
        
        else:
            error_msg = f"[ERROR] Unsupported backend: {backend}"
            history[-1]["content"] = error_msg
            return history, history, "N/A"
    
    except Exception as e:
        error_msg = f"[ERROR] Unexpected error: {str(e)}"
        history[-1]["content"] = error_msg
        return history, history, "N/A"
    
    # If there was no response
    if not response:
        history[-1]["content"] = f"[ERROR] No response received from model '{model}' with backend '{backend}'."
        return history, history, "N/A"
    
    elapsed = time.time() - start
    tps = f"{tokens / elapsed:.2f} tokens/sec" if tokens > 0 and elapsed > 0 else "N/A"
    
    return history, history, tps

# Streaming version
def chat_with_backend_stream_streaming(prompt, model, backend, history=None):
    if history is None:
        history = []
    
    start = time.time()
    response = ""
    tokens = 0
    error_msg = None
    
    # Add user message to history immediately for better UX
    history.append({"role": "user", "content": prompt})
    # Create a placeholder for assistant response
    history.append({"role": "assistant", "content": ""})
    
    # Convert history to the format expected by llm_client
    messages = []
    for msg in history[:-1]:  # Exclude the placeholder
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    try:
        # Use llm_client.py for all backends if available
        if backend in BACKENDS:
            try:
                # Get the appropriate client for streaming
                client = get_client(backend)
                if client is None:
                    error_msg = f"[ERROR] Failed to initialize {backend} client"
                    history[-1]["content"] = error_msg
                    yield history, history, "N/A"
                    return
                
                # Call the LLM using the unified client with streaming
                for chunk in call_llm(
                    messages=messages,
                    backend=backend,
                    model=model,
                    temperature=0.7,
                    max_tokens=1024,
                    stream=True
                ):
                    if chunk:
                        response += chunk
                        history[-1]["content"] = response
                        tokens += 1
                        yield history, history, "Streaming..."
                
            except Exception as e:
                error_msg = f"[ERROR] Failed to process with {backend} backend: {str(e)}"
                history[-1]["content"] = error_msg
                yield history, history, "N/A"
                return
        
        # Fallback to original implementation for backward compatibility
        elif backend == "ollama" or backend == "llama.cpp":
            # Prepare API request based on backend
            if backend == "ollama":
                payload = {"model": model, "prompt": prompt, "stream": True}
                url = OLLAMA_API_URL
            else:  # llama.cpp
                payload = {"prompt": prompt, "n_predict": 128, "stream": True}
                url = LLAMACPP_API_URL
            
            with requests.post(url, json=payload, stream=True, timeout=60) as r:
                if not r.ok:
                    error_msg = f"[ERROR] API returned status code {r.status_code}: {r.text}"
                    history[-1]["content"] = error_msg
                    yield history, history, "N/A"
                    return
                
                # Process streaming response
                for line in r.iter_lines():
                    if line:
                        try:
                            # Different parsing for different backends
                            if backend == "ollama":
                                try:
                                    chunk = json.loads(line)
                                    token = chunk.get("response", "")
                                    if token:
                                        response += token
                                        history[-1]["content"] = response
                                        tokens += 1
                                        yield history, history, "Streaming..."
                                    
                                    # Check if done
                                    if chunk.get("done", False):
                                        break
                                except json.JSONDecodeError:
                                    # Skip invalid JSON lines
                                    continue
                            else:  # llama.cpp
                                try:
                                    chunk = json.loads(line)
                                    token = chunk.get("content", "")
                                    if token:
                                        response += token
                                        history[-1]["content"] = response
                                        tokens += 1
                                        yield history, history, "Streaming..."
                                    
                                    # Check if done
                                    if chunk.get("stop", False):
                                        break
                                except json.JSONDecodeError:
                                    # Skip invalid JSON lines
                                    continue
                        except Exception as e:
                            error_msg = f"[ERROR] Failed to process chunk: {str(e)}"
                            history[-1]["content"] = error_msg
                            yield history, history, "N/A"
                            return
        
        else:
            error_msg = f"[ERROR] Unsupported backend: {backend}"
            history[-1]["content"] = error_msg
            yield history, history, "N/A"
            return
    
    except requests.RequestException as e:
        error_msg = f"[ERROR] Connection error: {str(e)}"
        history[-1]["content"] = error_msg
        yield history, history, "N/A"
        return
    except Exception as e:
        error_msg = f"[ERROR] Unexpected error: {str(e)}"
        history[-1]["content"] = error_msg
        yield history, history, "N/A"
        return
    
    # If there was no response
    if not response:
        history[-1]["content"] = f"[ERROR] No response received from model '{model}' with backend '{backend}'."
        yield history, history, "N/A"
        return
    
    elapsed = time.time() - start
    tps = f"{tokens / elapsed:.2f} tokens/sec" if tokens > 0 and elapsed > 0 else "N/A"
    
    yield history, history, tps

# Export chat history
def export_chat(history):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    md_path = os.path.join(CHAT_LOG_PATH, f"chat_{timestamp}.md")
    json_path = os.path.join(CHAT_LOG_PATH, f"chat_{timestamp}.json")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    
    with open(md_path, "w") as md_file:
        # Process messages in pairs (user, assistant)
        i = 0
        while i < len(history):
            user_msg = history[i] if i < len(history) else None
            assistant_msg = history[i+1] if i+1 < len(history) else None
            
            if user_msg and user_msg.get("role") == "user":
                md_file.write(f"**User:** {user_msg.get('content', '')}\n\n")
            
            if assistant_msg and assistant_msg.get("role") == "assistant":
                md_file.write(f"**Assistant:** {assistant_msg.get('content', '')}\n\n---\n")
            
            i += 2
    
    with open(json_path, "w") as json_file:
        json.dump(history, json_file, indent=2)
    
    return f"✅ Exported:\n- {md_path}\n- {json_path}"

# UI
with gr.Blocks(title="Ollama Chat UI", theme=Soft(primary_hue="indigo")) as demo:
    # Get platform name from system_monitor
    platform_name = system_monitor.get_platform_name()
    
    gr.Markdown(f"## 🧠 LLM Multi-Core Chat UI ({platform_name})")
    
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                backend_select = gr.Radio(
                    BACKENDS, 
                    value=BACKENDS[0] if BACKENDS else "ollama", 
                    label="LLM Backend",
                    info="Select from available LLM backends",
                    interactive=True,
                    scale=1
                )
                model_dropdown = gr.Dropdown(
                    choices=list_models("ollama"), 
                    label="Model",
                    info="Available models for the selected backend",
                    interactive=True,
                    scale=2
                )
            
            # Main chat interface
            chatbot = gr.Chatbot(
                label="Chat History",
                type="messages",
                height=600,  # Increased height for better visibility
                show_copy_button=True,
                avatar_images=(None, "https://em-content.zobj.net/source/twitter/376/robot_1f916.png"),
                bubble_full_width=False,
                show_label=True
            )
            
            with gr.Row():
                prompt = gr.Textbox(
                    label="Prompt", 
                    placeholder="Ask something...",
                    lines=2,
                    max_lines=5,
                    show_label=False,
                    scale=5
                )
                send = gr.Button("Send", scale=1, variant="primary")
            
            with gr.Row():
                clear_btn = gr.Button("🧹 Clear Chat", scale=1)
                refresh_btn = gr.Button("🔄 Refresh Models", scale=1)
                export_btn = gr.Button("💾 Export Chat", scale=1)
        
        # Right sidebar for system info
        with gr.Column(scale=1):
            dashboard = gr.Textbox(
                label="Live System Info", 
                lines=15, 
                max_lines=20,
                interactive=False, 
                value=system_info["text"],
                show_label=True
            )
            token_speed = gr.Textbox(label="Token Speed", interactive=False)
            
            # Platform-specific info
            if IS_JETSON:
                gr.Markdown("### Jetson Device")
                gr.Markdown("Optimized for NVIDIA Jetson hardware")
                gr.Markdown("*For best performance, ensure jetson-stats is installed*")
            elif IS_APPLE_SILICON:
                gr.Markdown("### Apple Silicon")
                gr.Markdown("Optimized for Apple M1/M2/M3 hardware")
                gr.Markdown("*For detailed GPU stats, try tools like 'mactop' (brew install mactop), 'macmon' (brew install macmon), or 'asitop' (pip install asitop)*")
            elif HAS_NVIDIA_GPU:
                gr.Markdown("### NVIDIA GPU")
                gr.Markdown("Optimized for NVIDIA graphics cards")
                gr.Markdown("*Using GPUtil or nvidia-smi for monitoring*")
            else:
                gr.Markdown("### CPU Mode")
                gr.Markdown("Running in CPU-only mode")
                gr.Markdown("*For better performance, consider using a GPU*")
    
    # Hidden state
    state = gr.State([])
    
    # Functions
    def refresh_dashboard():
        return system_info["text"]
    
    def update_model_list(backend):
        # In newer versions of Gradio, we return a list directly instead of using .update()
        models = list_models(backend)
        return gr.Dropdown(choices=models)
    
    def export_trigger(history):
        result = export_chat(history)
        gr.Info("Chat history exported successfully")
        return result
    
    def clear_chat():
        return [], []
    
    # Main chat function that handles both streaming and non-streaming
    def chat_fn(prompt, model, backend, history, streaming=True):
        if not prompt.strip():
            return "", history, "N/A"
        
        # Validate backend and model selection
        if not backend or backend not in BACKENDS:
            error_msg = f"[ERROR] Invalid backend selected: {backend}. Available backends: {', '.join(BACKENDS)}"
            if not history:
                history = []
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": error_msg})
            return "", history, "N/A"
        
        if not model:
            error_msg = f"[ERROR] No model selected for backend: {backend}. Please select a model."
            if not history:
                history = []
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": error_msg})
            return "", history, "N/A"
        
        # Process the chat with the selected backend and model
        if streaming:
            return "", chat_with_backend_stream_streaming(prompt, model, backend, history), "Streaming..."
        else:
            _, history, tps = chat_with_backend_stream(prompt, model, backend, history)
            return "", history, tps
    
    # Event handlers using the new Gradio 4.0+ syntax
    send_event = send.click(
        fn=chat_with_backend_stream,
        inputs=[prompt, model_dropdown, backend_select, state],
        outputs=[chatbot, state, token_speed],
        api_name="chat"
    )
    
    # Also trigger on Enter key
    prompt_event = prompt.submit(
        fn=chat_with_backend_stream,
        inputs=[prompt, model_dropdown, backend_select, state],
        outputs=[chatbot, state, token_speed]
    )
    
    refresh_event = refresh_btn.click(
        fn=update_model_list, 
        inputs=[backend_select], 
        outputs=[model_dropdown]
    )
    
    export_event = export_btn.click(
        fn=export_trigger, 
        inputs=[state], 
        outputs=[dashboard]
    )
    
    clear_event = clear_btn.click(
        fn=clear_chat,
        inputs=[],
        outputs=[chatbot, state]
    )
    
    # Real-time dashboard refresh every 2 seconds
    timer = gr.Timer(2)
    timer_event = timer.every(
        fn=refresh_dashboard, 
        inputs=None,
        outputs=[dashboard]
    )

if __name__ == "__main__":
    # Performance optimizations based on platform
    if IS_JETSON:
        print("Running on Jetson device - applying optimizations")
        # Reduce memory usage by limiting thread pool
        try:
            import torch
            if torch.cuda.is_available():
                # Set lower memory usage for CUDA if available
                #caps the memory usage to 70% of total GPU memory to avoid out-of-memory crashes and allow other GPU tasks to run.
                torch.cuda.set_per_process_memory_fraction(0.7)  # Use only 70% of GPU memory
                print(f"CUDA available: {torch.cuda.get_device_name(0)}")
                print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        except ImportError:
            print("PyTorch not available, skipping CUDA optimizations")
        
        # Set environment variables for better performance
        #Gradio by default collects usage analytics.
	    #On a resource-constrained Jetson device, turning this off can save bandwidth and slightly improve startup performance.
        os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"  # Disable analytics
        #PCI_BUS_ID ensures that CUDA device enumeration follows the PCI bus order, which is reliable for multi-GPU setups.
	    #On Jetson devices, which usually only have one GPU, this line is not necessary
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Use PCI bus ID for CUDA devices
    
    elif IS_APPLE_SILICON:
        print("Running on Apple Silicon - applying optimizations")
        # Set environment variables for better performance on Apple Silicon
        os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"  # Disable analytics
        
        # Check if PyTorch is available and configured for MPS (Metal Performance Shaders)
        try:
            import torch
            if torch.backends.mps.is_available():
                print("MPS (Metal Performance Shaders) is available for GPU acceleration")
                # No need to set memory fraction as Apple Silicon manages memory differently
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable MPS fallback for operations not supported by MPS
        except ImportError:
            print("PyTorch not available, skipping MPS optimizations")
        except AttributeError:
            print("PyTorch available but MPS not supported in this version")
    
    elif HAS_NVIDIA_GPU:
        print("Running on system with NVIDIA GPU - applying optimizations")
        # Optimize for NVIDIA GPU
        try:
            import torch
            if torch.cuda.is_available():
                # Set reasonable memory usage for CUDA
                torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
                print(f"CUDA available: {torch.cuda.get_device_name(0)}")
                print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        except ImportError:
            print("PyTorch not available, skipping CUDA optimizations")
        
        # Set environment variables for better performance
        os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"  # Disable analytics
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Use PCI bus ID for CUDA devices
    
    else:
        print("Running on CPU-only system")
        # Set environment variables for better performance on CPU
        os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"  # Disable analytics
    
    # Check if we need to start ollama server
    parser = argparse.ArgumentParser(description="LLM Multi-Core Gradio UI")
    parser.add_argument("--start-ollama", action="store_true", help="Start ollama server")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the Gradio server on")
    parser.add_argument("--share", action="store_true", help="Create a public share link")
    parser.add_argument("--backend", type=str, help="Specify default LLM backend to use")
    args = parser.parse_args()
    
    # Check if a specific backend was requested
    if args.backend and args.backend in BACKENDS:
        print(f"✅ Using specified backend: {args.backend}")
        # The backend will be set as default when the UI loads
    elif args.backend:
        print(f"⚠️ Requested backend '{args.backend}' not available. Using default.")
        print(f"Available backends: {', '.join(BACKENDS)}")
    
    # Print available backends
    print("\n🔌 Available LLM Backends:")
    for backend_name in BACKENDS:
        print(f"  - {backend_name}")
    
    if args.start_ollama:
        print("Starting ollama server...")
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Waiting for ollama server to start...")
        time.sleep(2)  # Give ollama time to start
    
    # Launch the Gradio app
    print(f"Starting Gradio UI on port {args.port}...")
    demo.queue(max_size=10).launch(
        server_name="0.0.0.0", 
        server_port=args.port,
        share=args.share,
        show_error=True
        # Removed favicon_path 
    )
    
    # Command for Docker/container environments:
    # $EXEC_CMD bash -c "ollama serve & sleep 2 && python3 /workspace/scripts/ollama_gradio_ui.py"