#!/usr/bin/env python3
"""
Docker Container Test Script for VisionLangAnnotate

This script tests all components of the VisionLangAnnotate Docker container:
- System configuration
- CUDA devices
- Python packages
- llama.cpp functionality
- Ollama availability
- vLLM functionality

Usage:
    python test_docker_container.py
"""

import os
import sys
import json
import subprocess
import platform
import importlib.util
from pathlib import Path
import requests
import time

# ANSI color codes for prettier output
COLORS = {
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "RED": "\033[91m",
    "BLUE": "\033[94m",
    "BOLD": "\033[1m",
    "END": "\033[0m"
}

def print_header(message):
    """Print a formatted header"""
    print(f"\n{COLORS['BOLD']}{COLORS['BLUE']}==== {message} ===={COLORS['END']}")

def print_success(message):
    """Print a success message"""
    print(f"{COLORS['GREEN']}✓ {message}{COLORS['END']}")

def print_warning(message):
    """Print a warning message"""
    print(f"{COLORS['YELLOW']}⚠ {message}{COLORS['END']}")

def print_error(message):
    """Print an error message"""
    print(f"{COLORS['RED']}✗ {message}{COLORS['END']}")

def run_command(command, shell=True):
    """Run a shell command and return the output"""
    try:
        result = subprocess.run(
            command,
            shell=shell,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {command}")
        print_error(f"Error: {e.stderr}")
        return None

def check_package_installed(package_name):
    """Check if a Python package is installed"""
    return importlib.util.find_spec(package_name) is not None

def get_package_version(package_name):
    """Get the version of an installed package"""
    if not check_package_installed(package_name):
        return None
    
    try:
        package = __import__(package_name)
        return getattr(package, '__version__', 'Unknown')
    except (ImportError, AttributeError):
        try:
            version = run_command(f"pip show {package_name} | grep Version | cut -d' ' -f2")
            return version
        except:
            return "Unknown"

def check_system():
    """Check system configuration"""
    print_header("System Information")
    
    # Check OS
    os_info = platform.platform()
    print(f"Operating System: {os_info}")
    
    # Check Python version
    python_version = platform.python_version()
    print(f"Python Version: {python_version}")
    
    # Check environment variables
    env_vars = [
        "CUDA_HOME",
        "LD_LIBRARY_PATH",
        "PATH",
        "HF_HOME",
        "TRANSFORMERS_CACHE"
    ]
    
    print("\nEnvironment Variables:")
    for var in env_vars:
        value = os.environ.get(var, "Not set")
        if value != "Not set":
            print_success(f"{var}: {value}")
        else:
            print_warning(f"{var}: Not set")
    
    # Check if running in Docker
    in_docker = os.path.exists("/.dockerenv")
    if in_docker:
        print_success("Running inside Docker container")
    else:
        print_warning("Not running inside Docker container")

def check_cuda():
    """Check CUDA devices and configuration"""
    print_header("CUDA Configuration")
    
    # Check if CUDA is available through nvidia-smi
    nvidia_smi = run_command("nvidia-smi")
    if nvidia_smi:
        print_success("NVIDIA GPU detected")
        # Extract some basic info
        gpu_info = run_command("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader")
        if gpu_info:
            for i, line in enumerate(gpu_info.strip().split('\n')):
                print(f"GPU {i}: {line}")
    else:
        print_error("NVIDIA GPU not detected or nvidia-smi not available")
    
    # Check CUDA version
    nvcc_version = run_command("nvcc --version")
    if nvcc_version:
        cuda_version = nvcc_version.split("release")[1].split(",")[0].strip()
        print_success(f"CUDA Version: {cuda_version}")
    else:
        print_warning("nvcc not found, cannot determine CUDA version")
    
    # Check cuBLAS
    if os.path.exists("/usr/local/cuda/lib64/libcublas.so"):
        print_success("cuBLAS library found")
    else:
        print_error("cuBLAS library not found")

def check_python_packages():
    """Check installed Python packages"""
    print_header("Python Packages")
    
    required_packages = [
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "scikit-learn",
        "tqdm",
        "huggingface_hub",
        "transformers",
        "sentence_transformers",
        "langchain",
        "requests",
        "fastapi",
        "uvicorn",
        "vllm",
        "llama_cpp_python",  # This is how it's imported
        "PIL",  # Pillow
    ]
    
    for package in required_packages:
        import_name = package
        # Handle special cases
        if package == "llama_cpp_python":
            import_name = "llama_cpp"
        elif package == "PIL":
            import_name = "PIL"
        
        if check_package_installed(import_name):
            version = get_package_version(import_name)
            print_success(f"{package}: {version}")
        else:
            print_error(f"{package}: Not installed")

def test_llamacpp():
    """Test llama.cpp functionality"""
    print_header("Testing llama.cpp")
    
    # Check if llama.cpp binary exists
    llamacpp_path = "/opt/llama.cpp/build/bin/main"
    if os.path.exists(llamacpp_path):
        print_success(f"llama.cpp binary found at {llamacpp_path}")
        
        # Check version
        version_output = run_command(f"{llamacpp_path} --version")
        if version_output:
            print_success(f"llama.cpp version: {version_output}")
        else:
            print_warning("Could not determine llama.cpp version")
    else:
        print_error(f"llama.cpp binary not found at {llamacpp_path}")
    
    # Test llama-cpp-python
    try:
        import llama_cpp
        print_success(f"llama-cpp-python imported successfully (version: {llama_cpp.__version__})")
        
        # Check if CUDA is enabled in llama-cpp-python
        if hasattr(llama_cpp, "LLAMA_CUDA_AVAILABLE") and llama_cpp.LLAMA_CUDA_AVAILABLE:
            print_success("CUDA support is enabled in llama-cpp-python")
        else:
            print_warning("CUDA support might not be enabled in llama-cpp-python")
            
    except ImportError as e:
        print_error(f"Failed to import llama_cpp: {e}")
    
    # Check if llama.cpp server can start (without actually loading a model)
    try:
        # Just check if the server module exists
        from llama_cpp import server
        print_success("llama-cpp-python server module is available")
    except ImportError as e:
        print_error(f"llama-cpp-python server module not available: {e}")

def test_ollama():
    """Test Ollama availability and functionality"""
    print_header("Testing Ollama")
    
    # Check if Ollama binary exists
    ollama_path = run_command("which ollama")
    if ollama_path:
        print_success(f"Ollama binary found at {ollama_path}")
        
        # Check Ollama version
        version_output = run_command("ollama --version")
        if version_output:
            print_success(f"Ollama version: {version_output}")
        else:
            print_warning("Could not determine Ollama version")
            
        # Check if Ollama server is running
        try:
            response = requests.get("http://localhost:11434/api/version", timeout=2)
            if response.status_code == 200:
                print_success(f"Ollama server is running (API version: {response.json().get('version', 'unknown')})")
            else:
                print_warning(f"Ollama server returned status code {response.status_code}")
        except requests.exceptions.RequestException:
            print_warning("Ollama server is not running or not responding")
            print("You may need to start it with: ollama serve")
    else:
        print_warning("Ollama binary not found in PATH")
        # Check if it's in the expected location from Dockerfile
        if os.path.exists("/usr/local/bin/ollama"):
            print_warning("Ollama binary found at /usr/local/bin/ollama but not in PATH")
        else:
            print_error("Ollama binary not found")

def test_vllm():
    """Test vLLM functionality"""
    print_header("Testing vLLM")
    
    try:
        import vllm
        print_success(f"vLLM imported successfully (version: {vllm.__version__})")
        
        # Check if CUDA is available for vLLM
        try:
            from vllm.utils import get_gpu_memory
            try:
                gpu_memory = get_gpu_memory()
                print_success(f"vLLM detected {len(gpu_memory)} GPUs with memory: {gpu_memory}")
            except Exception as e:
                print_warning(f"vLLM could not get GPU memory: {e}")
        except ImportError:
            print_warning("Could not import vllm.utils.get_gpu_memory")
            
    except ImportError as e:
        print_error(f"Failed to import vllm: {e}")

def test_model_download():
    """Test downloading a small model from Hugging Face"""
    print_header("Testing Model Download")
    
    try:
        from huggingface_hub import snapshot_download
        
        # Try to download a tiny model to test HF access
        print("Attempting to download a small test model...")
        model_id = "hf-internal-testing/tiny-random-gpt2"
        try:
            path = snapshot_download(repo_id=model_id, local_files_only=False)
            print_success(f"Successfully downloaded test model to {path}")
        except Exception as e:
            print_error(f"Failed to download test model: {e}")
            print_warning("Check your internet connection and HF_TOKEN if set")
    except ImportError as e:
        print_error(f"Failed to import huggingface_hub: {e}")

def test_jupyter():
    """Test JupyterLab functionality"""
    print_header("Testing JupyterLab")
    
    jupyter_path = run_command("which jupyter")
    if jupyter_path:
        print_success(f"JupyterLab binary found at {jupyter_path}")
        
        # Check JupyterLab version
        version_output = run_command("jupyter --version")
        if version_output:
            print_success("JupyterLab version information:")
            for line in version_output.split('\n'):
                if line.strip():
                    print(f"  {line.strip()}")
        else:
            print_warning("Could not determine JupyterLab version")
    else:
        print_error("JupyterLab binary not found in PATH")

def run_simple_inference_test():
    """Run a simple inference test with a tiny model if available"""
    print_header("Simple Inference Test")
    
    try:
        from transformers import pipeline
        
        print("Attempting to run a simple inference test...")
        try:
            # Use a tiny model for quick testing
            classifier = pipeline("sentiment-analysis", model="hf-internal-testing/tiny-random-bert")
            result = classifier("I love this container!")
            print_success(f"Inference test successful: {result}")
        except Exception as e:
            print_error(f"Inference test failed: {e}")
    except ImportError as e:
        print_error(f"Failed to import transformers: {e}")

def main():
    """Run all tests"""
    print(f"{COLORS['BOLD']}VisionLangAnnotate Docker Container Test{COLORS['END']}")
    print(f"Running tests at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all tests
    check_system()
    check_cuda()
    check_python_packages()
    test_llamacpp()
    test_ollama()
    test_vllm()
    test_model_download()
    test_jupyter()
    run_simple_inference_test()
    
    print(f"\n{COLORS['BOLD']}Test Summary:{COLORS['END']}")
    print("All tests completed. Check the output above for any warnings or errors.")
    print("If you see any errors, you may need to check the Docker container configuration.")

if __name__ == "__main__":
    main()