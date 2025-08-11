#!/usr/bin/env python3
"""
Vision Language Model Inference Test Script

This script tests the actual inference capabilities of the VLM components:
- Tests llamacpp_utils.py with a vision model if available
- Tests ollama_utils.py with a vision model if available
- Tests basic image captioning functionality

Usage:
    python test_vlm_inference.py
"""

import os
import sys
import json
import importlib.util
from pathlib import Path
import time
import io
import requests
import subprocess
import argparse

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

def import_module_from_path(module_name, module_path):
    """Import a module from a file path"""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        return None
    
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print_error(f"Error importing {module_name}: {e}")
        return None

def find_sample_image():
    """Find a sample image to use for testing"""
    # Check in the sample data directory
    sample_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sampledata")
    if os.path.exists(sample_data_dir):
        sample_images = [os.path.join(sample_data_dir, f) for f in os.listdir(sample_data_dir) 
                        if f.endswith(('.jpg', '.jpeg', '.png'))]
        if sample_images:
            return sample_images[0]
    
    # If no sample image found, create one
    try:
        from PIL import Image
        temp_img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_test_image.jpg")
        img = Image.new('RGB', (224, 224), color='red')
        img.save(temp_img_path)
        print_warning(f"Created temporary test image at {temp_img_path}")
        return temp_img_path
    except Exception as e:
        print_error(f"Could not create test image: {e}")
        return None

def check_llamacpp_server():
    """Check if llama.cpp server is running"""
    try:
        response = requests.get("http://localhost:8000/v1/models", timeout=2)
        if response.status_code == 200:
            models = response.json()
            print_success(f"llama.cpp server is running with {len(models['data'])} models")
            # Check if any of the models support vision
            vision_models = [m for m in models['data'] if 'vision' in m['id'].lower() or 'llava' in m['id'].lower()]
            if vision_models:
                print_success(f"Found vision-capable models: {', '.join([m['id'] for m in vision_models])}")
                return vision_models[0]['id']
            else:
                print_warning("No vision-capable models found in llama.cpp server")
                return None
        else:
            print_warning(f"llama.cpp server returned status code {response.status_code}")
            return None
    except requests.exceptions.RequestException:
        print_warning("llama.cpp server is not running or not responding")
        return None

def check_ollama_server():
    """Check if Ollama server is running and has vision models"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print_success(f"Ollama server is running with {len(models)} models")
            # Check if any of the models support vision
            vision_models = [m['name'] for m in models 
                            if any(v in m['name'].lower() for v in ['vision', 'llava', 'bakllava'])]
            if vision_models:
                print_success(f"Found vision-capable models: {', '.join(vision_models)}")
                return vision_models[0]
            else:
                print_warning("No vision-capable models found in Ollama server")
                return None
        else:
            print_warning(f"Ollama server returned status code {response.status_code}")
            return None
    except requests.exceptions.RequestException:
        print_warning("Ollama server is not running or not responding")
        return None

def test_llamacpp_inference(image_path):
    """Test llamacpp_utils.py with actual inference"""
    print_header("Testing LlamaCpp Inference")
    
    if not os.path.exists(image_path):
        print_error(f"Image file not found: {image_path}")
        return False
    
    # Check if llama.cpp server is running
    model_id = check_llamacpp_server()
    if not model_id:
        print_warning("Skipping LlamaCpp inference test as no server or vision models are available")
        return False
    
    # Import llamacpp_utils
    module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "VLM", "llamacpp_utils.py")
    if not os.path.exists(module_path):
        print_error(f"llamacpp_utils.py not found at {module_path}")
        return False
    
    llamacpp_utils = import_module_from_path("llamacpp_utils", module_path)
    if llamacpp_utils is None:
        return False
    
    # Test inference
    try:
        print(f"Running inference with model: {model_id}")
        client = llamacpp_utils.LlamaCppClient()
        response = client.process_vision(
            images=image_path,
            prompt="Describe this image briefly.",
            model=model_id
        )
        
        if response:
            print_success("LlamaCpp inference successful")
            print(f"Response: {response[:100]}..." if len(response) > 100 else f"Response: {response}")
            return True
        else:
            print_error("LlamaCpp inference failed - empty response")
            return False
    except Exception as e:
        print_error(f"LlamaCpp inference error: {e}")
        return False

def test_ollama_inference(image_path):
    """Test ollama_utils.py with actual inference"""
    print_header("Testing Ollama Inference")
    
    if not os.path.exists(image_path):
        print_error(f"Image file not found: {image_path}")
        return False
    
    # Check if Ollama server is running
    model_id = check_ollama_server()
    if not model_id:
        print_warning("Skipping Ollama inference test as no server or vision models are available")
        return False
    
    # Import ollama_utils
    module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "VLM", "ollama_utils.py")
    if not os.path.exists(module_path):
        print_error(f"ollama_utils.py not found at {module_path}")
        return False
    
    ollama_utils = import_module_from_path("ollama_utils", module_path)
    if ollama_utils is None:
        return False
    
    # Test inference
    try:
        print(f"Running inference with model: {model_id}")
        client = ollama_utils.OllamaClient()
        response = client.process_vision(
            images=image_path,
            prompt="Describe this image briefly.",
            model=model_id
        )
        
        if response:
            print_success("Ollama inference successful")
            print(f"Response: {response[:100]}..." if len(response) > 100 else f"Response: {response}")
            return True
        else:
            print_error("Ollama inference failed - empty response")
            return False
    except Exception as e:
        print_error(f"Ollama inference error: {e}")
        return False

def test_region_captioning(image_path):
    """Test region_captioning.py with actual inference"""
    print_header("Testing Region Captioning")
    
    if not os.path.exists(image_path):
        print_error(f"Image file not found: {image_path}")
        return False
    
    # Import region_captioning
    module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "VLM", "region_captioning.py")
    if not os.path.exists(module_path):
        print_error(f"region_captioning.py not found at {module_path}")
        return False
    
    region_captioning = import_module_from_path("region_captioning", module_path)
    if region_captioning is None:
        return False
    
    # Check if we have a model available
    ollama_model = check_ollama_server()
    llamacpp_model = check_llamacpp_server()
    
    if not (ollama_model or llamacpp_model):
        print_warning("Skipping region captioning test as no vision models are available")
        return False
    
    # Test inference
    try:
        # Choose which backend to use
        backend = "ollama" if ollama_model else "llamacpp"
        model = ollama_model if ollama_model else llamacpp_model
        
        print(f"Running region captioning with {backend} backend and model: {model}")
        
        # This is a simplified test - actual usage would involve detecting regions first
        # For testing, we'll just use the whole image as one region
        from PIL import Image
        img = Image.open(image_path)
        width, height = img.size
        
        # Create a dummy region (the whole image)
        regions = [{
            "bbox": [0, 0, width, height],
            "score": 0.99,
            "class": "test_object"
        }]
        
        # Initialize the captioner
        captioner = region_captioning.RegionCaptioner(
            backend=backend,
            model=model
        )
        
        # Get captions
        try:
            captions = captioner.get_captions(image_path, regions)
            if captions and len(captions) > 0:
                print_success("Region captioning successful")
                print(f"Caption: {captions[0][:100]}..." if len(captions[0]) > 100 else f"Caption: {captions[0]}")
                return True
            else:
                print_error("Region captioning failed - no captions generated")
                return False
        except Exception as e:
            print_error(f"Region captioning error during caption generation: {e}")
            return False
    except Exception as e:
        print_error(f"Region captioning error: {e}")
        return False

def main():
    """Run all tests"""
    parser = argparse.ArgumentParser(description="Test VLM inference capabilities")
    parser.add_argument("--image", help="Path to image file for testing")
    parser.add_argument("--skip-llamacpp", action="store_true", help="Skip LlamaCpp inference test")
    parser.add_argument("--skip-ollama", action="store_true", help="Skip Ollama inference test")
    parser.add_argument("--skip-region", action="store_true", help="Skip region captioning test")
    args = parser.parse_args()
    
    print(f"{COLORS['BOLD']}VisionLangAnnotate VLM Inference Test{COLORS['END']}")
    print(f"Running tests at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Find a sample image
    image_path = args.image if args.image else find_sample_image()
    if not image_path:
        print_error("No image available for testing. Please provide an image path with --image")
        return
    
    print_success(f"Using image: {image_path}")
    
    # Run inference tests
    llamacpp_ok = False if args.skip_llamacpp else test_llamacpp_inference(image_path)
    ollama_ok = False if args.skip_ollama else test_ollama_inference(image_path)
    region_ok = False if args.skip_region else test_region_captioning(image_path)
    
    # Print summary
    print(f"\n{COLORS['BOLD']}Test Summary:{COLORS['END']}")
    if not args.skip_llamacpp:
        print(f"LlamaCpp Inference: {'✓' if llamacpp_ok else '✗'}")
    if not args.skip_ollama:
        print(f"Ollama Inference: {'✓' if ollama_ok else '✗'}")
    if not args.skip_region:
        print(f"Region Captioning: {'✓' if region_ok else '✗'}")
    
    print("\nAll tests completed. Check the output above for any warnings or errors.")
    print("Note: These tests require running servers and available models to succeed.")
    print("If tests failed, ensure that the servers are running and vision models are available.")

if __name__ == "__main__":
    main()