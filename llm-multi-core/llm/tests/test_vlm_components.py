#!/usr/bin/env python3
"""
Vision Language Model Components Test Script

This script tests the VLM-specific components in the Docker container:
- llamacpp_utils.py functionality
- ollama_utils.py functionality
- Basic image processing capabilities
- Model loading and inference

Usage:
    python test_vlm_components.py
"""

import os
import sys
import json
import base64
import importlib.util
from pathlib import Path
import time
import io

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

def check_module_exists(module_path):
    """Check if a Python module file exists"""
    return os.path.exists(module_path)

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

def test_image_processing():
    """Test basic image processing capabilities"""
    print_header("Testing Image Processing")
    
    try:
        from PIL import Image
        print_success("PIL/Pillow is installed")
        
        # Create a simple test image
        img = Image.new('RGB', (224, 224), color='red')
        print_success("Created test image")
        
        # Test image resizing
        resized_img = img.resize((112, 112))
        print_success("Image resizing works")
        
        # Test image to base64
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        print_success("Image to base64 conversion works")
        
        return True
    except ImportError as e:
        print_error(f"PIL/Pillow import error: {e}")
        return False
    except Exception as e:
        print_error(f"Image processing error: {e}")
        return False

def test_llamacpp_utils():
    """Test llamacpp_utils.py functionality"""
    print_header("Testing llamacpp_utils.py")
    
    # Check if the module exists
    module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "VLM", "llamacpp_utils.py")
    if not check_module_exists(module_path):
        print_warning(f"llamacpp_utils.py not found at {module_path}")
        return False
    
    # Import the module
    llamacpp_utils = import_module_from_path("llamacpp_utils", module_path)
    if llamacpp_utils is None:
        return False
    
    print_success("Successfully imported llamacpp_utils.py")
    
    # Check for key components
    if hasattr(llamacpp_utils, "LlamaCppClient"):
        print_success("LlamaCppClient class found")
    else:
        print_error("LlamaCppClient class not found")
        return False
    
    # Check for utility functions
    required_functions = [
        "extract_llamacpp_response_text",
        "process_with_llamacpp_text",
        "process_with_llamacpp_vision",
        "process_with_llamacpp"
    ]
    
    for func in required_functions:
        if hasattr(llamacpp_utils, func):
            print_success(f"Function {func} found")
        else:
            print_error(f"Function {func} not found")
            return False
    
    # Check if the client can be initialized (without making actual API calls)
    try:
        client = llamacpp_utils.LlamaCppClient()
        print_success("LlamaCppClient initialized successfully")
        return True
    except Exception as e:
        print_error(f"Error initializing LlamaCppClient: {e}")
        return False

def test_ollama_utils():
    """Test ollama_utils.py functionality"""
    print_header("Testing ollama_utils.py")
    
    # Check if the module exists
    module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "VLM", "ollama_utils.py")
    if not check_module_exists(module_path):
        print_warning(f"ollama_utils.py not found at {module_path}")
        return False
    
    # Import the module
    ollama_utils = import_module_from_path("ollama_utils", module_path)
    if ollama_utils is None:
        return False
    
    print_success("Successfully imported ollama_utils.py")
    
    # Check for key components
    if hasattr(ollama_utils, "OllamaClient"):
        print_success("OllamaClient class found")
    else:
        print_error("OllamaClient class not found")
        return False
    
    # Check for utility functions
    required_functions = [
        "extract_ollama_response_text",
        "process_with_ollama_text",
        "process_with_ollama_vision",
        "process_with_ollama"
    ]
    
    for func in required_functions:
        if hasattr(ollama_utils, func):
            print_success(f"Function {func} found")
        else:
            print_error(f"Function {func} not found")
            return False
    
    # Check if the client can be initialized (without making actual API calls)
    try:
        client = ollama_utils.OllamaClient()
        print_success("OllamaClient initialized successfully")
        return True
    except Exception as e:
        print_error(f"Error initializing OllamaClient: {e}")
        return False

def test_openai_utils():
    """Test openai_utils.py functionality"""
    print_header("Testing openai_utils.py")
    
    # Check if the module exists
    module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "VLM", "openai_utils.py")
    if not check_module_exists(module_path):
        print_warning(f"openai_utils.py not found at {module_path}")
        return False
    
    # Import the module
    openai_utils = import_module_from_path("openai_utils", module_path)
    if openai_utils is None:
        return False
    
    print_success("Successfully imported openai_utils.py")
    
    # Check for key components and functions
    if hasattr(openai_utils, "process_with_openai"):
        print_success("process_with_openai function found")
    else:
        print_error("process_with_openai function not found")
        return False
    
    return True

def test_vlm_classifier():
    """Test vlm_classifier.py functionality"""
    print_header("Testing VLM Classifier")
    
    # Check if the module exists
    module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "VLM", "vlm_classifier.py")
    if not check_module_exists(module_path):
        print_warning(f"vlm_classifier.py not found at {module_path}")
        return False
    
    # Import the module
    vlm_classifier = import_module_from_path("vlm_classifier", module_path)
    if vlm_classifier is None:
        return False
    
    print_success("Successfully imported vlm_classifier.py")
    
    # Check for key components
    if hasattr(vlm_classifier, "VLMClassifier"):
        print_success("VLMClassifier class found")
    else:
        print_error("VLMClassifier class not found")
        return False
    
    return True

def test_region_captioning():
    """Test region_captioning.py functionality"""
    print_header("Testing Region Captioning")
    
    # Check if the module exists
    module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "VLM", "region_captioning.py")
    if not check_module_exists(module_path):
        print_warning(f"region_captioning.py not found at {module_path}")
        return False
    
    # Import the module
    region_captioning = import_module_from_path("region_captioning", module_path)
    if region_captioning is None:
        return False
    
    print_success("Successfully imported region_captioning.py")
    
    # Check for key components
    if hasattr(region_captioning, "RegionCaptioner"):
        print_success("RegionCaptioner class found")
    else:
        print_error("RegionCaptioner class not found")
        return False
    
    return True

def check_sample_data():
    """Check if sample data exists for testing"""
    print_header("Checking Sample Data")
    
    sample_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sampledata")
    if not os.path.exists(sample_data_dir):
        print_warning(f"Sample data directory not found at {sample_data_dir}")
        return False
    
    print_success(f"Sample data directory found at {sample_data_dir}")
    
    # Check for some sample images
    sample_images = [f for f in os.listdir(sample_data_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if sample_images:
        print_success(f"Found {len(sample_images)} sample images: {', '.join(sample_images)}")
        return True
    else:
        print_warning("No sample images found in the sample data directory")
        return False

def main():
    """Run all tests"""
    print(f"{COLORS['BOLD']}VisionLangAnnotate VLM Components Test{COLORS['END']}")
    print(f"Running tests at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all tests
    image_processing_ok = test_image_processing()
    llamacpp_utils_ok = test_llamacpp_utils()
    ollama_utils_ok = test_ollama_utils()
    openai_utils_ok = test_openai_utils()
    vlm_classifier_ok = test_vlm_classifier()
    region_captioning_ok = test_region_captioning()
    sample_data_ok = check_sample_data()
    
    # Print summary
    print(f"\n{COLORS['BOLD']}Test Summary:{COLORS['END']}")
    print(f"Image Processing: {'✓' if image_processing_ok else '✗'}")
    print(f"llamacpp_utils.py: {'✓' if llamacpp_utils_ok else '✗'}")
    print(f"ollama_utils.py: {'✓' if ollama_utils_ok else '✗'}")
    print(f"openai_utils.py: {'✓' if openai_utils_ok else '✗'}")
    print(f"vlm_classifier.py: {'✓' if vlm_classifier_ok else '✗'}")
    print(f"region_captioning.py: {'✓' if region_captioning_ok else '✗'}")
    print(f"Sample Data: {'✓' if sample_data_ok else '✗'}")
    
    print("\nAll tests completed. Check the output above for any warnings or errors.")
    print("Note: These tests only check for the presence of components, not their actual functionality.")
    print("To test actual inference, you need to have models downloaded and servers running.")

if __name__ == "__main__":
    main()