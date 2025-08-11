#!/usr/bin/env python3
"""
Ollama Integration with VisionLangAnnotate

This script demonstrates how to integrate Ollama with the VisionLangAnnotate project,
allowing you to use local models for vision-language tasks.
"""

import argparse
import os
import sys
import torch
from typing import Dict, List, Optional, Any

# Add the project root to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Import VisionLangAnnotate modules
    from VisionLangAnnotateModels.VLM.region_captioning import RegionCaptioner
    from VisionLangAnnotateModels.VLM.vlm_classifier import VLMClassifier
    from VisionLangAnnotateModels.detectors.ultralyticsyolo import UltralyticsYOLO
    
    # Import the OpenAI utilities that we'll extend
    from VisionLangAnnotateModels.VLM.openai_utils import OpenAIVLMInterface
    
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Error importing VisionLangAnnotate modules: {e}")
    print("Make sure you're running this script from the project root directory.")
    IMPORTS_SUCCESSFUL = False

# Import OpenAI SDK for Ollama
from openai import OpenAI


class OllamaVLMInterface(OpenAIVLMInterface):
    """
    Extension of OpenAIVLMInterface that uses Ollama for local inference
    
    This class overrides the OpenAI implementation to use Ollama with local models
    while maintaining the same interface.
    """
    
    def __init__(self, model_name="llava:latest", base_url="http://localhost:11434/v1"):
        """
        Initialize the Ollama VLM interface
        
        Args:
            model_name: The name of the Ollama model to use (e.g., "llava:latest")
            base_url: The base URL for the Ollama API
        """
        # Initialize with dummy values for OpenAI
        super().__init__(api_key="ollama", model="gpt-4-vision-preview")
        
        # Override with Ollama-specific settings
        self.model = model_name
        self.client = OpenAI(
            base_url=base_url,
            api_key="ollama"  # Ollama doesn't need a real API key
        )
        
        print(f"Initialized Ollama VLM interface with model: {model_name}")
    
    def check_model_availability(self):
        """
        Check if the model is available in Ollama
        
        Returns:
            bool: True if the model is available, False otherwise
        """
        try:
            models = self.client.models.list()
            model_names = [model.id for model in models.data]
            if self.model in model_names:
                return True
            else:
                print(f"Model {self.model} not found in Ollama.")
                print(f"Available models: {', '.join(model_names)}")
                print(f"Pull the model with: ollama pull {self.model}")
                return False
        except Exception as e:
            print(f"Error checking model availability: {e}")
            return False


def setup_region_captioner(model_name="llava:latest"):
    """
    Set up a RegionCaptioner using Ollama
    
    Args:
        model_name: The name of the Ollama model to use
        
    Returns:
        RegionCaptioner: A configured RegionCaptioner instance
    """
    # Create the Ollama VLM interface
    ollama_vlm = OllamaVLMInterface(model_name=model_name)
    
    # Check if the model is available
    if not ollama_vlm.check_model_availability():
        print(f"Please pull the model first: ollama pull {model_name}")
        return None
    
    # Create a detector (YOLO)
    detector = UltralyticsYOLO(model_name="yolov8n")
    
    # Create the region captioner
    captioner = RegionCaptioner(
        detector_type="yolo",
        detector_model=detector.model_name,
        confidence_threshold=0.3
    )
    
    return captioner


def setup_vlm_classifier(model_name="llava:latest"):
    """
    Set up a VLMClassifier using Ollama
    
    Args:
        model_name: The name of the Ollama model to use
        
    Returns:
        VLMClassifier: A configured VLMClassifier instance
    """
    # Create the Ollama VLM interface
    ollama_vlm = OllamaVLMInterface(model_name=model_name)
    
    # Check if the model is available
    if not ollama_vlm.check_model_availability():
        print(f"Please pull the model first: ollama pull {model_name}")
        return None
    
    # Create the VLM classifier
    classifier = VLMClassifier(
        model_name="Salesforce/blip2-opt-2.7b",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    return classifier


def find_sample_images() -> List[str]:
    """
    Find sample images in the project
    
    Returns:
        List[str]: List of paths to sample images
    """
    sample_dirs = [
        "./VisionLangAnnotateModels/sampledata",
        "./sampledata",
        "./samples",
        "./images",
    ]
    
    sample_images = []
    for directory in sample_dirs:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    sample_images.append(os.path.join(directory, file))
    
    return sample_images


def main():
    """
    Main function
    """
    if not IMPORTS_SUCCESSFUL:
        return 1
    
    parser = argparse.ArgumentParser(description="Ollama integration with VisionLangAnnotate")
    parser.add_argument("--image", "-i", type=str, help="Path to the image file")
    parser.add_argument("--model", "-m", type=str, default="llava:latest", 
                        help="Ollama model to use (default: llava:latest)")
    parser.add_argument("--mode", type=str, choices=["caption", "classify"], default="caption",
                        help="Mode: 'caption' for region captioning, 'classify' for classification")
    args = parser.parse_args()
    
    # Find an image to analyze
    image_path = args.image
    if not image_path:
        # Try to find sample images
        sample_images = find_sample_images()
        if sample_images:
            image_path = sample_images[0]
            print(f"Using sample image: {image_path}")
        else:
            print("No sample image found. Please provide an image path with --image")
            return 1
    
    # Make sure the image exists
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return 1
    
    # Process based on the selected mode
    if args.mode == "caption":
        # Set up the region captioner
        captioner = setup_region_captioner(model_name=args.model)
        if captioner is None:
            return 1
        
        # Process the image
        print(f"\nGenerating region captions for: {image_path}")
        results = captioner.process_image(image_path)
        
        # Display results
        print("\n=== Region Captions ===")
        for i, region in enumerate(results):
            print(f"Region {i+1}: {region['caption']}")
            print(f"  Bounding box: {region['bbox']}")
            print(f"  Class: {region['class']}")
            print(f"  Confidence: {region['confidence']:.2f}")
            print()
    
    elif args.mode == "classify":
        # Set up the VLM classifier
        classifier = setup_vlm_classifier(model_name=args.model)
        if classifier is None:
            return 1
        
        # Process the image
        print(f"\nClassifying image: {image_path}")
        results = classifier.classify_image(image_path)
        
        # Display results
        print("\n=== Classification Results ===")
        for class_name, confidence in results.items():
            print(f"{class_name}: {confidence:.2f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())