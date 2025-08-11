#!/usr/bin/env python3
"""
Ollama Vision Model Example

This script demonstrates how to use Ollama with multimodal vision models
to analyze images and generate descriptions or answer questions about them.

Requires a multimodal model like llava or bakllava to be pulled in Ollama.
"""

import argparse
import base64
import os
import sys
from typing import List, Optional

from openai import OpenAI

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434/v1"
DUMMY_API_KEY = "ollama"  # Ollama doesn't need a real API key
DEFAULT_MODEL = "llava:latest"  # or "bakllava:latest"


def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
        
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def analyze_image(client: OpenAI, image_path: str, prompt: str, model: str) -> str:
    """Analyze an image using a multimodal model via Ollama"""
    try:
        # Encode the image to base64
        base64_image = encode_image_to_base64(image_path)
        
        # Create the messages with the image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ]
        
        # Call the model
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1000,
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error analyzing image: {str(e)}"


def find_sample_images() -> List[str]:
    """Find sample images in the project"""
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
    """Main function"""
    parser = argparse.ArgumentParser(description="Analyze images using Ollama vision models")
    parser.add_argument("--image", "-i", type=str, help="Path to the image file")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL, 
                        help=f"Ollama model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--prompt", "-p", type=str, 
                        default="Describe this image in detail. What objects do you see?",
                        help="Prompt to send with the image")
    args = parser.parse_args()
    
    # Set up the OpenAI client for Ollama
    client = OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key=DUMMY_API_KEY
    )
    
    # Check if the model is available
    try:
        # Simple test to see if Ollama is running
        client.models.list()
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Make sure Ollama is running with 'ollama serve'")
        return 1
    
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
    
    # Make sure the model is pulled
    print(f"Using model: {args.model}")
    print(f"If the model is not already pulled, run: ollama pull {args.model}")
    
    # Analyze the image
    print(f"\nAnalyzing image: {image_path}")
    print(f"Prompt: {args.prompt}")
    print("\nGenerating response...")
    
    result = analyze_image(client, image_path, args.prompt, args.model)
    
    print("\n=== Analysis Result ===")
    print(result)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())