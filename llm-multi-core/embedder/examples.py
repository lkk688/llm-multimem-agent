# Examples of using the embedder module

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import embedder module
from . import (
    create_text_embedder,
    create_image_embedder,
    create_multimodal_embedder,
    get_available_embedders
)

def print_available_frameworks():
    """
    Print available embedding frameworks for each modality
    """
    available = get_available_embedders()
    
    print("\nAvailable Embedding Frameworks:")
    print("============================")
    
    # Text frameworks
    print("\nText Embedding Frameworks:")
    for framework, available in available["text"].items():
        status = "✅ Available" if available else "❌ Not available"
        print(f"  - {framework}: {status}")
    
    # Image frameworks
    print("\nImage Embedding Frameworks:")
    for framework, available in available["image"].items():
        status = "✅ Available" if available else "❌ Not available"
        print(f"  - {framework}: {status}")
    
    # Get first available framework for each modality
    text_framework = next((fw for fw, avail in available["text"].items() if avail), None)
    image_framework = next((fw for fw, avail in available["image"].items() if avail), None)
    
    return text_framework, image_framework

def text_embedding_example(framework=None):
    """
    Example of using text embedder
    """
    print("\nText Embedding Example:")
    print("======================")
    
    # Use first available framework if none specified
    if framework is None:
        available = get_available_embedders()["text"]
        framework = next((fw for fw, avail in available.items() if avail), None)
        if framework is None:
            print("No text embedding frameworks available. Please install one of the supported packages.")
            return
    
    print(f"Using framework: {framework}")
    
    try:
        # Create text embedder
        embedder = create_text_embedder(framework=framework)
        
        # Example texts
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "A fast auburn canine leaps above the sleepy hound.",
            "The sky is blue and the sun is shining.",
            "Artificial intelligence is transforming the world."
        ]
        
        # Generate embeddings
        embeddings = embedder.embed_batch(texts)
        
        print(f"Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
        
        # Calculate similarity matrix
        similarity_matrix = np.zeros((len(texts), len(texts)))
        for i in range(len(texts)):
            for j in range(len(texts)):
                similarity_matrix[i, j] = embedder.similarity(texts[i], texts[j])
        
        # Print similarity matrix
        print("\nSimilarity Matrix:")
        for i in range(len(texts)):
            for j in range(len(texts)):
                print(f"{similarity_matrix[i, j]:.4f}", end="  ")
            print()
        
        # Plot similarity matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(similarity_matrix, cmap="viridis")
        plt.colorbar()
        plt.title(f"Text Similarity Matrix ({framework})")
        plt.xticks(range(len(texts)), [f"Text {i+1}" for i in range(len(texts))], rotation=45)
        plt.yticks(range(len(texts)), [f"Text {i+1}" for i in range(len(texts))])
        plt.tight_layout()
        plt.savefig(f"text_similarity_{framework}.png")
        print(f"\nSimilarity matrix saved to text_similarity_{framework}.png")
        
    except Exception as e:
        print(f"Error in text embedding example: {str(e)}")

def image_embedding_example(framework=None):
    """
    Example of using image embedder
    """
    print("\nImage Embedding Example:")
    print("======================")
    
    # Use first available framework if none specified
    if framework is None:
        available = get_available_embedders()["image"]
        framework = next((fw for fw, avail in available.items() if avail), None)
        if framework is None:
            print("No image embedding frameworks available. Please install one of the supported packages.")
            return
    
    print(f"Using framework: {framework}")
    
    try:
        # Create image embedder
        embedder = create_image_embedder(framework=framework)
        
        # Example images (create simple test images)
        images = []
        for i in range(4):
            # Create a simple gradient image
            img = Image.new("RGB", (100, 100))
            pixels = img.load()
            for x in range(img.width):
                for y in range(img.height):
                    r = int(255 * (x / img.width))
                    g = int(255 * (y / img.height))
                    b = int(255 * ((i % 2) * 0.5))
                    pixels[x, y] = (r, g, b)
            images.append(img)
            img.save(f"test_image_{i+1}.png")
            print(f"Created test image: test_image_{i+1}.png")
        
        # Generate embeddings
        embeddings = embedder.embed_batch(images)
        
        print(f"Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
        
        # Calculate similarity matrix
        similarity_matrix = np.zeros((len(images), len(images)))
        for i in range(len(images)):
            for j in range(len(images)):
                similarity_matrix[i, j] = embedder.similarity(images[i], images[j])
        
        # Print similarity matrix
        print("\nSimilarity Matrix:")
        for i in range(len(images)):
            for j in range(len(images)):
                print(f"{similarity_matrix[i, j]:.4f}", end="  ")
            print()
        
        # Plot similarity matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(similarity_matrix, cmap="viridis")
        plt.colorbar()
        plt.title(f"Image Similarity Matrix ({framework})")
        plt.xticks(range(len(images)), [f"Image {i+1}" for i in range(len(images))], rotation=45)
        plt.yticks(range(len(images)), [f"Image {i+1}" for i in range(len(images))])
        plt.tight_layout()
        plt.savefig(f"image_similarity_{framework}.png")
        print(f"\nSimilarity matrix saved to image_similarity_{framework}.png")
        
    except Exception as e:
        print(f"Error in image embedding example: {str(e)}")

def multimodal_embedding_example(text_framework=None, image_framework=None):
    """
    Example of using multimodal embedder
    """
    print("\nMultimodal Embedding Example:")
    print("============================")
    
    # Use first available frameworks if none specified
    available = get_available_embedders()
    if text_framework is None:
        text_framework = next((fw for fw, avail in available["text"].items() if avail), None)
    if image_framework is None:
        image_framework = next((fw for fw, avail in available["image"].items() if avail), None)
    
    if text_framework is None or image_framework is None:
        print("Missing required frameworks for multimodal embedding.")
        return
    
    print(f"Using text framework: {text_framework}")
    print(f"Using image framework: {image_framework}")
    
    try:
        # Create multimodal embedder
        embedder = create_multimodal_embedder(
            text_framework=text_framework,
            image_framework=image_framework
        )
        
        # Example texts
        texts = [
            "A beautiful sunset over the ocean",
            "A cute puppy playing in the grass"
        ]
        
        # Example images (create simple test images)
        images = []
        for i in range(2):
            # Create a simple gradient image
            img = Image.new("RGB", (100, 100))
            pixels = img.load()
            for x in range(img.width):
                for y in range(img.height):
                    r = int(255 * (x / img.width))
                    g = int(255 * (y / img.height))
                    b = int(255 * (i * 0.5))
                    pixels[x, y] = (r, g, b)
            images.append(img)
            img.save(f"multimodal_image_{i+1}.png")
            print(f"Created test image: multimodal_image_{i+1}.png")
        
        # Combine inputs
        inputs = texts + images
        input_types = ["text"] * len(texts) + ["image"] * len(images)
        
        # Generate embeddings
        embeddings = embedder.embed_batch(inputs)
        
        print(f"Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
        
        # Calculate similarity matrix
        similarity_matrix = np.zeros((len(inputs), len(inputs)))
        for i in range(len(inputs)):
            for j in range(len(inputs)):
                similarity_matrix[i, j] = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
        
        # Print similarity matrix
        print("\nSimilarity Matrix:")
        for i in range(len(inputs)):
            for j in range(len(inputs)):
                print(f"{similarity_matrix[i, j]:.4f}", end="  ")
            print()
        
        # Plot similarity matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(similarity_matrix, cmap="viridis")
        plt.colorbar()
        plt.title(f"Multimodal Similarity Matrix ({text_framework}+{image_framework})")
        labels = [f"Text {i+1}" for i in range(len(texts))] + [f"Image {i+1}" for i in range(len(images))]
        plt.xticks(range(len(inputs)), labels, rotation=45)
        plt.yticks(range(len(inputs)), labels)
        plt.tight_layout()
        plt.savefig(f"multimodal_similarity_{text_framework}_{image_framework}.png")
        print(f"\nSimilarity matrix saved to multimodal_similarity_{text_framework}_{image_framework}.png")
        
    except Exception as e:
        print(f"Error in multimodal embedding example: {str(e)}")

def run_examples():
    """
    Run all embedding examples
    """
    # Print available frameworks
    text_framework, image_framework = print_available_frameworks()
    
    # Run examples
    if text_framework:
        text_embedding_example(text_framework)
    
    if image_framework:
        image_embedding_example(image_framework)
    
    if text_framework and image_framework:
        multimodal_embedding_example(text_framework, image_framework)

if __name__ == "__main__":
    run_examples()