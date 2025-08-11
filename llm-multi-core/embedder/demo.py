#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Embedder Module Demo

This script demonstrates the usage of the embedder module with various frameworks
for text, image, and multimodal embedding.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Import from the current package
from llm_multi_core.embedder import (
    get_available_embedders,
    create_text_embedder,
    create_image_embedder,
    create_multimodal_embedder
)


def print_available_frameworks():
    """Print available embedding frameworks for each modality."""
    available = get_available_embedders()
    
    print("\n" + "=" * 50)
    print("AVAILABLE EMBEDDING FRAMEWORKS")
    print("=" * 50)
    
    print("\nText Embedding Frameworks:")
    for framework, available in available["text"].items():
        status = "✅ Available" if available else "❌ Not available"
        print(f"  - {framework}: {status}")
    
    print("\nImage Embedding Frameworks:")
    for framework, available in available["image"].items():
        status = "✅ Available" if available else "❌ Not available"
        print(f"  - {framework}: {status}")
    
    print("\nMultimodal Embedding Frameworks:")
    for framework, available in available["multimodal"].items():
        status = "✅ Available" if available else "❌ Not available"
        print(f"  - {framework}: {status}")
    
    print("=" * 50 + "\n")


def demo_text_embedding():
    """Demonstrate text embedding with available frameworks."""
    print("\n" + "=" * 50)
    print("TEXT EMBEDDING DEMO")
    print("=" * 50)
    
    available = get_available_embedders()["text"]
    available_frameworks = [fw for fw, avail in available.items() if avail]
    
    if not available_frameworks:
        print("No text embedding frameworks available. Please install at least one.")
        return
    
    # Use the first available framework
    framework = available_frameworks[0]
    print(f"Using framework: {framework}")
    
    try:
        # Create text embedder
        embedder = create_text_embedder(framework=framework)
        print(f"Created embedder: {embedder}")
        print(f"Embedding dimension: {embedder.get_embedding_dim()}")
        
        # Sample texts
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "A fast auburn fox leaps above the sleepy canine.",
            "The sky is blue and the sun is shining.",
            "Artificial intelligence is transforming the world."
        ]
        
        # Generate embeddings
        print("\nGenerating embeddings for sample texts...")
        embeddings = embedder.embed_batch(texts)
        
        # Calculate similarities
        print("\nCalculating similarities between texts:")
        for i in range(len(texts)):
            for j in range(i+1, len(texts)):
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                print(f"Similarity between text {i+1} and text {j+1}: {similarity:.4f}")
        
        # Visualize similarities with a heatmap
        similarity_matrix = np.zeros((len(texts), len(texts)))
        for i in range(len(texts)):
            for j in range(len(texts)):
                similarity_matrix[i, j] = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
        
        plt.figure(figsize=(10, 8))
        plt.imshow(similarity_matrix, cmap='viridis')
        plt.colorbar(label='Cosine Similarity')
        plt.title(f'Text Similarity Matrix using {framework}')
        plt.xticks(range(len(texts)), [f'Text {i+1}' for i in range(len(texts))], rotation=45)
        plt.yticks(range(len(texts)), [f'Text {i+1}' for i in range(len(texts))])
        for i in range(len(texts)):
            for j in range(len(texts)):
                plt.text(j, i, f'{similarity_matrix[i, j]:.2f}', 
                         ha='center', va='center', color='white' if similarity_matrix[i, j] < 0.7 else 'black')
        plt.tight_layout()
        plt.savefig('text_similarity_matrix.png')
        print("\nSimilarity matrix saved as 'text_similarity_matrix.png'")
        
    except Exception as e:
        print(f"Error in text embedding demo: {e}")
    
    print("=" * 50 + "\n")


def demo_image_embedding():
    """Demonstrate image embedding with available frameworks."""
    print("\n" + "=" * 50)
    print("IMAGE EMBEDDING DEMO")
    print("=" * 50)
    
    available = get_available_embedders()["image"]
    available_frameworks = [fw for fw, avail in available.items() if avail]
    
    if not available_frameworks:
        print("No image embedding frameworks available. Please install at least one.")
        return
    
    # Use the first available framework
    framework = available_frameworks[0]
    print(f"Using framework: {framework}")
    
    # Check if we have sample images
    sample_images = [
        "sample_images/cat.jpg",
        "sample_images/dog.jpg",
        "sample_images/car.jpg",
        "sample_images/house.jpg"
    ]
    
    # Create sample directory if it doesn't exist
    os.makedirs("sample_images", exist_ok=True)
    
    # Create simple colored images if real images don't exist
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for i, img_path in enumerate(sample_images):
        if not os.path.exists(img_path):
            img = Image.new('RGB', (100, 100), colors[i])
            img.save(img_path)
            print(f"Created sample image: {img_path}")
    
    try:
        # Create image embedder
        embedder = create_image_embedder(framework=framework)
        print(f"Created embedder: {embedder}")
        print(f"Embedding dimension: {embedder.get_embedding_dim()}")
        
        # Load images
        images = [Image.open(path) for path in sample_images]
        
        # Generate embeddings
        print("\nGenerating embeddings for sample images...")
        embeddings = embedder.embed_batch(images)
        
        # Calculate similarities
        print("\nCalculating similarities between images:")
        for i in range(len(images)):
            for j in range(i+1, len(images)):
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                print(f"Similarity between {os.path.basename(sample_images[i])} and {os.path.basename(sample_images[j])}: {similarity:.4f}")
        
        # Visualize similarities with a heatmap
        similarity_matrix = np.zeros((len(images), len(images)))
        for i in range(len(images)):
            for j in range(len(images)):
                similarity_matrix[i, j] = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
        
        plt.figure(figsize=(10, 8))
        plt.imshow(similarity_matrix, cmap='viridis')
        plt.colorbar(label='Cosine Similarity')
        plt.title(f'Image Similarity Matrix using {framework}')
        plt.xticks(range(len(images)), [os.path.basename(path) for path in sample_images], rotation=45)
        plt.yticks(range(len(images)), [os.path.basename(path) for path in sample_images])
        for i in range(len(images)):
            for j in range(len(images)):
                plt.text(j, i, f'{similarity_matrix[i, j]:.2f}', 
                         ha='center', va='center', color='white' if similarity_matrix[i, j] < 0.7 else 'black')
        plt.tight_layout()
        plt.savefig('image_similarity_matrix.png')
        print("\nSimilarity matrix saved as 'image_similarity_matrix.png'")
        
    except Exception as e:
        print(f"Error in image embedding demo: {e}")
    
    print("=" * 50 + "\n")


def demo_multimodal_embedding():
    """Demonstrate multimodal embedding with available frameworks."""
    print("\n" + "=" * 50)
    print("MULTIMODAL EMBEDDING DEMO")
    print("=" * 50)
    
    available_text = get_available_embedders()["text"]
    available_image = get_available_embedders()["image"]
    
    available_text_frameworks = [fw for fw, avail in available_text.items() if avail]
    available_image_frameworks = [fw for fw, avail in available_image.items() if avail]
    
    if not available_text_frameworks or not available_image_frameworks:
        print("Both text and image embedding frameworks are required for multimodal demo.")
        return
    
    # Use the first available frameworks
    text_framework = available_text_frameworks[0]
    image_framework = available_image_frameworks[0]
    print(f"Using text framework: {text_framework}")
    print(f"Using image framework: {image_framework}")
    
    # Sample data
    texts = [
        "A cat sitting on a windowsill",
        "A dog running in the park",
        "A red sports car",
        "A house with a garden"
    ]
    
    # Check if we have sample images
    sample_images = [
        "sample_images/cat.jpg",
        "sample_images/dog.jpg",
        "sample_images/car.jpg",
        "sample_images/house.jpg"
    ]
    
    try:
        # Create multimodal embedder
        embedder = create_multimodal_embedder(
            text_framework=text_framework,
            image_framework=image_framework
        )
        print(f"Created embedder: {embedder}")
        
        # Load images
        images = [Image.open(path) for path in sample_images]
        
        # Create mixed input list (alternating text and image)
        mixed_inputs = []
        for i in range(len(texts)):
            mixed_inputs.append(texts[i])  # Text
            mixed_inputs.append(images[i])  # Image
        
        # Generate embeddings
        print("\nGenerating embeddings for mixed inputs...")
        embeddings = embedder.embed_batch(mixed_inputs)
        
        # Calculate cross-modal similarities
        print("\nCalculating similarities between text and images:")
        for i, text in enumerate(texts):
            for j, img_path in enumerate(sample_images):
                # Text index is 2*i, image index is 2*j+1 in the mixed_inputs list
                text_idx = 2 * i
                img_idx = 2 * j + 1
                
                similarity = np.dot(embeddings[text_idx], embeddings[img_idx]) / (
                    np.linalg.norm(embeddings[text_idx]) * np.linalg.norm(embeddings[img_idx])
                )
                print(f"Similarity between '{text}' and {os.path.basename(img_path)}: {similarity:.4f}")
        
        # Visualize cross-modal similarities with a heatmap
        similarity_matrix = np.zeros((len(texts), len(images)))
        for i in range(len(texts)):
            for j in range(len(images)):
                text_idx = 2 * i
                img_idx = 2 * j + 1
                similarity_matrix[i, j] = np.dot(embeddings[text_idx], embeddings[img_idx]) / (
                    np.linalg.norm(embeddings[text_idx]) * np.linalg.norm(embeddings[img_idx])
                )
        
        plt.figure(figsize=(12, 8))
        plt.imshow(similarity_matrix, cmap='viridis')
        plt.colorbar(label='Cosine Similarity')
        plt.title(f'Text-Image Similarity Matrix using {text_framework}/{image_framework}')
        plt.xticks(range(len(images)), [os.path.basename(path) for path in sample_images], rotation=45)
        plt.yticks(range(len(texts)), texts)
        for i in range(len(texts)):
            for j in range(len(images)):
                plt.text(j, i, f'{similarity_matrix[i, j]:.2f}', 
                         ha='center', va='center', color='white' if similarity_matrix[i, j] < 0.7 else 'black')
        plt.tight_layout()
        plt.savefig('multimodal_similarity_matrix.png')
        print("\nSimilarity matrix saved as 'multimodal_similarity_matrix.png'")
        
    except Exception as e:
        print(f"Error in multimodal embedding demo: {e}")
    
    print("=" * 50 + "\n")


def main():
    """Main function to run the demo."""
    print("\n" + "=" * 50)
    print("EMBEDDER MODULE DEMO")
    print("=" * 50)
    
    print_available_frameworks()
    
    # Run demos
    demo_text_embedding()
    demo_image_embedding()
    demo_multimodal_embedding()
    
    print("\nDemo completed!")


if __name__ == "__main__":
    main()