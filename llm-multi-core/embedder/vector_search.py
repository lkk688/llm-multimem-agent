#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Vector Search Utility

This script demonstrates how to use embeddings for vector search and retrieval.
It provides a simple in-memory vector database implementation for text and image search.
"""

import os
import sys
import numpy as np
import argparse
import json
import pickle
from typing import List, Dict, Any, Union, Optional, Tuple
from PIL import Image
import matplotlib.pyplot as plt

# Import from the current package
from llm_multi_core.embedder import (
    BaseEmbedder,
    create_text_embedder,
    create_image_embedder,
    create_multimodal_embedder
)


class VectorDatabase:
    """Simple in-memory vector database for demonstration purposes."""
    
    def __init__(self, embedder: BaseEmbedder):
        """Initialize the vector database.
        
        Args:
            embedder: The embedder to use for generating embeddings
        """
        self.embedder = embedder
        self.items = []
        self.embeddings = []
        self.metadata = []
    
    def add_item(self, item: Any, metadata: Dict[str, Any] = None):
        """Add a single item to the database.
        
        Args:
            item: The item to add (text, image, etc.)
            metadata: Optional metadata associated with the item
        """
        embedding = self.embedder.embed(item)
        self.items.append(item)
        self.embeddings.append(embedding)
        self.metadata.append(metadata or {})
    
    def add_items(self, items: List[Any], metadata_list: List[Dict[str, Any]] = None):
        """Add multiple items to the database.
        
        Args:
            items: List of items to add
            metadata_list: Optional list of metadata for each item
        """
        if metadata_list is None:
            metadata_list = [{} for _ in items]
        
        embeddings = self.embedder.embed_batch(items)
        
        self.items.extend(items)
        self.embeddings.extend(embeddings)
        self.metadata.extend(metadata_list)
    
    def search(self, query: Any, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for items similar to the query.
        
        Args:
            query: The query item (text, image, etc.)
            top_k: Number of results to return
        
        Returns:
            List of dictionaries with search results
        """
        if not self.embeddings:
            return []
        
        # Generate query embedding
        query_embedding = self.embedder.embed(query)
        
        # Calculate similarities
        similarities = []
        for idx, item_embedding in enumerate(self.embeddings):
            similarity = np.dot(query_embedding, item_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(item_embedding)
            )
            similarities.append((idx, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k results
        results = []
        for idx, similarity in similarities[:top_k]:
            result = {
                "item": self.items[idx],
                "similarity": float(similarity),
                "metadata": self.metadata[idx]
            }
            results.append(result)
        
        return results
    
    def save(self, filepath: str):
        """Save the vector database to a file.
        
        Args:
            filepath: Path to save the database
        """
        data = {
            "embedder_type": str(type(self.embedder)),
            "embedder_str": str(self.embedder),
            "items": self.items,
            "embeddings": [emb.tolist() for emb in self.embeddings],
            "metadata": self.metadata
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, filepath: str, embedder: BaseEmbedder) -> 'VectorDatabase':
        """Load a vector database from a file.
        
        Args:
            filepath: Path to the database file
            embedder: Embedder to use (should match the one used to create the database)
        
        Returns:
            Loaded VectorDatabase instance
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        db = cls(embedder)
        db.items = data["items"]
        db.embeddings = [np.array(emb) for emb in data["embeddings"]]
        db.metadata = data["metadata"]
        
        return db


def create_text_database(texts: List[str], metadata_list: List[Dict[str, Any]] = None, 
                        framework: str = "sentence-transformers", model: str = None) -> VectorDatabase:
    """Create a vector database for text search.
    
    Args:
        texts: List of texts to add to the database
        metadata_list: Optional list of metadata for each text
        framework: Text embedding framework to use
        model: Model name to use (if None, uses the default model)
    
    Returns:
        VectorDatabase instance
    """
    embedder = create_text_embedder(framework=framework, model=model)
    db = VectorDatabase(embedder)
    db.add_items(texts, metadata_list)
    return db


def create_image_database(image_paths: List[str], metadata_list: List[Dict[str, Any]] = None,
                         framework: str = "clip", model: str = None) -> VectorDatabase:
    """Create a vector database for image search.
    
    Args:
        image_paths: List of paths to images to add to the database
        metadata_list: Optional list of metadata for each image
        framework: Image embedding framework to use
        model: Model name to use (if None, uses the default model)
    
    Returns:
        VectorDatabase instance
    """
    embedder = create_image_embedder(framework=framework, model=model)
    
    # Load images
    images = []
    valid_paths = []
    valid_metadata = []
    
    for i, path in enumerate(image_paths):
        try:
            img = Image.open(path)
            images.append(img)
            valid_paths.append(path)
            
            if metadata_list:
                valid_metadata.append(metadata_list[i])
        except Exception as e:
            print(f"Error loading image {path}: {e}")
    
    if not valid_metadata and metadata_list:
        valid_metadata = None
    
    # Create database
    db = VectorDatabase(embedder)
    db.add_items(images, valid_metadata)
    
    # Replace image objects with paths in the database for serialization
    db.items = valid_paths
    
    return db


def create_multimodal_database(items: List[Any], metadata_list: List[Dict[str, Any]] = None,
                              text_framework: str = "sentence-transformers", 
                              image_framework: str = "clip") -> VectorDatabase:
    """Create a vector database for multimodal search.
    
    Args:
        items: List of items (texts or image paths) to add to the database
        metadata_list: Optional list of metadata for each item
        text_framework: Text embedding framework to use
        image_framework: Image embedding framework to use
    
    Returns:
        VectorDatabase instance
    """
    embedder = create_multimodal_embedder(
        text_framework=text_framework,
        image_framework=image_framework
    )
    
    # Process items (load images if needed)
    processed_items = []
    for item in items:
        if isinstance(item, str) and os.path.isfile(item) and item.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
            try:
                processed_items.append(Image.open(item))
            except Exception as e:
                print(f"Error loading image {item}: {e}")
                processed_items.append(item)  # Fall back to treating as text
        else:
            processed_items.append(item)
    
    # Create database
    db = VectorDatabase(embedder)
    db.add_items(processed_items, metadata_list)
    
    # Replace image objects with original items for serialization
    db.items = items
    
    return db


def demo_text_search():
    """Demonstrate text search using vector database."""
    print("\n" + "=" * 50)
    print("TEXT SEARCH DEMO")
    print("=" * 50)
    
    # Sample texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast auburn fox leaps above the sleepy canine.",
        "The sky is blue and the sun is shining.",
        "Clouds cover the azure sky, blocking the bright sun.",
        "Artificial intelligence is transforming the world.",
        "Machine learning algorithms are changing how we solve problems.",
        "The cat sat on the windowsill watching birds.",
        "A feline perched on the ledge observing the avian creatures.",
        "Python is a popular programming language for data science.",
        "JavaScript is widely used for web development."
    ]
    
    # Metadata for each text
    metadata = [
        {"category": "animals", "length": len(texts[0])},
        {"category": "animals", "length": len(texts[1])},
        {"category": "nature", "length": len(texts[2])},
        {"category": "nature", "length": len(texts[3])},
        {"category": "technology", "length": len(texts[4])},
        {"category": "technology", "length": len(texts[5])},
        {"category": "animals", "length": len(texts[6])},
        {"category": "animals", "length": len(texts[7])},
        {"category": "technology", "length": len(texts[8])},
        {"category": "technology", "length": len(texts[9])}
    ]
    
    try:
        # Create text database
        print("Creating text vector database...")
        db = create_text_database(texts, metadata)
        
        # Sample queries
        queries = [
            "Fox jumping over dog",
            "Blue sky with sun",
            "AI and machine learning",
            "Cat watching birds",
            "Programming languages"
        ]
        
        # Search for each query
        for i, query in enumerate(queries):
            print(f"\nQuery {i+1}: \"{query}\"")
            results = db.search(query, top_k=3)
            
            print("Results:")
            for j, result in enumerate(results):
                print(f"  {j+1}. \"{result['item']}\" (Similarity: {result['similarity']:.4f}, Category: {result['metadata']['category']})")
        
        # Save and load database
        db_path = "text_vector_db.pkl"
        print(f"\nSaving database to {db_path}...")
        db.save(db_path)
        
        print(f"Loading database from {db_path}...")
        loaded_db = VectorDatabase.load(db_path, db.embedder)
        
        # Verify loaded database
        print("Verifying loaded database...")
        query = "Programming languages"
        results = loaded_db.search(query, top_k=3)
        
        print(f"\nQuery: \"{query}\"")
        print("Results from loaded database:")
        for j, result in enumerate(results):
            print(f"  {j+1}. \"{result['item']}\" (Similarity: {result['similarity']:.4f}, Category: {result['metadata']['category']})")
        
    except Exception as e:
        print(f"Error in text search demo: {e}")
    
    print("=" * 50 + "\n")


def demo_image_search():
    """Demonstrate image search using vector database."""
    print("\n" + "=" * 50)
    print("IMAGE SEARCH DEMO")
    print("=" * 50)
    
    # Create sample images if they don't exist
    os.makedirs("sample_images", exist_ok=True)
    
    # Sample image categories and colors
    categories = ["red", "green", "blue", "yellow", "purple", "cyan", "orange", "pink", "brown", "gray"]
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (128, 0, 128),  # Purple
        (0, 255, 255),  # Cyan
        (255, 165, 0),  # Orange
        (255, 192, 203),# Pink
        (165, 42, 42),  # Brown
        (128, 128, 128) # Gray
    ]
    
    # Create sample images
    image_paths = []
    metadata = []
    
    for i, (category, color) in enumerate(zip(categories, colors)):
        img_path = f"sample_images/{category}.jpg"
        
        # Create a simple colored image
        img = Image.new('RGB', (100, 100), color)
        img.save(img_path)
        
        image_paths.append(img_path)
        metadata.append({"category": category, "color": color})
    
    try:
        # Create image database
        print("Creating image vector database...")
        db = create_image_database(image_paths, metadata)
        
        # Sample queries (using some of the same images)
        query_paths = [
            "sample_images/red.jpg",
            "sample_images/blue.jpg",
            "sample_images/yellow.jpg",
            "sample_images/purple.jpg",
            "sample_images/gray.jpg"
        ]
        
        # Search for each query
        for i, query_path in enumerate(query_paths):
            print(f"\nQuery {i+1}: {query_path}")
            query_img = Image.open(query_path)
            results = db.search(query_img, top_k=3)
            
            print("Results:")
            for j, result in enumerate(results):
                print(f"  {j+1}. {result['item']} (Similarity: {result['similarity']:.4f}, Category: {result['metadata']['category']})")
        
        # Save and load database
        db_path = "image_vector_db.pkl"
        print(f"\nSaving database to {db_path}...")
        db.save(db_path)
        
        print(f"Loading database from {db_path}...")
        loaded_db = VectorDatabase.load(db_path, db.embedder)
        
        # Verify loaded database
        print("Verifying loaded database...")
        query_path = "sample_images/green.jpg"
        query_img = Image.open(query_path)
        results = loaded_db.search(query_img, top_k=3)
        
        print(f"\nQuery: {query_path}")
        print("Results from loaded database:")
        for j, result in enumerate(results):
            print(f"  {j+1}. {result['item']} (Similarity: {result['similarity']:.4f}, Category: {result['metadata']['category']})")
        
    except Exception as e:
        print(f"Error in image search demo: {e}")
    
    print("=" * 50 + "\n")


def demo_multimodal_search():
    """Demonstrate multimodal search using vector database."""
    print("\n" + "=" * 50)
    print("MULTIMODAL SEARCH DEMO")
    print("=" * 50)
    
    # Sample texts
    texts = [
        "A red square",
        "A green rectangle",
        "A blue circle",
        "A yellow triangle",
        "A purple star"
    ]
    
    # Sample image paths (reuse from image demo)
    image_paths = [
        "sample_images/red.jpg",
        "sample_images/green.jpg",
        "sample_images/blue.jpg",
        "sample_images/yellow.jpg",
        "sample_images/purple.jpg"
    ]
    
    # Combined items and metadata
    items = []
    metadata = []
    
    for text, image_path in zip(texts, image_paths):
        items.append(text)
        metadata.append({"type": "text", "content": text})
        
        items.append(image_path)
        metadata.append({"type": "image", "path": image_path})
    
    try:
        # Create multimodal database
        print("Creating multimodal vector database...")
        db = create_multimodal_database(items, metadata)
        
        # Sample queries (mix of text and images)
        queries = [
            "A red object",  # Text query
            "sample_images/blue.jpg",  # Image query
            "A yellow shape",  # Text query
            "sample_images/purple.jpg"  # Image query
        ]
        
        # Search for each query
        for i, query in enumerate(queries):
            print(f"\nQuery {i+1}: {query}")
            
            # Process query (load image if needed)
            if os.path.isfile(query) and query.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
                query_processed = Image.open(query)
                query_type = "image"
            else:
                query_processed = query
                query_type = "text"
            
            results = db.search(query_processed, top_k=5)
            
            print(f"Results for {query_type} query:")
            for j, result in enumerate(results):
                item_type = result['metadata']['type']
                item_display = result['item'] if item_type == "image" else f"\"{result['item']}\""
                print(f"  {j+1}. [{item_type}] {item_display} (Similarity: {result['similarity']:.4f})")
        
    except Exception as e:
        print(f"Error in multimodal search demo: {e}")
    
    print("=" * 50 + "\n")


def main():
    """Main function to run the vector search demo."""
    parser = argparse.ArgumentParser(description="Vector search demo")
    parser.add_argument("--text", action="store_true", help="Run text search demo")
    parser.add_argument("--image", action="store_true", help="Run image search demo")
    parser.add_argument("--multimodal", action="store_true", help="Run multimodal search demo")
    
    args = parser.parse_args()
    
    # If no specific demo is requested, run all demos
    if not (args.text or args.image or args.multimodal):
        args.text = True
        args.image = True
        args.multimodal = True
    
    print("\n" + "=" * 50)
    print("VECTOR SEARCH DEMO")
    print("=" * 50)
    
    if args.text:
        demo_text_search()
    
    if args.image:
        demo_image_search()
    
    if args.multimodal:
        demo_multimodal_search()
    
    print("Vector search demo completed!")


if __name__ == "__main__":
    main()