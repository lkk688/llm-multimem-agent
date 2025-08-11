#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Embedder Benchmark Tool

This script benchmarks different embedding frameworks for performance comparison,
measuring embedding generation time, memory usage, and embedding quality.
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Dict, Any, Tuple

# Import from the current package
from llm_multi_core.embedder import (
    get_available_embedders,
    create_text_embedder,
    create_image_embedder
)


def measure_memory_usage():
    """Measure current memory usage of the process."""
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # in MB


def benchmark_text_embedders(texts: List[str], batch_sizes: List[int] = [1, 10, 100]):
    """Benchmark text embedding frameworks.
    
    Args:
        texts: List of texts to embed
        batch_sizes: List of batch sizes to test
    
    Returns:
        Dictionary with benchmark results
    """
    available = get_available_embedders()["text"]
    available_frameworks = [fw for fw, avail in available.items() if avail]
    
    if not available_frameworks:
        print("No text embedding frameworks available. Please install at least one.")
        return {}
    
    results = {}
    
    for framework in available_frameworks:
        print(f"\nBenchmarking {framework}...")
        framework_results = {}
        
        try:
            # Create embedder
            start_time = time.time()
            start_memory = measure_memory_usage()
            
            embedder = create_text_embedder(framework=framework)
            
            init_time = time.time() - start_time
            init_memory = measure_memory_usage() - start_memory
            
            framework_results["init_time"] = init_time
            framework_results["init_memory"] = init_memory
            framework_results["embedding_dim"] = embedder.get_embedding_dim()
            
            print(f"  Initialization: {init_time:.2f}s, {init_memory:.2f}MB")
            print(f"  Embedding dimension: {embedder.get_embedding_dim()}")
            
            # Benchmark different batch sizes
            batch_results = {}
            for batch_size in batch_sizes:
                if batch_size > len(texts):
                    continue
                    
                batch_texts = texts[:batch_size]
                
                # Warm-up
                _ = embedder.embed_batch(batch_texts)
                
                # Benchmark
                start_time = time.time()
                start_memory = measure_memory_usage()
                
                embeddings = embedder.embed_batch(batch_texts)
                
                batch_time = time.time() - start_time
                batch_memory = measure_memory_usage() - start_memory
                
                # Calculate time per item
                time_per_item = batch_time / batch_size
                
                batch_results[batch_size] = {
                    "total_time": batch_time,
                    "time_per_item": time_per_item,
                    "memory_usage": batch_memory
                }
                
                print(f"  Batch size {batch_size}: {batch_time:.4f}s total, {time_per_item:.4f}s per item, {batch_memory:.2f}MB")
            
            framework_results["batch_results"] = batch_results
            results[framework] = framework_results
            
        except Exception as e:
            print(f"  Error benchmarking {framework}: {e}")
            results[framework] = {"error": str(e)}
    
    return results


def benchmark_image_embedders(image_paths: List[str], batch_sizes: List[int] = [1, 5, 10]):
    """Benchmark image embedding frameworks.
    
    Args:
        image_paths: List of paths to images to embed
        batch_sizes: List of batch sizes to test
    
    Returns:
        Dictionary with benchmark results
    """
    available = get_available_embedders()["image"]
    available_frameworks = [fw for fw, avail in available.items() if avail]
    
    if not available_frameworks:
        print("No image embedding frameworks available. Please install at least one.")
        return {}
    
    # Load images
    images = []
    for path in image_paths:
        try:
            img = Image.open(path)
            images.append(img)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
    
    if not images:
        print("No valid images to benchmark.")
        return {}
    
    results = {}
    
    for framework in available_frameworks:
        print(f"\nBenchmarking {framework}...")
        framework_results = {}
        
        try:
            # Create embedder
            start_time = time.time()
            start_memory = measure_memory_usage()
            
            embedder = create_image_embedder(framework=framework)
            
            init_time = time.time() - start_time
            init_memory = measure_memory_usage() - start_memory
            
            framework_results["init_time"] = init_time
            framework_results["init_memory"] = init_memory
            framework_results["embedding_dim"] = embedder.get_embedding_dim()
            
            print(f"  Initialization: {init_time:.2f}s, {init_memory:.2f}MB")
            print(f"  Embedding dimension: {embedder.get_embedding_dim()}")
            
            # Benchmark different batch sizes
            batch_results = {}
            for batch_size in batch_sizes:
                if batch_size > len(images):
                    continue
                    
                batch_images = images[:batch_size]
                
                # Warm-up
                _ = embedder.embed_batch(batch_images)
                
                # Benchmark
                start_time = time.time()
                start_memory = measure_memory_usage()
                
                embeddings = embedder.embed_batch(batch_images)
                
                batch_time = time.time() - start_time
                batch_memory = measure_memory_usage() - start_memory
                
                # Calculate time per item
                time_per_item = batch_time / batch_size
                
                batch_results[batch_size] = {
                    "total_time": batch_time,
                    "time_per_item": time_per_item,
                    "memory_usage": batch_memory
                }
                
                print(f"  Batch size {batch_size}: {batch_time:.4f}s total, {time_per_item:.4f}s per item, {batch_memory:.2f}MB")
            
            framework_results["batch_results"] = batch_results
            results[framework] = framework_results
            
        except Exception as e:
            print(f"  Error benchmarking {framework}: {e}")
            results[framework] = {"error": str(e)}
    
    return results


def plot_benchmark_results(text_results: Dict[str, Any] = None, image_results: Dict[str, Any] = None):
    """Plot benchmark results.
    
    Args:
        text_results: Results from text embedding benchmark
        image_results: Results from image embedding benchmark
    """
    if text_results:
        plot_text_benchmark_results(text_results)
    
    if image_results:
        plot_image_benchmark_results(image_results)


def plot_text_benchmark_results(results: Dict[str, Any]):
    """Plot text embedding benchmark results.
    
    Args:
        results: Results from text embedding benchmark
    """
    # Filter out frameworks with errors
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    
    if not valid_results:
        print("No valid text benchmark results to plot.")
        return
    
    # Get batch sizes (assuming all frameworks have the same batch sizes)
    first_framework = next(iter(valid_results))
    batch_sizes = list(valid_results[first_framework]["batch_results"].keys())
    
    # Plot initialization time
    plt.figure(figsize=(12, 6))
    frameworks = list(valid_results.keys())
    init_times = [valid_results[fw]["init_time"] for fw in frameworks]
    
    plt.subplot(1, 2, 1)
    plt.bar(frameworks, init_times)
    plt.title('Initialization Time')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    
    # Plot initialization memory
    plt.subplot(1, 2, 2)
    init_memory = [valid_results[fw]["init_memory"] for fw in frameworks]
    plt.bar(frameworks, init_memory)
    plt.title('Initialization Memory Usage')
    plt.ylabel('Memory (MB)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('text_init_benchmark.png')
    print("\nText initialization benchmark saved as 'text_init_benchmark.png'")
    
    # Plot time per item for each batch size
    plt.figure(figsize=(12, 6))
    
    for i, batch_size in enumerate(batch_sizes):
        plt.subplot(1, len(batch_sizes), i+1)
        
        frameworks = []
        times = []
        
        for fw, result in valid_results.items():
            if batch_size in result["batch_results"]:
                frameworks.append(fw)
                times.append(result["batch_results"][batch_size]["time_per_item"])
        
        plt.bar(frameworks, times)
        plt.title(f'Batch Size {batch_size}')
        plt.ylabel('Time per Item (seconds)')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('text_batch_benchmark.png')
    print("Text batch benchmark saved as 'text_batch_benchmark.png'")


def plot_image_benchmark_results(results: Dict[str, Any]):
    """Plot image embedding benchmark results.
    
    Args:
        results: Results from image embedding benchmark
    """
    # Filter out frameworks with errors
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    
    if not valid_results:
        print("No valid image benchmark results to plot.")
        return
    
    # Get batch sizes (assuming all frameworks have the same batch sizes)
    first_framework = next(iter(valid_results))
    batch_sizes = list(valid_results[first_framework]["batch_results"].keys())
    
    # Plot initialization time
    plt.figure(figsize=(12, 6))
    frameworks = list(valid_results.keys())
    init_times = [valid_results[fw]["init_time"] for fw in frameworks]
    
    plt.subplot(1, 2, 1)
    plt.bar(frameworks, init_times)
    plt.title('Initialization Time')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    
    # Plot initialization memory
    plt.subplot(1, 2, 2)
    init_memory = [valid_results[fw]["init_memory"] for fw in frameworks]
    plt.bar(frameworks, init_memory)
    plt.title('Initialization Memory Usage')
    plt.ylabel('Memory (MB)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('image_init_benchmark.png')
    print("\nImage initialization benchmark saved as 'image_init_benchmark.png'")
    
    # Plot time per item for each batch size
    plt.figure(figsize=(12, 6))
    
    for i, batch_size in enumerate(batch_sizes):
        plt.subplot(1, len(batch_sizes), i+1)
        
        frameworks = []
        times = []
        
        for fw, result in valid_results.items():
            if batch_size in result["batch_results"]:
                frameworks.append(fw)
                times.append(result["batch_results"][batch_size]["time_per_item"])
        
        plt.bar(frameworks, times)
        plt.title(f'Batch Size {batch_size}')
        plt.ylabel('Time per Item (seconds)')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('image_batch_benchmark.png')
    print("Image batch benchmark saved as 'image_batch_benchmark.png'")


def generate_sample_data(num_texts: int = 100, num_images: int = 20) -> Tuple[List[str], List[str]]:
    """Generate sample data for benchmarking.
    
    Args:
        num_texts: Number of sample texts to generate
        num_images: Number of sample images to generate
    
    Returns:
        Tuple of (text_list, image_paths)
    """
    # Generate sample texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Machine learning models can process natural language.",
        "Deep neural networks have revolutionized computer vision.",
        "Embedding vectors represent semantic meaning in a high-dimensional space.",
        "Transfer learning allows models to leverage knowledge from pre-training.",
        "Transformer architectures have become the foundation of modern NLP.",
        "Attention mechanisms help models focus on relevant parts of the input.",
        "Self-supervised learning reduces the need for labeled data.",
        "Multimodal models can process both text and images simultaneously."
    ]
    
    # Repeat texts to reach desired number
    text_list = []
    while len(text_list) < num_texts:
        text_list.extend(texts[:min(len(texts), num_texts - len(text_list))])
    
    # Generate sample images
    os.makedirs("benchmark_images", exist_ok=True)
    image_paths = []
    
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 0),    # Maroon
        (0, 128, 0),    # Green
        (0, 0, 128),    # Navy
        (128, 128, 0)   # Olive
    ]
    
    for i in range(num_images):
        color_idx = i % len(colors)
        img_path = f"benchmark_images/sample_{i}.jpg"
        
        # Create a simple colored image
        img = Image.new('RGB', (100, 100), colors[color_idx])
        img.save(img_path)
        
        image_paths.append(img_path)
    
    return text_list, image_paths


def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark embedding frameworks")
    parser.add_argument("--text", action="store_true", help="Benchmark text embedders")
    parser.add_argument("--image", action="store_true", help="Benchmark image embedders")
    parser.add_argument("--num-texts", type=int, default=100, help="Number of texts to benchmark")
    parser.add_argument("--num-images", type=int, default=20, help="Number of images to benchmark")
    parser.add_argument("--text-batch-sizes", type=int, nargs="+", default=[1, 10, 100], 
                        help="Batch sizes for text embedding")
    parser.add_argument("--image-batch-sizes", type=int, nargs="+", default=[1, 5, 10], 
                        help="Batch sizes for image embedding")
    
    args = parser.parse_args()
    
    # If neither --text nor --image is specified, benchmark both
    if not args.text and not args.image:
        args.text = True
        args.image = True
    
    print("\n" + "=" * 50)
    print("EMBEDDER BENCHMARK")
    print("=" * 50)
    
    # Generate sample data
    texts, image_paths = generate_sample_data(args.num_texts, args.num_images)
    
    text_results = None
    image_results = None
    
    # Benchmark text embedders
    if args.text:
        print("\n" + "=" * 50)
        print("TEXT EMBEDDER BENCHMARK")
        print("=" * 50)
        text_results = benchmark_text_embedders(texts, args.text_batch_sizes)
    
    # Benchmark image embedders
    if args.image:
        print("\n" + "=" * 50)
        print("IMAGE EMBEDDER BENCHMARK")
        print("=" * 50)
        image_results = benchmark_image_embedders(image_paths, args.image_batch_sizes)
    
    # Plot results
    plot_benchmark_results(text_results, image_results)
    
    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()