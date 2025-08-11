#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmark script for audio embedding using the AudioEmbedder class.

This script benchmarks different audio embedding frameworks in terms of:
1. Initialization time
2. Embedding generation time
3. Embedding dimension
4. Memory usage
"""

import os
import sys
import time
import argparse
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tabulate import tabulate
import psutil
import gc

# Add the parent directory to the path so we can import the embedder module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from llm_multi_core.embedder import (
    create_audio_embedder,
    get_available_audio_frameworks
)


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB


def benchmark_framework(framework, audio, num_runs=3):
    """Benchmark a single audio embedding framework."""
    results = {
        "framework": framework,
        "init_time": 0,
        "embed_time": 0,
        "dimension": 0,
        "memory": 0
    }
    
    # Force garbage collection before starting
    gc.collect()
    initial_memory = get_memory_usage()
    
    # Measure initialization time
    start_time = time.time()
    try:
        embedder = create_audio_embedder(framework=framework)
        init_time = time.time() - start_time
        results["init_time"] = init_time
        
        # Get embedding dimension
        results["dimension"] = embedder.get_embedding_dim()
        
        # Measure memory usage after initialization
        after_init_memory = get_memory_usage()
        results["memory"] = after_init_memory - initial_memory
        
        # Measure embedding time (average of num_runs)
        embed_times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = embedder.embed(audio)
            embed_times.append(time.time() - start_time)
        
        results["embed_time"] = sum(embed_times) / len(embed_times)
        
        # Success
        results["status"] = "Success"
    except Exception as e:
        results["status"] = f"Error: {str(e)}"
    
    # Force garbage collection after benchmark
    del embedder if 'embedder' in locals() else None
    gc.collect()
    
    return results


def plot_benchmark_results(results):
    """Plot benchmark results."""
    # Filter successful benchmarks
    successful_results = [r for r in results if r["status"] == "Success"]
    if not successful_results:
        print("No successful benchmarks to plot.")
        return
    
    frameworks = [r["framework"] for r in successful_results]
    init_times = [r["init_time"] for r in successful_results]
    embed_times = [r["embed_time"] for r in successful_results]
    dimensions = [r["dimension"] for r in successful_results]
    memories = [r["memory"] for r in successful_results]
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Initialization time
    axs[0, 0].bar(frameworks, init_times)
    axs[0, 0].set_title("Initialization Time (s)")
    axs[0, 0].set_ylabel("Time (s)")
    axs[0, 0].tick_params(axis='x', rotation=45)
    
    # Embedding time
    axs[0, 1].bar(frameworks, embed_times)
    axs[0, 1].set_title("Embedding Time (s)")
    axs[0, 1].set_ylabel("Time (s)")
    axs[0, 1].tick_params(axis='x', rotation=45)
    
    # Embedding dimension
    axs[1, 0].bar(frameworks, dimensions)
    axs[1, 0].set_title("Embedding Dimension")
    axs[1, 0].set_ylabel("Dimension")
    axs[1, 0].tick_params(axis='x', rotation=45)
    
    # Memory usage
    axs[1, 1].bar(frameworks, memories)
    axs[1, 1].set_title("Memory Usage (MB)")
    axs[1, 1].set_ylabel("Memory (MB)")
    axs[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Audio Embedder Benchmark")
    parser.add_argument(
        "--audio", 
        type=str, 
        default=None,
        help="Path to audio file for benchmarking"
    )
    parser.add_argument(
        "--frameworks",
        type=str,
        default="all",
        help="Comma-separated list of frameworks to benchmark, or 'all'"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs for embedding time measurement"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting of results"
    )
    args = parser.parse_args()
    
    # Get available frameworks
    available_frameworks = get_available_audio_frameworks()
    available_frameworks = {k: v for k, v in available_frameworks.items() if v}
    
    if not available_frameworks:
        print("No audio embedding frameworks available. Please install the required dependencies.")
        return
    
    print("Available Audio Embedding Frameworks:")
    for framework in available_frameworks:
        print(f"  - {framework}")
    
    # Determine which frameworks to benchmark
    if args.frameworks.lower() == "all":
        frameworks_to_benchmark = list(available_frameworks.keys())
    else:
        frameworks_to_benchmark = [f.strip() for f in args.frameworks.split(",")]
        # Filter out unavailable frameworks
        frameworks_to_benchmark = [
            f for f in frameworks_to_benchmark 
            if f in available_frameworks and available_frameworks[f]
        ]
    
    if not frameworks_to_benchmark:
        print("No valid frameworks specified for benchmarking.")
        return
    
    print(f"\nBenchmarking frameworks: {', '.join(frameworks_to_benchmark)}")
    
    # Load audio file or use a test tone
    if args.audio and os.path.exists(args.audio):
        print(f"Loading audio file: {args.audio}")
        audio, sr = librosa.load(args.audio, sr=16000)
    else:
        print("No audio file provided or file not found. Using a test tone.")
        sr = 16000
        duration = 5  # seconds
        audio = librosa.tone(440, sr=sr, duration=duration)  # 440 Hz tone
    
    print(f"Audio duration: {len(audio) / sr:.2f} seconds")
    print(f"Running {args.runs} embedding passes for each framework...\n")
    
    # Run benchmarks
    results = []
    for framework in frameworks_to_benchmark:
        print(f"Benchmarking {framework}...")
        result = benchmark_framework(framework, audio, num_runs=args.runs)
        results.append(result)
        print(f"  Status: {result['status']}")
        if result['status'] == 'Success':
            print(f"  Init time: {result['init_time']:.4f} s")
            print(f"  Embed time: {result['embed_time']:.4f} s")
            print(f"  Dimension: {result['dimension']}")
            print(f"  Memory: {result['memory']:.2f} MB")
        print()
    
    # Display results in a table
    table_data = [
        [
            r["framework"],
            f"{r['init_time']:.4f} s" if r['status'] == 'Success' else "N/A",
            f"{r['embed_time']:.4f} s" if r['status'] == 'Success' else "N/A",
            r["dimension"] if r['status'] == 'Success' else "N/A",
            f"{r['memory']:.2f} MB" if r['status'] == 'Success' else "N/A",
            r["status"]
        ]
        for r in results
    ]
    
    headers = ["Framework", "Init Time", "Embed Time", "Dimension", "Memory", "Status"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Plot results if enabled
    if not args.no_plot:
        plot_benchmark_results(results)


if __name__ == "__main__":
    main()