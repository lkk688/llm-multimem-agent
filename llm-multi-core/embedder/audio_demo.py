#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demo script for audio embedding using the AudioEmbedder class.

This script demonstrates how to use the AudioEmbedder class to generate
embeddings for audio files using different frameworks.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Add the parent directory to the path so we can import the embedder module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from llm_multi_core.embedder import (
    create_audio_embedder,
    get_available_audio_frameworks
)


def plot_waveform(audio, sr, title="Waveform"):
    """Plot the waveform of an audio signal."""
    plt.figure(figsize=(10, 3))
    plt.title(title)
    librosa.display.waveshow(audio, sr=sr)
    plt.tight_layout()
    plt.show()


def plot_embedding(embedding, title="Audio Embedding"):
    """Plot the embedding vector."""
    plt.figure(figsize=(10, 3))
    plt.title(f"{title} (Dimension: {len(embedding)})")
    plt.plot(embedding)
    plt.tight_layout()
    plt.show()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Audio Embedder Demo")
    parser.add_argument(
        "--framework", 
        type=str, 
        default="wav2vec2",
        help="Audio embedding framework to use"
    )
    parser.add_argument(
        "--audio", 
        type=str, 
        default=None,
        help="Path to audio file"
    )
    parser.add_argument(
        "--list-frameworks", 
        action="store_true",
        help="List available audio embedding frameworks"
    )
    args = parser.parse_args()

    # List available frameworks if requested
    if args.list_frameworks:
        available_frameworks = get_available_audio_frameworks()
        print("Available Audio Embedding Frameworks:")
        for framework, available in available_frameworks.items():
            status = "Available" if available else "Not available"
            print(f"  - {framework}: {status}")
        return

    # Load audio file or use a test tone if none provided
    if args.audio:
        print(f"Loading audio file: {args.audio}")
        audio, sr = librosa.load(args.audio, sr=16000)
    else:
        print("No audio file provided. Using a test tone.")
        sr = 16000
        duration = 3  # seconds
        audio = librosa.tone(440, sr=sr, duration=duration)  # 440 Hz tone

    # Plot the waveform
    plot_waveform(audio, sr, title=f"Audio Waveform")

    # Create the audio embedder
    print(f"Creating audio embedder with framework: {args.framework}")
    try:
        embedder = create_audio_embedder(framework=args.framework)
    except Exception as e:
        print(f"Error creating embedder: {e}")
        return

    # Generate embedding
    print("Generating embedding...")
    try:
        embedding = embedder.embed(audio)
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding dimension: {embedder.get_embedding_dim()}")
        
        # Plot the embedding
        plot_embedding(embedding, title=f"{args.framework.capitalize()} Embedding")
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return

    # If we have two audio files, calculate similarity
    if args.audio and os.path.exists(args.audio):
        # Generate a slightly modified version of the audio for comparison
        modified_audio = audio + 0.05 * np.random.randn(len(audio))  # Add some noise
        
        # Generate embedding for modified audio
        modified_embedding = embedder.embed(modified_audio)
        
        # Calculate similarity
        similarity = np.dot(embedding, modified_embedding) / (
            np.linalg.norm(embedding) * np.linalg.norm(modified_embedding)
        )
        print(f"Similarity between original and modified audio: {similarity:.4f}")


if __name__ == "__main__":
    main()