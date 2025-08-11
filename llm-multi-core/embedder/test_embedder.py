#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for the embedder module.

This script contains unit tests for the text, image, and multimodal embedders.
It tests the basic functionality of each embedder type with available frameworks.
"""

import os
import sys
import unittest
import numpy as np
from PIL import Image
from typing import List, Dict, Any

# Import from the current package using relative imports
from . import (
    get_available_embedders,
    create_text_embedder,
    create_image_embedder,
    create_audio_embedder,
    create_multimodal_embedder
)


class TestTextEmbedder(unittest.TestCase):
    """Test cases for TextEmbedder."""
    
    def setUp(self):
        """Set up test environment."""
        self.available_frameworks = [
            fw for fw, available in get_available_embedders()["text"].items() 
            if available
        ]
        
        self.test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming the world."
        ]
    
    def test_available_frameworks(self):
        """Test that at least one text embedding framework is available."""
        self.assertTrue(
            len(self.available_frameworks) > 0,
            "No text embedding frameworks available. Please install at least one."
        )
    
    def test_text_embedder_creation(self):
        """Test creating a text embedder with each available framework."""
        for framework in self.available_frameworks:
            with self.subTest(framework=framework):
                embedder = create_text_embedder(framework=framework)
                self.assertIsNotNone(embedder, f"Failed to create embedder with framework {framework}")
                self.assertTrue(hasattr(embedder, 'embed'), "Embedder missing 'embed' method")
                self.assertTrue(hasattr(embedder, 'embed_batch'), "Embedder missing 'embed_batch' method")
                self.assertTrue(hasattr(embedder, 'similarity'), "Embedder missing 'similarity' method")
    
    def test_text_embedding(self):
        """Test generating embeddings for text with each available framework."""
        for framework in self.available_frameworks:
            with self.subTest(framework=framework):
                embedder = create_text_embedder(framework=framework)
                
                # Test single text embedding
                embedding = embedder.embed(self.test_texts[0])
                self.assertIsInstance(embedding, np.ndarray, "Embedding should be a numpy array")
                self.assertEqual(embedding.ndim, 1, "Embedding should be a 1D array")
                self.assertEqual(embedding.shape[0], embedder.get_embedding_dim(), 
                                "Embedding dimension mismatch")
                
                # Test batch embedding
                embeddings = embedder.embed_batch(self.test_texts)
                self.assertIsInstance(embeddings, list, "Batch embeddings should be a list")
                self.assertEqual(len(embeddings), len(self.test_texts), 
                                "Number of embeddings should match number of texts")
                self.assertIsInstance(embeddings[0], np.ndarray, "Each embedding should be a numpy array")
                self.assertEqual(embeddings[0].shape[0], embedder.get_embedding_dim(), 
                                "Embedding dimension mismatch")
    
    def test_text_similarity(self):
        """Test calculating similarity between texts with each available framework."""
        for framework in self.available_frameworks:
            with self.subTest(framework=framework):
                embedder = create_text_embedder(framework=framework)
                
                # Test similarity between identical texts
                similarity = embedder.similarity(self.test_texts[0], self.test_texts[0])
                self.assertIsInstance(similarity, float, "Similarity should be a float")
                self.assertAlmostEqual(similarity, 1.0, places=4, 
                                      msg="Similarity between identical texts should be close to 1.0")
                
                # Test similarity between different texts
                similarity = embedder.similarity(self.test_texts[0], self.test_texts[1])
                self.assertIsInstance(similarity, float, "Similarity should be a float")
                self.assertGreaterEqual(similarity, -1.0, "Similarity should be >= -1.0")
                self.assertLessEqual(similarity, 1.0, "Similarity should be <= 1.0")


class TestImageEmbedder(unittest.TestCase):
    """Test cases for ImageEmbedder."""
    
    def setUp(self):
        """Set up test environment."""
        self.available_frameworks = [
            fw for fw, available in get_available_embedders()["image"].items() 
            if available
        ]
        
        # Create test images
        os.makedirs("test_images", exist_ok=True)
        
        self.test_image_paths = [
            "test_images/red.jpg",
            "test_images/blue.jpg"
        ]
        
        # Create simple colored images
        Image.new('RGB', (100, 100), (255, 0, 0)).save(self.test_image_paths[0])  # Red
        Image.new('RGB', (100, 100), (0, 0, 255)).save(self.test_image_paths[1])  # Blue
        
        self.test_images = [Image.open(path) for path in self.test_image_paths]
    
    def test_available_frameworks(self):
        """Test that at least one image embedding framework is available."""
        self.assertTrue(
            len(self.available_frameworks) > 0,
            "No image embedding frameworks available. Please install at least one."
        )
    
    def test_image_embedder_creation(self):
        """Test creating an image embedder with each available framework."""
        for framework in self.available_frameworks:
            with self.subTest(framework=framework):
                embedder = create_image_embedder(framework=framework)
                self.assertIsNotNone(embedder, f"Failed to create embedder with framework {framework}")
                self.assertTrue(hasattr(embedder, 'embed'), "Embedder missing 'embed' method")
                self.assertTrue(hasattr(embedder, 'embed_batch'), "Embedder missing 'embed_batch' method")
                self.assertTrue(hasattr(embedder, 'similarity'), "Embedder missing 'similarity' method")
    
    def test_image_embedding(self):
        """Test generating embeddings for images with each available framework."""
        for framework in self.available_frameworks:
            with self.subTest(framework=framework):
                embedder = create_image_embedder(framework=framework)
                
                # Test single image embedding
                embedding = embedder.embed(self.test_images[0])
                self.assertIsInstance(embedding, np.ndarray, "Embedding should be a numpy array")
                self.assertEqual(embedding.ndim, 1, "Embedding should be a 1D array")
                self.assertEqual(embedding.shape[0], embedder.get_embedding_dim(), 
                                "Embedding dimension mismatch")
                
                # Test batch embedding
                embeddings = embedder.embed_batch(self.test_images)
                self.assertIsInstance(embeddings, list, "Batch embeddings should be a list")
                self.assertEqual(len(embeddings), len(self.test_images), 
                                "Number of embeddings should match number of images")
                self.assertIsInstance(embeddings[0], np.ndarray, "Each embedding should be a numpy array")
                self.assertEqual(embeddings[0].shape[0], embedder.get_embedding_dim(), 
                                "Embedding dimension mismatch")
    
    def test_image_similarity(self):
        """Test calculating similarity between images with each available framework."""
        for framework in self.available_frameworks:
            with self.subTest(framework=framework):
                embedder = create_image_embedder(framework=framework)
                
                # Test similarity between identical images
                similarity = embedder.similarity(self.test_images[0], self.test_images[0])
                self.assertIsInstance(similarity, float, "Similarity should be a float")
                self.assertAlmostEqual(similarity, 1.0, places=4, 
                                      msg="Similarity between identical images should be close to 1.0")
                
                # Test similarity between different images
                similarity = embedder.similarity(self.test_images[0], self.test_images[1])
                self.assertIsInstance(similarity, float, "Similarity should be a float")
                self.assertGreaterEqual(similarity, -1.0, "Similarity should be >= -1.0")
                self.assertLessEqual(similarity, 1.0, "Similarity should be <= 1.0")


class TestAudioEmbedder(unittest.TestCase):
    """Test cases for AudioEmbedder."""
    
    def setUp(self):
        """Set up test environment."""
        self.available_frameworks = [
            fw for fw, available in get_available_embedders()["audio"].items() 
            if available
        ]
        
        # Create test audio files or use sample rate arrays
        # For simplicity, we'll use numpy arrays representing audio waveforms
        sample_rate = 16000  # 16kHz
        duration = 1  # 1 second
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # Create a simple sine wave at 440Hz (A4 note)
        self.test_audio1 = np.sin(2 * np.pi * 440 * t)
        
        # Create another sine wave at 880Hz (A5 note)
        self.test_audio2 = np.sin(2 * np.pi * 880 * t)
        
        self.test_audios = [self.test_audio1, self.test_audio2]
    
    def test_available_frameworks(self):
        """Test that at least one audio embedding framework is available."""
        self.assertTrue(
            len(self.available_frameworks) > 0,
            "No audio embedding frameworks available. Please install at least one."
        )
    
    def test_audio_embedder_creation(self):
        """Test creating an audio embedder with each available framework."""
        for framework in self.available_frameworks:
            with self.subTest(framework=framework):
                embedder = create_audio_embedder(framework=framework)
                self.assertIsNotNone(embedder, f"Failed to create audio embedder with {framework}")
                self.assertTrue(embedder.initialized, f"Failed to initialize {framework} embedder")
    
    def test_audio_embedding(self):
        """Test generating embeddings with each available framework."""
        for framework in self.available_frameworks:
            with self.subTest(framework=framework):
                embedder = create_audio_embedder(framework=framework)
                
                # Test single audio embedding
                embedding = embedder.embed(self.test_audio1)
                self.assertIsInstance(embedding, np.ndarray, "Embedding should be a numpy array")
                self.assertEqual(embedding.ndim, 1, "Embedding should be a 1D array")
                self.assertEqual(embedding.shape[0], embedder.embedding_dim, 
                                "Embedding dimension mismatch")
                
                # Test batch embedding
                embeddings = embedder.embed_batch(self.test_audios)
                self.assertIsInstance(embeddings, np.ndarray, "Batch embeddings should be a numpy array")
                self.assertEqual(len(embeddings), len(self.test_audios), 
                                "Number of embeddings should match number of audio inputs")
                self.assertIsInstance(embeddings[0], np.ndarray, "Each embedding should be a numpy array")
                self.assertEqual(embeddings[0].shape[0], embedder.embedding_dim, 
                                "Embedding dimension mismatch")
    
    def test_audio_similarity(self):
        """Test calculating similarity between audio inputs with each available framework."""
        for framework in self.available_frameworks:
            with self.subTest(framework=framework):
                embedder = create_audio_embedder(framework=framework)
                
                # Test similarity between identical audio
                similarity = embedder.similarity(self.test_audio1, self.test_audio1)
                self.assertIsInstance(similarity, float, "Similarity should be a float")
                self.assertAlmostEqual(similarity, 1.0, places=4, 
                                      msg="Similarity between identical audio should be close to 1.0")
                
                # Test similarity between different audio
                similarity = embedder.similarity(self.test_audio1, self.test_audio2)
                self.assertIsInstance(similarity, float, "Similarity should be a float")
                self.assertGreaterEqual(similarity, -1.0, "Similarity should be >= -1.0")
                self.assertLessEqual(similarity, 1.0, "Similarity should be <= 1.0")


class TestMultiModalEmbedder(unittest.TestCase):
    """Test cases for MultiModalEmbedder."""
    
    def setUp(self):
        """Set up test environment."""
        self.available_text_frameworks = [
            fw for fw, available in get_available_embedders()["text"].items() 
            if available
        ]
        
        self.available_image_frameworks = [
            fw for fw, available in get_available_embedders()["image"].items() 
            if available
        ]
        
        self.available_audio_frameworks = [
            fw for fw, available in get_available_embedders()["audio"].items() 
            if available
        ]
        
        self.test_texts = [
            "A red square",
            "A blue circle"
        ]
        
        # Reuse test images from ImageEmbedder test
        self.test_image_paths = [
            "test_images/red.jpg",
            "test_images/blue.jpg"
        ]
        
        self.test_images = [Image.open(path) for path in self.test_image_paths]
        
        # Create test audio data
        sample_rate = 16000  # 16kHz
        duration = 1  # 1 second
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        
        # Create a simple sine wave at 440Hz (A4 note)
        self.test_audio1 = np.sin(2 * np.pi * 440 * t)
        
        # Create another sine wave at 880Hz (A5 note)
        self.test_audio2 = np.sin(2 * np.pi * 880 * t)
        
        self.test_audios = [self.test_audio1, self.test_audio2]
        
        # Mixed inputs (text, image, and audio)
        self.mixed_inputs = [
            self.test_texts[0],  # Text
            self.test_images[0],  # Image
            self.test_audios[0],  # Audio
            self.test_texts[1],  # Text
            self.test_images[1],  # Image
            self.test_audios[1]   # Audio
        ]
    
    def test_available_frameworks(self):
        """Test that at least one text and one image embedding framework is available."""
        self.assertTrue(
            len(self.available_text_frameworks) > 0,
            "No text embedding frameworks available. Please install at least one."
        )
        
        self.assertTrue(
            len(self.available_image_frameworks) > 0,
            "No image embedding frameworks available. Please install at least one."
        )
    
    def test_multimodal_embedder_creation(self):
        """Test creating a multimodal embedder with available frameworks."""
        if not self.available_text_frameworks or not self.available_image_frameworks or not self.available_audio_frameworks:
            self.skipTest("Missing required frameworks for multimodal embedder test")
        
        text_framework = self.available_text_frameworks[0]
        image_framework = self.available_image_frameworks[0]
        audio_framework = self.available_audio_frameworks[0]
        
        embedder = create_multimodal_embedder(
            text_framework=text_framework,
            image_framework=image_framework,
            audio_framework=audio_framework
        )
        
        self.assertIsNotNone(embedder, "Failed to create multimodal embedder")
        self.assertTrue(hasattr(embedder, 'embed'), "Embedder missing 'embed' method")
        self.assertTrue(hasattr(embedder, 'embed_batch'), "Embedder missing 'embed_batch' method")
        self.assertTrue(hasattr(embedder, 'similarity'), "Embedder missing 'similarity' method")
    
    def test_multimodal_embedding(self):
        """Test generating embeddings for mixed inputs with available frameworks."""
        if not self.available_text_frameworks or not self.available_image_frameworks or not self.available_audio_frameworks:
            self.skipTest("Missing required frameworks for multimodal embedder test")
        
        text_framework = self.available_text_frameworks[0]
        image_framework = self.available_image_frameworks[0]
        audio_framework = self.available_audio_frameworks[0]
        
        embedder = create_multimodal_embedder(
            text_framework=text_framework,
            image_framework=image_framework,
            audio_framework=audio_framework
        )
        
        # Test single text embedding
        text_embedding = embedder.embed(self.test_texts[0])
        self.assertIsInstance(text_embedding, np.ndarray, "Text embedding should be a numpy array")
        
        # Test single image embedding
        image_embedding = embedder.embed(self.test_images[0])
        self.assertIsInstance(image_embedding, np.ndarray, "Image embedding should be a numpy array")
        
        # Test single audio embedding
        audio_embedding = embedder.embed(self.test_audios[0])
        self.assertIsInstance(audio_embedding, np.ndarray, "Audio embedding should be a numpy array")
        
        # Test batch embedding with mixed inputs
        embeddings = embedder.embed_batch(self.mixed_inputs)
        self.assertIsInstance(embeddings, list, "Batch embeddings should be a list")
        self.assertEqual(len(embeddings), len(self.mixed_inputs), 
                        "Number of embeddings should match number of inputs")
        
        # All embeddings should have the same dimension
        dim = embeddings[0].shape[0]
        for emb in embeddings:
            self.assertEqual(emb.shape[0], dim, "All embeddings should have the same dimension")
    
    def test_multimodal_similarity(self):
        """Test calculating similarity between mixed inputs with available frameworks."""
        if not self.available_text_frameworks or not self.available_image_frameworks or not self.available_audio_frameworks:
            self.skipTest("Missing required frameworks for multimodal embedder test")
        
        text_framework = self.available_text_frameworks[0]
        image_framework = self.available_image_frameworks[0]
        audio_framework = self.available_audio_frameworks[0]
        
        embedder = create_multimodal_embedder(
            text_framework=text_framework,
            image_framework=image_framework,
            audio_framework=audio_framework
        )
        
        # Test similarity between text and image
        similarity = embedder.similarity(self.test_texts[0], self.test_images[0])
        self.assertIsInstance(similarity, float, "Similarity should be a float")
        self.assertGreaterEqual(similarity, -1.0, "Similarity should be >= -1.0")
        self.assertLessEqual(similarity, 1.0, "Similarity should be <= 1.0")
        
        # Test similarity between text and audio
        similarity = embedder.similarity(self.test_texts[0], self.test_audios[0])
        self.assertIsInstance(similarity, float, "Similarity should be a float")
        self.assertGreaterEqual(similarity, -1.0, "Similarity should be >= -1.0")
        self.assertLessEqual(similarity, 1.0, "Similarity should be <= 1.0")
        
        # Test similarity between image and audio
        similarity = embedder.similarity(self.test_images[0], self.test_audios[0])
        self.assertIsInstance(similarity, float, "Similarity should be a float")
        self.assertGreaterEqual(similarity, -1.0, "Similarity should be >= -1.0")
        self.assertLessEqual(similarity, 1.0, "Similarity should be <= 1.0")
        
        # Test similarity between text and text
        similarity = embedder.similarity(self.test_texts[0], self.test_texts[1])
        self.assertIsInstance(similarity, float, "Similarity should be a float")
        self.assertGreaterEqual(similarity, -1.0, "Similarity should be >= -1.0")
        self.assertLessEqual(similarity, 1.0, "Similarity should be <= 1.0")
        
        # Test similarity between image and image
        similarity = embedder.similarity(self.test_images[0], self.test_images[1])
        self.assertIsInstance(similarity, float, "Similarity should be a float")
        self.assertGreaterEqual(similarity, -1.0, "Similarity should be >= -1.0")
        self.assertLessEqual(similarity, 1.0, "Similarity should be <= 1.0")
        
        # Test similarity between audio and audio
        similarity = embedder.similarity(self.test_audios[0], self.test_audios[1])
        self.assertIsInstance(similarity, float, "Similarity should be a float")
        self.assertGreaterEqual(similarity, -1.0, "Similarity should be >= -1.0")
        self.assertLessEqual(similarity, 1.0, "Similarity should be <= 1.0")


if __name__ == "__main__":
    unittest.main()