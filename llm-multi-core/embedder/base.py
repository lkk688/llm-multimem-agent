# Base embedder class

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Union, Dict, Any, Optional

class BaseEmbedder(ABC):
    """
    Abstract base class for all embedders (text, image, etc.)
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the embedder with a model name and optional parameters
        
        Args:
            model_name: Name of the embedding model to use
            **kwargs: Additional parameters for the specific embedder implementation
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
        self.initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the embedding model
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def embed(self, data: Any) -> np.ndarray:
        """
        Generate embeddings for the input data
        
        Args:
            data: Input data to embed (text, image, etc.)
            
        Returns:
            np.ndarray: Embedding vector(s)
        """
        pass
    
    def embed_batch(self, data_batch: List[Any]) -> np.ndarray:
        """
        Generate embeddings for a batch of input data
        
        Args:
            data_batch: List of input data to embed
            
        Returns:
            np.ndarray: Batch of embedding vectors
        """
        # Default implementation: embed each item individually
        # Subclasses should override this for more efficient batch processing
        return np.vstack([self.embed(item) for item in data_batch])
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            float: Cosine similarity score (between -1 and 1)
        """
        # Normalize vectors
        embedding1_norm = embedding1 / np.linalg.norm(embedding1)
        embedding2_norm = embedding2 / np.linalg.norm(embedding2)
        
        # Calculate cosine similarity
        return float(np.dot(embedding1_norm, embedding2_norm))
    
    def get_embedding_dim(self) -> int:
        """
        Get the dimensionality of the embeddings
        
        Returns:
            int: Dimension of the embedding vectors
        """
        return self.embedding_dim
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, dim={self.embedding_dim})"