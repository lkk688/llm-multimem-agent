# Multimodal embedder

import os
import numpy as np
from typing import List, Union, Dict, Any, Optional, Tuple
import logging
from PIL import Image

# Use relative imports instead of absolute imports
from .base import BaseEmbedder
from .text_embedder import TextEmbedder, get_available_frameworks as get_available_text_frameworks
from .image_embedder import ImageEmbedder, get_available_frameworks as get_available_image_frameworks
from .audio_embedder import AudioEmbedder, get_available_frameworks as get_available_audio_frameworks

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/llm-multi-core/embeddings")

# Default models for multimodal embedding
DEFAULT_MODELS = {
    "clip": "ViT-B/32",  # CLIP supports both text and image
    "openai": "text-embedding-3-large",  # OpenAI's text embedding model
    "gemini": "embedding-001",  # Google's embedding model
    "wav2vec2": "facebook/wav2vec2-base-960h",  # Facebook's audio embedding model
    "whisper": "openai/whisper-base",  # OpenAI's audio embedding model
}

class MultiModalEmbedder(BaseEmbedder):
    """
    Multimodal embedder supporting text, image, and audio inputs
    """
    
    def __init__(self, 
                 text_framework: str = "sentence-transformers", 
                 image_framework: str = "clip",
                 audio_framework: str = "wav2vec2",
                 text_model_name: Optional[str] = None,
                 image_model_name: Optional[str] = None,
                 audio_model_name: Optional[str] = None,
                 api_key: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 device: Optional[str] = None,
                 **kwargs):
        """
        Initialize the multimodal embedder
        
        Args:
            text_framework: Text embedding framework to use
            image_framework: Image embedding framework to use
            text_model_name: Name of the text embedding model to use
            image_model_name: Name of the image embedding model to use
            api_key: API key for cloud-based embedding services
            cache_dir: Directory to cache embeddings and models
            device: Device to use for computation ("cpu", "cuda", etc.)
            **kwargs: Additional parameters for the specific frameworks
        """
        # Initialize base class
        super().__init__(model_name=f"{text_framework}+{image_framework}+{audio_framework}", **kwargs)
        
        # Set attributes
        self.text_framework = text_framework
        self.image_framework = image_framework
        self.audio_framework = audio_framework
        self.text_model_name = text_model_name
        self.image_model_name = image_model_name
        self.audio_model_name = audio_model_name
        self.api_key = api_key
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.device = device
        self.kwargs = kwargs
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize embedders
        self.text_embedder = None
        self.image_embedder = None
        self.audio_embedder = None
        self.initialize()
    
    def initialize(self) -> bool:
        """
        Initialize the text, image, and audio embedders
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            # Initialize text embedder
            self.text_embedder = TextEmbedder(
                framework=self.text_framework,
                model_name=self.text_model_name,
                api_key=self.api_key,
                cache_dir=self.cache_dir,
                device=self.device,
                **self.kwargs
            )
            
            # Initialize image embedder
            self.image_embedder = ImageEmbedder(
                framework=self.image_framework,
                model_name=self.image_model_name,
                api_key=self.api_key,
                cache_dir=self.cache_dir,
                device=self.device,
                **self.kwargs
            )
            
            # Initialize audio embedder
            self.audio_embedder = AudioEmbedder(
                framework=self.audio_framework,
                model_name=self.audio_model_name,
                api_key=self.api_key,
                cache_dir=self.cache_dir,
                device=self.device,
                **self.kwargs
            )
            
            # Set embedding dimensions
            self.text_embedding_dim = self.text_embedder.embedding_dim
            self.image_embedding_dim = self.image_embedder.embedding_dim
            self.audio_embedding_dim = self.audio_embedder.embedding_dim
            
            # For compatibility with BaseEmbedder, use text embedding dimension
            self.embedding_dim = self.text_embedding_dim
            
            self.initialized = True
            logger.info(f"Initialized MultiModalEmbedder with text framework: {self.text_framework} (dim={self.text_embedding_dim}), "
                       f"image framework: {self.image_framework} (dim={self.image_embedding_dim}), and "
                       f"audio framework: {self.audio_framework} (dim={self.audio_embedding_dim})")
            return True
        except Exception as e:
            logger.error(f"Error initializing MultiModalEmbedder: {str(e)}")
            return False
    
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text input
        
        Args:
            text: Input text or list of texts
            
        Returns:
            np.ndarray: Text embedding vector(s)
        """
        if not self.initialized or not self.text_embedder.initialized:
            raise RuntimeError("Text embedder not initialized. Call initialize() first.")
        
        if isinstance(text, list):
            return self.text_embedder.embed_batch(text)
        else:
            return self.text_embedder.embed(text)
    
    def embed_image(self, image: Union[str, Image.Image, np.ndarray, List[Union[str, Image.Image, np.ndarray]]]) -> np.ndarray:
        """
        Generate embeddings for image input
        
        Args:
            image: Input image or list of images (file paths, PIL Images, base64 strings, or numpy arrays)
            
        Returns:
            np.ndarray: Image embedding vector(s)
        """
        if not self.initialized or not self.image_embedder.initialized:
            raise RuntimeError("Image embedder not initialized. Call initialize() first.")
        
        if isinstance(image, list):
            return self.image_embedder.embed_batch(image)
        else:
            return self.image_embedder.embed(image)
    
    def embed_audio(self, audio: Union[str, np.ndarray, List[Union[str, np.ndarray]]]) -> np.ndarray:
        """
        Generate embeddings for audio input
        
        Args:
            audio: Input audio or list of audio (file paths or numpy arrays)
            
        Returns:
            np.ndarray: Audio embedding vector(s)
        """
        if not self.initialized or not self.audio_embedder.initialized:
            raise RuntimeError("Audio embedder not initialized. Call initialize() first.")
        
        if isinstance(audio, list):
            return self.audio_embedder.embed_batch(audio)
        else:
            return self.audio_embedder.embed(audio)
    
    def embed(self, input_data: Any) -> np.ndarray:
        """
        Generate embeddings for the input data, automatically detecting the type
        
        Args:
            input_data: Input data (text, image, audio, or list of any)
            
        Returns:
            np.ndarray: Embedding vector
        """
        if not self.initialized:
            raise RuntimeError("Embedder not initialized. Call initialize() first.")
        
        # Detect input type
        if isinstance(input_data, str):
            # Check if it's an image path or URL
            if os.path.exists(input_data) and input_data.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                return self.embed_image(input_data)
            # Check if it's an audio path
            elif os.path.exists(input_data) and input_data.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
                return self.embed_audio(input_data)
            elif input_data.startswith("data:image") or input_data.startswith("http") and any(input_data.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp']):
                return self.embed_image(input_data)
            else:
                # Assume it's text
                return self.embed_text(input_data)
        elif isinstance(input_data, Image.Image):
            # It's a PIL image
            return self.embed_image(input_data)
        elif isinstance(input_data, np.ndarray):
            # Determine if it's an image or audio based on shape
            if len(input_data.shape) >= 2:
                # Multi-dimensional array is likely an image
                return self.embed_image(input_data)
            else:
                # 1D array is likely audio
                return self.embed_audio(input_data)
        elif isinstance(input_data, list):
            if len(input_data) == 0:
                raise ValueError("Empty list provided")
            
            # Check the type of the first element
            first_elem = input_data[0]
            if isinstance(first_elem, str):
                # Check if it's an image path or URL
                if os.path.exists(first_elem) and first_elem.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                    return self.embed_image(input_data)
                # Check if it's an audio path
                elif os.path.exists(first_elem) and first_elem.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
                    return self.embed_audio(input_data)
                elif first_elem.startswith("data:image") or first_elem.startswith("http") and any(first_elem.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp']):
                    return self.embed_image(input_data)
                else:
                    # Assume it's text
                    return self.embed_text(input_data)
            elif isinstance(first_elem, Image.Image):
                # It's a list of PIL images
                return self.embed_image(input_data)
            elif isinstance(first_elem, np.ndarray):
                # Determine if it's a list of images or audio based on shape
                if len(first_elem.shape) >= 2:
                    # Multi-dimensional arrays are likely images
                    return self.embed_image(input_data)
                else:
                    # 1D arrays are likely audio
                    return self.embed_audio(input_data)
            else:
                raise ValueError(f"Unsupported input type in list: {type(first_elem)}")
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
    
    def embed_batch(self, inputs: List[Any]) -> np.ndarray:
        """
        Generate embeddings for a batch of inputs
        
        Args:
            inputs: List of input data (can be mixed text, images, and audio)
            
        Returns:
            np.ndarray: Batch of embedding vectors
        """
        if not self.initialized:
            raise RuntimeError("Embedder not initialized. Call initialize() first.")
        
        # Separate text, image, and audio inputs
        text_inputs = []
        text_indices = []
        image_inputs = []
        image_indices = []
        audio_inputs = []
        audio_indices = []
        
        for i, input_data in enumerate(inputs):
            if isinstance(input_data, str):
                # Check if it's an image path or URL
                if os.path.exists(input_data) and input_data.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                    image_inputs.append(input_data)
                    image_indices.append(i)
                # Check if it's an audio path
                elif os.path.exists(input_data) and input_data.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
                    audio_inputs.append(input_data)
                    audio_indices.append(i)
                elif input_data.startswith("data:image") or input_data.startswith("http") and any(input_data.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp']):
                    image_inputs.append(input_data)
                    image_indices.append(i)
                else:
                    # Assume it's text
                    text_inputs.append(input_data)
                    text_indices.append(i)
            elif isinstance(input_data, Image.Image):
                # It's a PIL image
                image_inputs.append(input_data)
                image_indices.append(i)
            elif isinstance(input_data, np.ndarray):
                # Determine if it's an image or audio based on shape
                if len(input_data.shape) >= 2:
                    # Multi-dimensional array is likely an image
                    image_inputs.append(input_data)
                    image_indices.append(i)
                else:
                    # 1D array is likely audio
                    audio_inputs.append(input_data)
                    audio_indices.append(i)
            else:
                raise ValueError(f"Unsupported input type: {type(input_data)}")
        
        # Initialize result array
        result = np.zeros((len(inputs), self.embedding_dim))
        
        # Get text embeddings
        if text_inputs:
            text_embeddings = self.text_embedder.embed_batch(text_inputs)
            for i, idx in enumerate(text_indices):
                result[idx] = text_embeddings[i]
        
        # Get image embeddings
        if image_inputs:
            image_embeddings = self.image_embedder.embed_batch(image_inputs)
            # If image embedding dimension is different from text, we need to project
            if self.image_embedding_dim != self.text_embedding_dim:
                # Simple projection: truncate or pad with zeros
                projected_embeddings = np.zeros((len(image_embeddings), self.text_embedding_dim))
                for i, emb in enumerate(image_embeddings):
                    min_dim = min(self.image_embedding_dim, self.text_embedding_dim)
                    projected_embeddings[i, :min_dim] = emb[:min_dim]
                image_embeddings = projected_embeddings
            
            for i, idx in enumerate(image_indices):
                result[idx] = image_embeddings[i]
        
        # Get audio embeddings
        if audio_inputs:
            audio_embeddings = self.audio_embedder.embed_batch(audio_inputs)
            # If audio embedding dimension is different from text, we need to project
            if self.audio_embedding_dim != self.text_embedding_dim:
                # Simple projection: truncate or pad with zeros
                projected_embeddings = np.zeros((len(audio_embeddings), self.text_embedding_dim))
                for i, emb in enumerate(audio_embeddings):
                    min_dim = min(self.audio_embedding_dim, self.text_embedding_dim)
                    projected_embeddings[i, :min_dim] = emb[:min_dim]
                audio_embeddings = projected_embeddings
            
            for i, idx in enumerate(audio_indices):
                result[idx] = audio_embeddings[i]
        
        return result
    
    def similarity(self, a: Union[str, Image.Image, np.ndarray], b: Union[str, Image.Image, np.ndarray]) -> float:
        """
        Calculate cosine similarity between two inputs (text or image)
        
        Args:
            a: First input (text or image)
            b: Second input (text or image)
            
        Returns:
            float: Cosine similarity score
        """
        if not self.initialized:
            raise RuntimeError("Embedder not initialized. Call initialize() first.")
        
        # Get embeddings
        embedding_a = self.embed(a)
        embedding_b = self.embed(b)
        
        # Calculate cosine similarity
        return np.dot(embedding_a, embedding_b) / (np.linalg.norm(embedding_a) * np.linalg.norm(embedding_b))
    
    def __str__(self) -> str:
        return f"MultiModalEmbedder(text_framework={self.text_framework}, image_framework={self.image_framework}, audio_framework={self.audio_framework})"

# Factory function to create multimodal embedder
def create_multimodal_embedder(text_framework: str = "sentence-transformers", 
                              image_framework: str = "clip",
                              audio_framework: str = "wav2vec2",
                              **kwargs) -> MultiModalEmbedder:
    """
    Create a multimodal embedder with the specified frameworks
    
    Args:
        text_framework: Text embedding framework to use
        image_framework: Image embedding framework to use
        audio_framework: Audio embedding framework to use
        **kwargs: Additional parameters for the embedder
        
    Returns:
        MultiModalEmbedder: Initialized multimodal embedder
    """
    return MultiModalEmbedder(text_framework=text_framework, image_framework=image_framework, audio_framework=audio_framework, **kwargs)

# Get available frameworks for text, image, and audio
def get_available_frameworks() -> Dict[str, Dict[str, bool]]:
    """
    Get a dictionary of available embedding frameworks for text, image, and audio
    
    Returns:
        Dict[str, Dict[str, bool]]: Dictionary mapping modality to framework availability
    """
    return {
        "text": get_available_text_frameworks(),
        "image": get_available_image_frameworks(),
        "audio": get_available_audio_frameworks()
    }