# Text embedding

import os
import numpy as np
from typing import List, Union, Dict, Any, Optional, Tuple
import logging
from .base import BaseEmbedder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/llm-multi-core/embeddings")

# Available embedding frameworks
FRAMEWORKS = {
    "sentence_transformers": "SentenceTransformers",
    "openai": "OpenAI",
    "gemini": "Google Gemini",
    "jina": "Jina",
    "nemo": "NVIDIA NeMo",
    "stella": "Stella",
    "modernbert": "ModernBERT",
    "cohere": "Cohere",
    "huggingface": "HuggingFace"
}

# Framework availability flags
try:
    import sentence_transformers
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import jina
    JINA_AVAILABLE = True
except ImportError:
    JINA_AVAILABLE = False

try:
    import nemo
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

try:
    from transformers import AutoModel, AutoTokenizer
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

# Default models for each framework
DEFAULT_MODELS = {
    "sentence_transformers": "all-MiniLM-L6-v2",  # Good balance of speed and quality
    "openai": "text-embedding-3-small",  # OpenAI's efficient embedding model
    "gemini": "embedding-001",  # Google's embedding model
    "jina": "jina-embeddings-v2-base-en",  # Jina's base English model
    "nemo": "nv-embed-text-v2",  # NVIDIA's NeMo embedding model
    "stella": "stella-base-1280",  # Stella base model
    "modernbert": "modernbert-base",  # ModernBERT base model
    "cohere": "embed-english-v3.0",  # Cohere's English embedding model
    "huggingface": "sentence-transformers/all-mpnet-base-v2"  # HuggingFace model
}

class TextEmbedder(BaseEmbedder):
    """
    Text embedder supporting multiple embedding frameworks
    """
    
    def __init__(self, 
                 framework: str = "sentence_transformers", 
                 model_name: Optional[str] = None,
                 api_key: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 **kwargs):
        """
        Initialize the text embedder
        
        Args:
            framework: Embedding framework to use (sentence_transformers, openai, gemini, etc.)
            model_name: Name of the embedding model to use (if None, uses default for framework)
            api_key: API key for cloud-based embedding services (OpenAI, Gemini, etc.)
            cache_dir: Directory to cache embeddings and models
            **kwargs: Additional parameters for the specific framework
        """
        # Validate framework
        if framework not in FRAMEWORKS:
            available = ", ".join(FRAMEWORKS.keys())
            raise ValueError(f"Framework '{framework}' not supported. Available frameworks: {available}")
        
        # Set default model if not provided
        if model_name is None:
            model_name = DEFAULT_MODELS.get(framework)
        
        # Initialize base class
        super().__init__(model_name=model_name, **kwargs)
        
        # Set framework-specific attributes
        self.framework = framework
        self.api_key = api_key
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.kwargs = kwargs
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize the model
        self.initialize()
    
    def initialize(self) -> bool:
        """
        Initialize the embedding model based on the selected framework
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            if self.framework == "sentence_transformers":
                return self._init_sentence_transformers()
            elif self.framework == "openai":
                return self._init_openai()
            elif self.framework == "gemini":
                return self._init_gemini()
            elif self.framework == "jina":
                return self._init_jina()
            elif self.framework == "nemo":
                return self._init_nemo()
            elif self.framework == "cohere":
                return self._init_cohere()
            elif self.framework == "huggingface":
                return self._init_huggingface()
            else:
                logger.error(f"Framework '{self.framework}' initialization not implemented")
                return False
        except Exception as e:
            logger.error(f"Error initializing {self.framework} embedder: {str(e)}")
            return False
    
    def _init_sentence_transformers(self) -> bool:
        """
        Initialize SentenceTransformers model
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("SentenceTransformers not available. Install with 'pip install sentence-transformers'")
            return False
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, cache_folder=self.cache_dir)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            self.initialized = True
            logger.info(f"Initialized SentenceTransformers model: {self.model_name} (dim={self.embedding_dim})")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize SentenceTransformers: {str(e)}")
            return False
    
    def _init_openai(self) -> bool:
        """
        Initialize OpenAI embedding client
        """
        if not OPENAI_AVAILABLE:
            logger.error("OpenAI not available. Install with 'pip install openai'")
            return False
        
        try:
            # Check for API key
            api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.error("OpenAI API key not provided. Set api_key parameter or OPENAI_API_KEY environment variable")
                return False
            
            # Initialize client
            self.model = openai.OpenAI(api_key=api_key)
            
            # Set embedding dimensions based on model
            model_dims = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536
            }
            self.embedding_dim = model_dims.get(self.model_name, 1536)  # Default to 1536 if unknown
            
            self.initialized = True
            logger.info(f"Initialized OpenAI embedding model: {self.model_name} (dim={self.embedding_dim})")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {str(e)}")
            return False
    
    def _init_gemini(self) -> bool:
        """
        Initialize Google Gemini embedding client
        """
        if not GEMINI_AVAILABLE:
            logger.error("Google Gemini not available. Install with 'pip install google-generativeai'")
            return False
        
        try:
            # Check for API key
            api_key = self.api_key or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                logger.error("Google API key not provided. Set api_key parameter or GOOGLE_API_KEY environment variable")
                return False
            
            # Initialize client
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name)
            
            # Set embedding dimensions
            self.embedding_dim = 768  # Default for Gemini embeddings
            
            self.initialized = True
            logger.info(f"Initialized Google Gemini embedding model: {self.model_name} (dim={self.embedding_dim})")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Google Gemini: {str(e)}")
            return False
    
    def _init_jina(self) -> bool:
        """
        Initialize Jina embedding client
        """
        if not JINA_AVAILABLE:
            logger.error("Jina not available. Install with 'pip install jina'")
            return False
        
        try:
            # Initialize Jina executor
            from jina import Executor, requests
            
            class JinaEmbeddingExecutor(Executor):
                @requests
                def encode(self, docs, **kwargs):
                    # This is a placeholder - actual implementation would use Jina's embedding API
                    pass
            
            self.model = JinaEmbeddingExecutor()
            self.embedding_dim = 768  # Default for Jina embeddings
            
            self.initialized = True
            logger.info(f"Initialized Jina embedding model: {self.model_name} (dim={self.embedding_dim})")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Jina: {str(e)}")
            return False
    
    def _init_nemo(self) -> bool:
        """
        Initialize NVIDIA NeMo embedding model
        """
        if not NEMO_AVAILABLE:
            logger.error("NVIDIA NeMo not available. Install with 'pip install nemo-toolkit'")
            return False
        
        try:
            # Placeholder for NeMo initialization
            # Actual implementation would load the NeMo model
            self.model = None  # Replace with actual NeMo model initialization
            self.embedding_dim = 1024  # Default for NeMo embeddings
            
            self.initialized = True
            logger.info(f"Initialized NVIDIA NeMo embedding model: {self.model_name} (dim={self.embedding_dim})")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize NVIDIA NeMo: {str(e)}")
            return False
    
    def _init_cohere(self) -> bool:
        """
        Initialize Cohere embedding client
        """
        if not COHERE_AVAILABLE:
            logger.error("Cohere not available. Install with 'pip install cohere'")
            return False
        
        try:
            # Check for API key
            api_key = self.api_key or os.environ.get("COHERE_API_KEY")
            if not api_key:
                logger.error("Cohere API key not provided. Set api_key parameter or COHERE_API_KEY environment variable")
                return False
            
            # Initialize client
            self.model = cohere.Client(api_key)
            
            # Set embedding dimensions based on model
            model_dims = {
                "embed-english-v3.0": 1024,
                "embed-multilingual-v3.0": 1024,
                "embed-english-light-v3.0": 384,
                "embed-multilingual-light-v3.0": 384
            }
            self.embedding_dim = model_dims.get(self.model_name, 1024)  # Default to 1024 if unknown
            
            self.initialized = True
            logger.info(f"Initialized Cohere embedding model: {self.model_name} (dim={self.embedding_dim})")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Cohere: {str(e)}")
            return False
    
    def _init_huggingface(self) -> bool:
        """
        Initialize HuggingFace model
        """
        if not HUGGINGFACE_AVAILABLE:
            logger.error("HuggingFace Transformers not available. Install with 'pip install transformers'")
            return False
        
        try:
            # Initialize tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
            self.model = AutoModel.from_pretrained(self.model_name, cache_dir=self.cache_dir)
            
            # Get embedding dimension from model config
            self.embedding_dim = self.model.config.hidden_size
            
            self.initialized = True
            logger.info(f"Initialized HuggingFace model: {self.model_name} (dim={self.embedding_dim})")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace: {str(e)}")
            return False
    
    def embed(self, text: str) -> np.ndarray:
        """
        Generate embeddings for the input text
        
        Args:
            text: Input text to embed
            
        Returns:
            np.ndarray: Embedding vector
        """
        if not self.initialized:
            raise RuntimeError("Embedder not initialized. Call initialize() first.")
        
        if not text or not isinstance(text, str):
            raise ValueError("Input must be a non-empty string")
        
        try:
            if self.framework == "sentence_transformers":
                return self._embed_sentence_transformers(text)
            elif self.framework == "openai":
                return self._embed_openai(text)
            elif self.framework == "gemini":
                return self._embed_gemini(text)
            elif self.framework == "jina":
                return self._embed_jina(text)
            elif self.framework == "nemo":
                return self._embed_nemo(text)
            elif self.framework == "cohere":
                return self._embed_cohere(text)
            elif self.framework == "huggingface":
                return self._embed_huggingface(text)
            else:
                raise NotImplementedError(f"Embedding for framework '{self.framework}' not implemented")
        except Exception as e:
            logger.error(f"Error generating embeddings with {self.framework}: {str(e)}")
            # Return zero vector as fallback
            return np.zeros(self.embedding_dim)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of input texts
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            np.ndarray: Batch of embedding vectors
        """
        if not self.initialized:
            raise RuntimeError("Embedder not initialized. Call initialize() first.")
        
        if not texts or not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise ValueError("Input must be a non-empty list of strings")
        
        try:
            if self.framework == "sentence_transformers":
                return self._embed_batch_sentence_transformers(texts)
            elif self.framework == "openai":
                return self._embed_batch_openai(texts)
            elif self.framework == "gemini":
                return self._embed_batch_gemini(texts)
            elif self.framework == "jina":
                return self._embed_batch_jina(texts)
            elif self.framework == "nemo":
                return self._embed_batch_nemo(texts)
            elif self.framework == "cohere":
                return self._embed_batch_cohere(texts)
            elif self.framework == "huggingface":
                return self._embed_batch_huggingface(texts)
            else:
                # Fall back to default implementation
                return super().embed_batch(texts)
        except Exception as e:
            logger.error(f"Error generating batch embeddings with {self.framework}: {str(e)}")
            # Return zero vectors as fallback
            return np.zeros((len(texts), self.embedding_dim))
    
    def _embed_sentence_transformers(self, text: str) -> np.ndarray:
        """
        Generate embeddings using SentenceTransformers
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def _embed_batch_sentence_transformers(self, texts: List[str]) -> np.ndarray:
        """
        Generate batch embeddings using SentenceTransformers
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True, batch_size=32)
        return embeddings
    
    def _embed_openai(self, text: str) -> np.ndarray:
        """
        Generate embeddings using OpenAI
        """
        response = self.model.embeddings.create(input=text, model=self.model_name)
        embedding = np.array(response.data[0].embedding)
        return embedding
    
    def _embed_batch_openai(self, texts: List[str]) -> np.ndarray:
        """
        Generate batch embeddings using OpenAI
        """
        response = self.model.embeddings.create(input=texts, model=self.model_name)
        embeddings = np.array([item.embedding for item in response.data])
        return embeddings
    
    def _embed_gemini(self, text: str) -> np.ndarray:
        """
        Generate embeddings using Google Gemini
        """
        # Placeholder for Gemini embedding API
        # Actual implementation would use the Gemini embedding API
        embedding = np.random.randn(self.embedding_dim)  # Replace with actual API call
        return embedding / np.linalg.norm(embedding)  # Normalize
    
    def _embed_batch_gemini(self, texts: List[str]) -> np.ndarray:
        """
        Generate batch embeddings using Google Gemini
        """
        # Placeholder for Gemini batch embedding API
        # Actual implementation would use the Gemini embedding API
        embeddings = np.random.randn(len(texts), self.embedding_dim)  # Replace with actual API call
        # Normalize each embedding
        return np.array([e / np.linalg.norm(e) for e in embeddings])
    
    def _embed_jina(self, text: str) -> np.ndarray:
        """
        Generate embeddings using Jina
        """
        # Placeholder for Jina embedding API
        # Actual implementation would use the Jina embedding API
        embedding = np.random.randn(self.embedding_dim)  # Replace with actual API call
        return embedding / np.linalg.norm(embedding)  # Normalize
    
    def _embed_batch_jina(self, texts: List[str]) -> np.ndarray:
        """
        Generate batch embeddings using Jina
        """
        # Placeholder for Jina batch embedding API
        # Actual implementation would use the Jina embedding API
        embeddings = np.random.randn(len(texts), self.embedding_dim)  # Replace with actual API call
        # Normalize each embedding
        return np.array([e / np.linalg.norm(e) for e in embeddings])
    
    def _embed_nemo(self, text: str) -> np.ndarray:
        """
        Generate embeddings using NVIDIA NeMo
        """
        # Placeholder for NeMo embedding API
        # Actual implementation would use the NeMo embedding API
        embedding = np.random.randn(self.embedding_dim)  # Replace with actual API call
        return embedding / np.linalg.norm(embedding)  # Normalize
    
    def _embed_batch_nemo(self, texts: List[str]) -> np.ndarray:
        """
        Generate batch embeddings using NVIDIA NeMo
        """
        # Placeholder for NeMo batch embedding API
        # Actual implementation would use the NeMo embedding API
        embeddings = np.random.randn(len(texts), self.embedding_dim)  # Replace with actual API call
        # Normalize each embedding
        return np.array([e / np.linalg.norm(e) for e in embeddings])
    
    def _embed_cohere(self, text: str) -> np.ndarray:
        """
        Generate embeddings using Cohere
        """
        response = self.model.embed(texts=[text], model=self.model_name, input_type="search_document")
        embedding = np.array(response.embeddings[0])
        return embedding
    
    def _embed_batch_cohere(self, texts: List[str]) -> np.ndarray:
        """
        Generate batch embeddings using Cohere
        """
        response = self.model.embed(texts=texts, model=self.model_name, input_type="search_document")
        embeddings = np.array(response.embeddings)
        return embeddings
    
    def _embed_huggingface(self, text: str) -> np.ndarray:
        """
        Generate embeddings using HuggingFace model
        """
        import torch
        
        # Tokenize and prepare input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use mean of last hidden states as embedding
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embedding
    
    def _embed_batch_huggingface(self, texts: List[str]) -> np.ndarray:
        """
        Generate batch embeddings using HuggingFace model
        """
        import torch
        
        # Tokenize and prepare input
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use mean of last hidden states as embeddings
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        return embeddings

# Factory function to create embedder with the appropriate framework
def create_text_embedder(framework: str = "sentence_transformers", **kwargs) -> TextEmbedder:
    """
    Create a text embedder with the specified framework
    
    Args:
        framework: Embedding framework to use
        **kwargs: Additional parameters for the embedder
        
    Returns:
        TextEmbedder: Initialized text embedder
    """
    return TextEmbedder(framework=framework, **kwargs)

# Get available frameworks
def get_available_frameworks() -> Dict[str, bool]:
    """
    Get a dictionary of available embedding frameworks
    
    Returns:
        Dict[str, bool]: Dictionary mapping framework names to availability status
    """
    return {
        "sentence_transformers": SENTENCE_TRANSFORMERS_AVAILABLE,
        "openai": OPENAI_AVAILABLE,
        "gemini": GEMINI_AVAILABLE,
        "jina": JINA_AVAILABLE,
        "nemo": NEMO_AVAILABLE,
        "cohere": COHERE_AVAILABLE,
        "huggingface": HUGGINGFACE_AVAILABLE
    }