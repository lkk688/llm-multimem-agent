# Audio embedding

import os
import numpy as np
from typing import List, Union, Dict, Any, Optional, Tuple
import logging
import io
from .base import BaseEmbedder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/llm-multi-core/embeddings")

# Available embedding frameworks
FRAMEWORKS = {
    "wav2vec2": "Wav2Vec2",
    "whisper": "OpenAI Whisper",
    "hubert": "HuBERT",
    "wavlm": "WavLM",
    "data2vec": "Data2Vec",
    "openai": "OpenAI",
    "gemini": "Google Gemini"
}

# Framework availability flags
try:
    from transformers import Wav2Vec2Model, Wav2Vec2Processor
    WAV2VEC2_AVAILABLE = True
except ImportError:
    WAV2VEC2_AVAILABLE = False

try:
    from transformers import WhisperModel, WhisperProcessor
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    from transformers import HubertModel, Wav2Vec2Processor as HubertProcessor
    HUBERT_AVAILABLE = True
except ImportError:
    HUBERT_AVAILABLE = False

try:
    from transformers import WavLMModel, Wav2Vec2Processor as WavLMProcessor
    WAVLM_AVAILABLE = True
except ImportError:
    WAVLM_AVAILABLE = False

try:
    from transformers import Data2VecAudioModel, Wav2Vec2Processor as Data2VecProcessor
    DATA2VEC_AVAILABLE = True
except ImportError:
    DATA2VEC_AVAILABLE = False

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
    import torch
    import torchaudio
    TORCH_AUDIO_AVAILABLE = True
except ImportError:
    TORCH_AUDIO_AVAILABLE = False

# Default models for each framework
DEFAULT_MODELS = {
    "wav2vec2": "facebook/wav2vec2-base-960h",  # Good balance of speed and quality
    "whisper": "openai/whisper-base",  # OpenAI's Whisper base model
    "hubert": "facebook/hubert-base-ls960",  # Facebook's HuBERT base model
    "wavlm": "microsoft/wavlm-base",  # Microsoft's WavLM base model
    "data2vec": "facebook/data2vec-audio-base-960h",  # Facebook's Data2Vec audio model
    "openai": "whisper-1",  # OpenAI's Whisper API
    "gemini": "embedding-001"  # Google's embedding model
}

class AudioEmbedder(BaseEmbedder):
    """
    Audio embedder supporting multiple embedding frameworks
    """
    
    def __init__(self, 
                 framework: str = "wav2vec2", 
                 model_name: Optional[str] = None,
                 api_key: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 device: Optional[str] = None,
                 sample_rate: int = 16000,
                 **kwargs):
        """
        Initialize the audio embedder
        
        Args:
            framework: Embedding framework to use (wav2vec2, whisper, hubert, etc.)
            model_name: Name of the embedding model to use (if None, uses default for framework)
            api_key: API key for cloud-based embedding services (OpenAI, Gemini, etc.)
            cache_dir: Directory to cache embeddings and models
            device: Device to use for computation ("cpu", "cuda", etc.)
            sample_rate: Sample rate for audio processing (default: 16000 Hz)
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
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AUDIO_AVAILABLE else None
        self.sample_rate = sample_rate
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
            if self.framework == "wav2vec2":
                return self._init_wav2vec2()
            elif self.framework == "whisper":
                return self._init_whisper()
            elif self.framework == "hubert":
                return self._init_hubert()
            elif self.framework == "wavlm":
                return self._init_wavlm()
            elif self.framework == "data2vec":
                return self._init_data2vec()
            elif self.framework == "openai":
                return self._init_openai()
            elif self.framework == "gemini":
                return self._init_gemini()
            else:
                logger.error(f"Framework '{self.framework}' initialization not implemented")
                return False
        except Exception as e:
            logger.error(f"Error initializing {self.framework} embedder: {str(e)}")
            return False
    
    def _init_wav2vec2(self) -> bool:
        """
        Initialize Wav2Vec2 model
        """
        if not WAV2VEC2_AVAILABLE:
            logger.error("Wav2Vec2 not available. Install with 'pip install transformers'")
            return False
        
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name, cache_dir=self.cache_dir)
            self.model = Wav2Vec2Model.from_pretrained(self.model_name, cache_dir=self.cache_dir).to(self.device)
            self.embedding_dim = self.model.config.hidden_size
            self.initialized = True
            logger.info(f"Initialized Wav2Vec2 model: {self.model_name} (dim={self.embedding_dim})")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Wav2Vec2: {str(e)}")
            return False
    
    def _init_whisper(self) -> bool:
        """
        Initialize Whisper model
        """
        if not WHISPER_AVAILABLE:
            logger.error("Whisper not available. Install with 'pip install transformers'")
            return False
        
        try:
            self.processor = WhisperProcessor.from_pretrained(self.model_name, cache_dir=self.cache_dir)
            self.model = WhisperModel.from_pretrained(self.model_name, cache_dir=self.cache_dir).to(self.device)
            self.embedding_dim = self.model.config.d_model
            self.initialized = True
            logger.info(f"Initialized Whisper model: {self.model_name} (dim={self.embedding_dim})")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Whisper: {str(e)}")
            return False
    
    def _init_hubert(self) -> bool:
        """
        Initialize HuBERT model
        """
        if not HUBERT_AVAILABLE:
            logger.error("HuBERT not available. Install with 'pip install transformers'")
            return False
        
        try:
            self.processor = HubertProcessor.from_pretrained(self.model_name, cache_dir=self.cache_dir)
            self.model = HubertModel.from_pretrained(self.model_name, cache_dir=self.cache_dir).to(self.device)
            self.embedding_dim = self.model.config.hidden_size
            self.initialized = True
            logger.info(f"Initialized HuBERT model: {self.model_name} (dim={self.embedding_dim})")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize HuBERT: {str(e)}")
            return False
    
    def _init_wavlm(self) -> bool:
        """
        Initialize WavLM model
        """
        if not WAVLM_AVAILABLE:
            logger.error("WavLM not available. Install with 'pip install transformers'")
            return False
        
        try:
            self.processor = WavLMProcessor.from_pretrained(self.model_name, cache_dir=self.cache_dir)
            self.model = WavLMModel.from_pretrained(self.model_name, cache_dir=self.cache_dir).to(self.device)
            self.embedding_dim = self.model.config.hidden_size
            self.initialized = True
            logger.info(f"Initialized WavLM model: {self.model_name} (dim={self.embedding_dim})")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize WavLM: {str(e)}")
            return False
    
    def _init_data2vec(self) -> bool:
        """
        Initialize Data2Vec model
        """
        if not DATA2VEC_AVAILABLE:
            logger.error("Data2Vec not available. Install with 'pip install transformers'")
            return False
        
        try:
            self.processor = Data2VecProcessor.from_pretrained(self.model_name, cache_dir=self.cache_dir)
            self.model = Data2VecAudioModel.from_pretrained(self.model_name, cache_dir=self.cache_dir).to(self.device)
            self.embedding_dim = self.model.config.hidden_size
            self.initialized = True
            logger.info(f"Initialized Data2Vec model: {self.model_name} (dim={self.embedding_dim})")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Data2Vec: {str(e)}")
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
            
            # Set embedding dimensions
            self.embedding_dim = 1536  # Default for OpenAI audio embeddings
            
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
    
    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and convert to the required format
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Tuple[np.ndarray, int]: Audio waveform and sample rate
        """
        if not TORCH_AUDIO_AVAILABLE:
            raise ImportError("torchaudio is required for audio loading. Install with 'pip install torchaudio'")
        
        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)
        
        return waveform.squeeze().numpy(), self.sample_rate
    
    def embed(self, audio: Union[str, np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Generate embeddings for the input audio
        
        Args:
            audio: Input audio to embed (file path, numpy array, or torch tensor)
            
        Returns:
            np.ndarray: Embedding vector
        """
        if not self.initialized:
            raise RuntimeError("Embedder not initialized. Call initialize() first.")
        
        # Handle different input types
        if isinstance(audio, str):
            # Load audio from file path
            waveform, sample_rate = self._load_audio(audio)
        elif isinstance(audio, np.ndarray):
            # Use numpy array directly
            waveform = audio
        elif isinstance(audio, torch.Tensor):
            # Convert torch tensor to numpy
            waveform = audio.detach().cpu().numpy()
        else:
            raise ValueError(f"Unsupported audio type: {type(audio)}")
        
        # Process based on framework
        if self.framework == "wav2vec2":
            return self._embed_wav2vec2(waveform)
        elif self.framework == "whisper":
            return self._embed_whisper(waveform)
        elif self.framework == "hubert":
            return self._embed_hubert(waveform)
        elif self.framework == "wavlm":
            return self._embed_wavlm(waveform)
        elif self.framework == "data2vec":
            return self._embed_data2vec(waveform)
        elif self.framework == "openai":
            return self._embed_openai(audio if isinstance(audio, str) else None)
        elif self.framework == "gemini":
            return self._embed_gemini(audio if isinstance(audio, str) else None)
        else:
            raise ValueError(f"Embedding for framework '{self.framework}' not implemented")
    
    def _embed_wav2vec2(self, waveform: np.ndarray) -> np.ndarray:
        """
        Generate embeddings using Wav2Vec2 model
        
        Args:
            waveform: Audio waveform as numpy array
            
        Returns:
            np.ndarray: Embedding vector
        """
        # Convert to tensor and move to device
        inputs = self.processor(waveform, sampling_rate=self.sample_rate, return_tensors="pt").to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use the mean of the last hidden state as the embedding
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        return embeddings.flatten()
    
    def _embed_whisper(self, waveform: np.ndarray) -> np.ndarray:
        """
        Generate embeddings using Whisper model
        
        Args:
            waveform: Audio waveform as numpy array
            
        Returns:
            np.ndarray: Embedding vector
        """
        # Convert to tensor and move to device
        inputs = self.processor(waveform, sampling_rate=self.sample_rate, return_tensors="pt").to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use the mean of the encoder hidden states as the embedding
        embeddings = outputs.encoder_last_hidden_state.mean(dim=1).cpu().numpy()
        
        return embeddings.flatten()
    
    def _embed_hubert(self, waveform: np.ndarray) -> np.ndarray:
        """
        Generate embeddings using HuBERT model
        
        Args:
            waveform: Audio waveform as numpy array
            
        Returns:
            np.ndarray: Embedding vector
        """
        # Convert to tensor and move to device
        inputs = self.processor(waveform, sampling_rate=self.sample_rate, return_tensors="pt").to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use the mean of the last hidden state as the embedding
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        return embeddings.flatten()
    
    def _embed_wavlm(self, waveform: np.ndarray) -> np.ndarray:
        """
        Generate embeddings using WavLM model
        
        Args:
            waveform: Audio waveform as numpy array
            
        Returns:
            np.ndarray: Embedding vector
        """
        # Convert to tensor and move to device
        inputs = self.processor(waveform, sampling_rate=self.sample_rate, return_tensors="pt").to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use the mean of the last hidden state as the embedding
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        return embeddings.flatten()
    
    def _embed_data2vec(self, waveform: np.ndarray) -> np.ndarray:
        """
        Generate embeddings using Data2Vec model
        
        Args:
            waveform: Audio waveform as numpy array
            
        Returns:
            np.ndarray: Embedding vector
        """
        # Convert to tensor and move to device
        inputs = self.processor(waveform, sampling_rate=self.sample_rate, return_tensors="pt").to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use the mean of the last hidden state as the embedding
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        return embeddings.flatten()
    
    def _embed_openai(self, audio_path: Optional[str]) -> np.ndarray:
        """
        Generate embeddings using OpenAI API
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            np.ndarray: Embedding vector
        """
        if audio_path is None:
            raise ValueError("OpenAI embedding requires an audio file path")
        
        with open(audio_path, "rb") as audio_file:
            response = self.model.audio.embeddings.create(
                model=self.model_name,
                file=audio_file
            )
        
        # Extract embedding from response
        embedding = np.array(response.embedding)
        
        return embedding
    
    def _embed_gemini(self, audio_path: Optional[str]) -> np.ndarray:
        """
        Generate embeddings using Google Gemini API
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            np.ndarray: Embedding vector
        """
        if audio_path is None:
            raise ValueError("Gemini embedding requires an audio file path")
        
        # Note: This is a placeholder as Gemini doesn't currently support direct audio embeddings
        # In a real implementation, you might first transcribe the audio and then embed the text
        logger.warning("Gemini direct audio embedding not supported. Using placeholder implementation.")
        
        # Placeholder implementation - in reality, you would use the actual Gemini API
        # This would likely involve transcribing the audio first, then embedding the text
        embedding = np.random.randn(self.embedding_dim)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        
        return embedding
    
    def embed_batch(self, audio_batch: List[Union[str, np.ndarray, torch.Tensor]]) -> np.ndarray:
        """
        Generate embeddings for a batch of input audio
        
        Args:
            audio_batch: List of input audio to embed (file paths, numpy arrays, or torch tensors)
            
        Returns:
            np.ndarray: Batch of embedding vectors
        """
        # For most frameworks, we'll use the base implementation that processes each item individually
        # This could be optimized for specific frameworks that support batch processing
        return np.vstack([self.embed(audio) for audio in audio_batch])


def create_audio_embedder(framework: str = "wav2vec2", model_name: Optional[str] = None, **kwargs) -> AudioEmbedder:
    """
    Create an audio embedder with the specified framework and model
    
    Args:
        framework: Embedding framework to use
        model_name: Name of the embedding model to use (if None, uses default for framework)
        **kwargs: Additional parameters for the embedder
        
    Returns:
        AudioEmbedder: Initialized audio embedder
    """
    return AudioEmbedder(framework=framework, model_name=model_name, **kwargs)


def get_available_frameworks() -> Dict[str, bool]:
    """
    Get available audio embedding frameworks
    
    Returns:
        Dict[str, bool]: Dictionary of framework availability
    """
    return {
        "wav2vec2": WAV2VEC2_AVAILABLE,
        "whisper": WHISPER_AVAILABLE,
        "hubert": HUBERT_AVAILABLE,
        "wavlm": WAVLM_AVAILABLE,
        "data2vec": DATA2VEC_AVAILABLE,
        "openai": OPENAI_AVAILABLE,
        "gemini": GEMINI_AVAILABLE
    }