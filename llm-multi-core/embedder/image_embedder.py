# Image embedding

import os
import numpy as np
from typing import List, Union, Dict, Any, Optional, Tuple
import logging
from PIL import Image
import base64
import io
from .base import BaseEmbedder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/llm-multi-core/embeddings")

# Available embedding frameworks
FRAMEWORKS = {
    "clip": "CLIP",
    "openai": "OpenAI",
    "gemini": "Google Gemini",
    "timm": "PyTorch Image Models",
    "vit": "Vision Transformer",
    "resnet": "ResNet"
}

# Framework availability flags
try:
    import clip
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

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
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

try:
    from transformers import ViTImageProcessor, ViTModel
    VIT_AVAILABLE = True
except ImportError:
    VIT_AVAILABLE = False

try:
    import torchvision.models as models
    import torchvision.transforms as transforms
    RESNET_AVAILABLE = True
except ImportError:
    RESNET_AVAILABLE = False

# Default models for each framework
DEFAULT_MODELS = {
    "clip": "ViT-B/32",  # Good balance of speed and quality
    "openai": "clip",  # OpenAI's CLIP model
    "gemini": "embedding-001",  # Google's embedding model
    "timm": "vit_base_patch16_224",  # Vision Transformer from timm
    "vit": "google/vit-base-patch16-224",  # Vision Transformer from HuggingFace
    "resnet": "resnet50"  # ResNet-50 model
}

class ImageEmbedder(BaseEmbedder):
    """
    Image embedder supporting multiple embedding frameworks
    """
    
    def __init__(self, 
                 framework: str = "clip", 
                 model_name: Optional[str] = None,
                 api_key: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 device: Optional[str] = None,
                 **kwargs):
        """
        Initialize the image embedder
        
        Args:
            framework: Embedding framework to use (clip, openai, gemini, etc.)
            model_name: Name of the embedding model to use (if None, uses default for framework)
            api_key: API key for cloud-based embedding services (OpenAI, Gemini, etc.)
            cache_dir: Directory to cache embeddings and models
            device: Device to use for computation ("cpu", "cuda", etc.)
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
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu") if framework in ["clip", "timm", "vit", "resnet"] else None
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
            if self.framework == "clip":
                return self._init_clip()
            elif self.framework == "openai":
                return self._init_openai()
            elif self.framework == "gemini":
                return self._init_gemini()
            elif self.framework == "timm":
                return self._init_timm()
            elif self.framework == "vit":
                return self._init_vit()
            elif self.framework == "resnet":
                return self._init_resnet()
            else:
                logger.error(f"Framework '{self.framework}' initialization not implemented")
                return False
        except Exception as e:
            logger.error(f"Error initializing {self.framework} embedder: {str(e)}")
            return False
    
    def _init_clip(self) -> bool:
        """
        Initialize CLIP model
        """
        if not CLIP_AVAILABLE:
            logger.error("CLIP not available. Install with 'pip install ftfy regex tqdm git+https://github.com/openai/CLIP.git'")
            return False
        
        try:
            self.model, self.preprocess = clip.load(self.model_name, device=self.device)
            self.embedding_dim = self.model.visual.output_dim
            self.initialized = True
            logger.info(f"Initialized CLIP model: {self.model_name} (dim={self.embedding_dim}) on {self.device}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize CLIP: {str(e)}")
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
            self.embedding_dim = 1024  # Default for OpenAI image embeddings
            
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
    
    def _init_timm(self) -> bool:
        """
        Initialize timm model
        """
        if not TIMM_AVAILABLE:
            logger.error("timm not available. Install with 'pip install timm'")
            return False
        
        try:
            # Initialize model
            self.model = timm.create_model(self.model_name, pretrained=True)
            self.model.eval()
            self.model = self.model.to(self.device)
            
            # Set up preprocessing
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            
            # Get embedding dimension
            self.embedding_dim = self.model.num_features
            
            self.initialized = True
            logger.info(f"Initialized timm model: {self.model_name} (dim={self.embedding_dim}) on {self.device}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize timm: {str(e)}")
            return False
    
    def _init_vit(self) -> bool:
        """
        Initialize Vision Transformer model
        """
        if not VIT_AVAILABLE:
            logger.error("Vision Transformer not available. Install with 'pip install transformers'")
            return False
        
        try:
            # Initialize processor and model
            self.processor = ViTImageProcessor.from_pretrained(self.model_name)
            self.model = ViTModel.from_pretrained(self.model_name)
            self.model.eval()
            self.model = self.model.to(self.device)
            
            # Get embedding dimension
            self.embedding_dim = self.model.config.hidden_size
            
            self.initialized = True
            logger.info(f"Initialized Vision Transformer model: {self.model_name} (dim={self.embedding_dim}) on {self.device}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Vision Transformer: {str(e)}")
            return False
    
    def _init_resnet(self) -> bool:
        """
        Initialize ResNet model
        """
        if not RESNET_AVAILABLE:
            logger.error("ResNet not available. Install with 'pip install torch torchvision'")
            return False
        
        try:
            # Initialize model
            if self.model_name == "resnet18":
                self.model = models.resnet18(pretrained=True)
                self.embedding_dim = 512
            elif self.model_name == "resnet34":
                self.model = models.resnet34(pretrained=True)
                self.embedding_dim = 512
            elif self.model_name == "resnet50":
                self.model = models.resnet50(pretrained=True)
                self.embedding_dim = 2048
            elif self.model_name == "resnet101":
                self.model = models.resnet101(pretrained=True)
                self.embedding_dim = 2048
            elif self.model_name == "resnet152":
                self.model = models.resnet152(pretrained=True)
                self.embedding_dim = 2048
            else:
                logger.error(f"Unknown ResNet model: {self.model_name}")
                return False
            
            # Remove the final classification layer
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.model.eval()
            self.model = self.model.to(self.device)
            
            # Set up preprocessing
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            
            self.initialized = True
            logger.info(f"Initialized ResNet model: {self.model_name} (dim={self.embedding_dim}) on {self.device}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize ResNet: {str(e)}")
            return False
    
    def _process_image(self, image: Union[str, Image.Image, np.ndarray]) -> Image.Image:
        """
        Process the input image into a PIL Image
        
        Args:
            image: Input image (file path, PIL Image, base64 string, or numpy array)
            
        Returns:
            PIL.Image.Image: Processed PIL Image
        """
        if isinstance(image, str):
            # Check if it's a file path or base64 string
            if os.path.exists(image):
                return Image.open(image).convert("RGB")
            elif image.startswith("data:image") or image.startswith("http"):
                # Handle base64 or URL
                if image.startswith("data:image"):
                    # Extract the base64 part
                    base64_data = image.split(",")[1]
                    image_data = base64.b64decode(base64_data)
                    return Image.open(io.BytesIO(image_data)).convert("RGB")
                else:
                    # Handle URL (requires requests)
                    import requests
                    response = requests.get(image, stream=True)
                    response.raise_for_status()
                    return Image.open(io.BytesIO(response.content)).convert("RGB")
            else:
                raise ValueError("Invalid image path or URL")
        elif isinstance(image, Image.Image):
            return image.convert("RGB")
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        else:
            raise ValueError("Unsupported image type. Must be file path, PIL Image, base64 string, or numpy array")
    
    def embed(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Generate embeddings for the input image
        
        Args:
            image: Input image (file path, PIL Image, base64 string, or numpy array)
            
        Returns:
            np.ndarray: Embedding vector
        """
        if not self.initialized:
            raise RuntimeError("Embedder not initialized. Call initialize() first.")
        
        try:
            # Process the image
            pil_image = self._process_image(image)
            
            # Generate embeddings based on the framework
            if self.framework == "clip":
                return self._embed_clip(pil_image)
            elif self.framework == "openai":
                return self._embed_openai(pil_image)
            elif self.framework == "gemini":
                return self._embed_gemini(pil_image)
            elif self.framework == "timm":
                return self._embed_timm(pil_image)
            elif self.framework == "vit":
                return self._embed_vit(pil_image)
            elif self.framework == "resnet":
                return self._embed_resnet(pil_image)
            else:
                raise NotImplementedError(f"Embedding for framework '{self.framework}' not implemented")
        except Exception as e:
            logger.error(f"Error generating embeddings with {self.framework}: {str(e)}")
            # Return zero vector as fallback
            return np.zeros(self.embedding_dim)
    
    def embed_batch(self, images: List[Union[str, Image.Image, np.ndarray]]) -> np.ndarray:
        """
        Generate embeddings for a batch of input images
        
        Args:
            images: List of input images (file paths, PIL Images, base64 strings, or numpy arrays)
            
        Returns:
            np.ndarray: Batch of embedding vectors
        """
        if not self.initialized:
            raise RuntimeError("Embedder not initialized. Call initialize() first.")
        
        try:
            # Process all images
            pil_images = [self._process_image(img) for img in images]
            
            # Generate embeddings based on the framework
            if self.framework == "clip":
                return self._embed_batch_clip(pil_images)
            elif self.framework == "openai":
                return self._embed_batch_openai(pil_images)
            elif self.framework == "gemini":
                return self._embed_batch_gemini(pil_images)
            elif self.framework == "timm":
                return self._embed_batch_timm(pil_images)
            elif self.framework == "vit":
                return self._embed_batch_vit(pil_images)
            elif self.framework == "resnet":
                return self._embed_batch_resnet(pil_images)
            else:
                # Fall back to default implementation
                return super().embed_batch(images)
        except Exception as e:
            logger.error(f"Error generating batch embeddings with {self.framework}: {str(e)}")
            # Return zero vectors as fallback
            return np.zeros((len(images), self.embedding_dim))
    
    def _embed_clip(self, image: Image.Image) -> np.ndarray:
        """
        Generate embeddings using CLIP
        """
        import torch
        
        # Preprocess and encode the image
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            # Normalize the features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Convert to numpy array
        embedding = image_features.cpu().numpy().flatten()
        return embedding
    
    def _embed_batch_clip(self, images: List[Image.Image]) -> np.ndarray:
        """
        Generate batch embeddings using CLIP
        """
        import torch
        
        # Preprocess all images
        image_inputs = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        
        # Encode all images
        with torch.no_grad():
            image_features = self.model.encode_image(image_inputs)
            # Normalize the features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Convert to numpy array
        embeddings = image_features.cpu().numpy()
        return embeddings
    
    def _embed_openai(self, image: Image.Image) -> np.ndarray:
        """
        Generate embeddings using OpenAI
        """
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Call OpenAI API
        response = self.model.embeddings.create(
            input=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}
                }
            ],
            model="clip"
        )
        
        # Extract embedding
        embedding = np.array(response.data[0].embedding)
        return embedding
    
    def _embed_batch_openai(self, images: List[Image.Image]) -> np.ndarray:
        """
        Generate batch embeddings using OpenAI
        """
        # Convert images to base64
        image_inputs = []
        for img in images:
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            image_inputs.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}
            })
        
        # Call OpenAI API
        response = self.model.embeddings.create(
            input=image_inputs,
            model="clip"
        )
        
        # Extract embeddings
        embeddings = np.array([item.embedding for item in response.data])
        return embeddings
    
    def _embed_gemini(self, image: Image.Image) -> np.ndarray:
        """
        Generate embeddings using Google Gemini
        """
        # Placeholder for Gemini embedding API
        # Actual implementation would use the Gemini embedding API
        embedding = np.random.randn(self.embedding_dim)  # Replace with actual API call
        return embedding / np.linalg.norm(embedding)  # Normalize
    
    def _embed_batch_gemini(self, images: List[Image.Image]) -> np.ndarray:
        """
        Generate batch embeddings using Google Gemini
        """
        # Placeholder for Gemini batch embedding API
        # Actual implementation would use the Gemini embedding API
        embeddings = np.random.randn(len(images), self.embedding_dim)  # Replace with actual API call
        # Normalize each embedding
        return np.array([e / np.linalg.norm(e) for e in embeddings])
    
    def _embed_timm(self, image: Image.Image) -> np.ndarray:
        """
        Generate embeddings using timm
        """
        import torch
        
        # Preprocess the image
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            features = self.model(img_tensor)
            # Flatten and normalize
            features = features.squeeze().cpu().numpy()
            features = features / np.linalg.norm(features)
        
        return features
    
    def _embed_batch_timm(self, images: List[Image.Image]) -> np.ndarray:
        """
        Generate batch embeddings using timm
        """
        import torch
        
        # Preprocess all images
        img_tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            features = self.model(img_tensors)
            # Flatten and normalize
            features = features.squeeze().cpu().numpy()
            if len(images) == 1:
                features = features.reshape(1, -1)
            # Normalize each embedding
            features = np.array([f / np.linalg.norm(f) for f in features])
        
        return features
    
    def _embed_vit(self, image: Image.Image) -> np.ndarray:
        """
        Generate embeddings using Vision Transformer
        """
        import torch
        
        # Preprocess the image
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use the [CLS] token as the image representation
            embedding = outputs.last_hidden_state[:, 0].cpu().numpy().flatten()
            embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def _embed_batch_vit(self, images: List[Image.Image]) -> np.ndarray:
        """
        Generate batch embeddings using Vision Transformer
        """
        import torch
        
        # Preprocess all images
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use the [CLS] token as the image representation
            embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
            # Normalize each embedding
            embeddings = np.array([e / np.linalg.norm(e) for e in embeddings])
        
        return embeddings
    
    def _embed_resnet(self, image: Image.Image) -> np.ndarray:
        """
        Generate embeddings using ResNet
        """
        import torch
        
        # Preprocess the image
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            features = self.model(img_tensor)
            # Flatten and normalize
            features = features.squeeze().cpu().numpy()
            features = features / np.linalg.norm(features)
        
        return features
    
    def _embed_batch_resnet(self, images: List[Image.Image]) -> np.ndarray:
        """
        Generate batch embeddings using ResNet
        """
        import torch
        
        # Preprocess all images
        img_tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            features = self.model(img_tensors)
            # Flatten and normalize
            features = features.squeeze().cpu().numpy()
            if len(images) == 1:
                features = features.reshape(1, -1)
            # Normalize each embedding
            features = np.array([f / np.linalg.norm(f) for f in features])
        
        return features

# Factory function to create embedder with the appropriate framework
def create_image_embedder(framework: str = "clip", **kwargs) -> ImageEmbedder:
    """
    Create an image embedder with the specified framework
    
    Args:
        framework: Embedding framework to use
        **kwargs: Additional parameters for the embedder
        
    Returns:
        ImageEmbedder: Initialized image embedder
    """
    return ImageEmbedder(framework=framework, **kwargs)

# Get available frameworks
def get_available_frameworks() -> Dict[str, bool]:
    """
    Get a dictionary of available embedding frameworks
    
    Returns:
        Dict[str, bool]: Dictionary mapping framework names to availability status
    """
    return {
        "clip": CLIP_AVAILABLE,
        "openai": OPENAI_AVAILABLE,
        "gemini": GEMINI_AVAILABLE,
        "timm": TIMM_AVAILABLE,
        "vit": VIT_AVAILABLE,
        "resnet": RESNET_AVAILABLE
    }