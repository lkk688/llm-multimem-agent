# Embedder module initialization

# Use direct imports to avoid circular imports and make the package importable
# from both inside and outside the package

# Import base embedder
from .base import BaseEmbedder

# Import text embedder components
from .text_embedder import (
    TextEmbedder,
    create_text_embedder,
    get_available_frameworks as get_available_text_frameworks
)

# Import image embedder components
from .image_embedder import (
    ImageEmbedder,
    create_image_embedder,
    get_available_frameworks as get_available_image_frameworks
)

# Import audio embedder components
from .audio_embedder import (
    AudioEmbedder,
    create_audio_embedder,
    get_available_frameworks as get_available_audio_frameworks
)

# Import multimodal embedder components
from .multimodal_embedder import (
    MultiModalEmbedder,
    create_multimodal_embedder,
    get_available_frameworks as get_available_multimodal_frameworks
)


def create_embedder(modality: str = "text", **kwargs):
    """Create an embedder for the specified modality.
    
    Args:
        modality: The modality of the embedder ("text", "image", "audio", or "multimodal")
        **kwargs: Additional arguments to pass to the specific embedder creation function
    
    Returns:
        An embedder instance for the specified modality
    """
    if modality.lower() == "text":
        return create_text_embedder(**kwargs)
    elif modality.lower() == "image":
        return create_image_embedder(**kwargs)
    elif modality.lower() == "audio":
        return create_audio_embedder(**kwargs)
    elif modality.lower() == "multimodal":
        return create_multimodal_embedder(**kwargs)
    else:
        raise ValueError(f"Unknown modality: {modality}. Must be one of: text, image, audio, multimodal")


def get_available_embedders():
    """Get available embedding frameworks for all modalities.
    
    Returns:
        Dictionary with available frameworks for each modality
    """
    return {
        "text": get_available_text_frameworks(),
        "image": get_available_image_frameworks(),
        "audio": get_available_audio_frameworks(),
        "multimodal": get_available_multimodal_frameworks()
    }


__all__ = [
    "BaseEmbedder",
    "TextEmbedder",
    "ImageEmbedder",
    "AudioEmbedder",
    "MultiModalEmbedder",
    "create_text_embedder",
    "create_image_embedder",
    "create_audio_embedder",
    "create_multimodal_embedder",
    "create_embedder",
    "get_available_text_frameworks",
    "get_available_image_frameworks",
    "get_available_audio_frameworks",
    "get_available_multimodal_frameworks",
    "get_available_embedders"
]