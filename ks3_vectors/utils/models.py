"""Model definitions and capabilities for KS3 Vectors CLI."""

import uuid
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import click


@dataclass
class ProcessingInput:
    """Unified input structure for processing."""
    content_type: str  # "text", "image", "video", "multi_images", "multimodal"
    data: Dict[str, Any]  # Content data
    source_location: str  # Original source location
    metadata: Dict[str, Any]  # Base metadata
    custom_key: Optional[str] = None  # Custom vector key
    filename_as_key: bool = False  # Use filename as vector key
    key_prefix: Optional[str] = None  # Prefix to prepend to all vector keys


def determine_content_type(text_value, text, image, video, multi_images, is_multimodal=False, batch_text_url=None) -> str:
    """Determine content type from CLI parameters."""
    if is_multimodal:
        return "multimodal"
    if video:
        return "video"
    if multi_images and len(multi_images) > 0:
        return "multi_images"
    if image:
        return "image"
    if text or text_value:
        return "text"
    if batch_text_url:
        return "batch_text"
    raise ValueError("No input type specified")


def prepare_processing_input(text_value, text, image, video, multi_images, is_multimodal, metadata_dict=None, custom_key=None, filename_as_key=False, key_prefix=None, batch_text_url=None) -> ProcessingInput:
    """Prepare unified processing input for both PUT and QUERY operations."""
    metadata = metadata_dict or {}
    
    if is_multimodal:
        return ProcessingInput(
            content_type="multimodal",
            data={"multimodal": {"text": text_value if text_value is not None else text, "image": image, "video": video, "multi_images": multi_images}},
            source_location=image,  # Use image path as primary source location
            metadata=metadata,
            custom_key=custom_key,
            filename_as_key=filename_as_key,
            key_prefix=key_prefix
        )
    elif text_value:
        return ProcessingInput(
            content_type="text",
            data={"text": text_value},
            source_location="direct_text_input",
            metadata=metadata,
            custom_key=custom_key,
            filename_as_key=filename_as_key,
            key_prefix=key_prefix
        )
    elif text:
        return ProcessingInput(
            content_type="text",
            data={"file_path": text},
            source_location=text,
            metadata=metadata,
            custom_key=custom_key,
            filename_as_key=filename_as_key,
            key_prefix=key_prefix
        )
    elif batch_text_url:
        return ProcessingInput(
            content_type="batch_text",
            data={"batch_text_url": batch_text_url},
            source_location=batch_text_url,
            metadata=metadata,
            custom_key=custom_key,
            filename_as_key=filename_as_key,
            key_prefix=key_prefix
        )
    elif image:
        return ProcessingInput(
            content_type="image",
            data={"file_path": image},
            source_location=image,
            metadata=metadata,
            custom_key=custom_key,
            filename_as_key=filename_as_key,
            key_prefix=key_prefix
        )
    elif video:
        return ProcessingInput(
            content_type="video",
            data={"file_path": video},
            source_location=video,
            metadata=metadata,
            custom_key=custom_key,
            filename_as_key=filename_as_key,
            key_prefix=key_prefix
        )
    elif multi_images:
        # Handle both single string and tuple of strings for multi_images
        if isinstance(multi_images, tuple):
            multi_images_data = list(multi_images)
        else:
            multi_images_data = multi_images
            
        return ProcessingInput(
            content_type="multi_images",
            data={"file_path": multi_images_data},
            source_location=str(multi_images_data),
            metadata=metadata,
            custom_key=custom_key,
            filename_as_key=filename_as_key,
            key_prefix=key_prefix
        )
    else:
        raise click.ClickException("No valid input provided")


@dataclass
class ModelCapabilities:
    """Capabilities and properties of an embedding model."""
    is_async: bool
    supported_modalities: List[str]  # text, image, video, multi_images
    description: str
    supports_multimodal_input: bool = False  # Can accept multiple modalities simultaneously
    max_local_file_size: int = None  # Maximum local file size in bytes for async models (None = no limit)
    
    # Schema-based payload and response definitions
    payload_schema: Dict[str, Any] = None
    response_embedding_path: str = None  # Path to extract embedding from response
    endpoint_path: str = "embeddings"  # Model-specific API path (e.g. multimodal-embedding)


class SupportedModel(Enum):
    """Enumeration of supported embedding models with their capabilities."""

    QWEN3_VL_EMBEDDING = ("qwen3-vl-embedding", ModelCapabilities(
        is_async=False,
        endpoint_path="multimodal-embedding",
        supported_modalities=["text", "image", "video"],
        description="KS3 Qwen3-VL multimodal embedding",
        supports_multimodal_input=True,
        payload_schema={
            "model": "{model_id}",
            "input": {"contents": "{content.contents}"},
            "parameters": {"dimension": "{index.dimensions}"}
        },
        response_embedding_path="output.embeddings"
    ))

    def __init__(self, model_id: str, capabilities: ModelCapabilities):
        self.model_id = model_id
        self.capabilities = capabilities
    
    @classmethod
    def from_model_id(cls, model_id: str) -> Optional['SupportedModel']:
        """Get SupportedModel enum from model ID string."""
        for model in cls:
            if model.model_id == model_id:
                return model
        return None
    
    def is_async(self) -> bool:
        """Check if model requires async processing."""
        return self.capabilities.is_async
    
    def get_system_keys(self, content_type: str) -> List[str]:
        """Extract top-level keys from payload schema without building payload."""
        schema = self.capabilities.payload_schema
        if isinstance(schema, dict) and content_type in schema:
            schema = schema[content_type]
        return list(schema.keys()) if isinstance(schema, dict) else []
    
    def supports_modality(self, modality: str) -> bool:
        """Check if model supports a specific modality."""
        return modality in self.capabilities.supported_modalities
    
    def supports_multimodal_input(self) -> bool:
        """Check if model supports multiple modalities simultaneously."""
        return self.capabilities.supports_multimodal_input
    
    def build_payload(self, content_type: str, content: dict, user_params: dict = None) -> dict:
        """Build model-specific payload using schema."""
        user_params = user_params or {}

        # Create context for schema substitution
        context = {
            "model_id": self.model_id,
            "content_type": content_type,
            "content": content,
            "index": content.get("index", {}),  # Flatten index to root level
            "user": user_params,
        }

        # Handle conditional schemas
        schema = self.capabilities.payload_schema
        if isinstance(schema, dict) and content_type in schema:
            # Use content_type-specific schema
            schema = schema[content_type]

        # Apply schema to get system payload
        system_payload = self._apply_schema(schema, context)

        # Deep merge user parameters into system payload
        return self._deep_merge(system_payload, user_params)
    
    def extract_embedding(self, response: dict) -> list:
        """Extract embedding from model response using schema."""
        return self._extract_by_path(response, self.capabilities.response_embedding_path)
    
    def _apply_schema(self, schema: Any, context: dict) -> Any:
        """Recursively apply context to schema template."""

        if isinstance(schema, dict):
            result = {}
            for key, value in schema.items():
                applied_value = self._apply_schema(value, context)
                if applied_value is not None:  # Skip None values
                    result[key] = applied_value
            return result
        elif isinstance(schema, list):
            return [self._apply_schema(item, context) for item in schema]
        elif isinstance(schema, str) and schema.startswith("{") and schema.endswith("}"):
            # Template substitution
            path = schema[1:-1]  # Remove { }
            return self._get_by_path(context, path)
        else:
            return schema
    
    def _deep_merge(self, system_payload: dict, user_params: dict) -> dict:
        """Deep merge user parameters into system payload."""

        result = system_payload.copy()
        
        for key, value in user_params.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                result[key] = self._deep_merge(result[key], value)
            else:
                # Add new key or overwrite non-dict values
                result[key] = value
        
        return result
    
    def _get_by_path(self, obj: dict, path: str) -> Any:
        """Get value from nested dict by dot notation path."""
        parts = path.split(".")
        current = obj
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None  # Skip optional parameters
        return current
    
    def _extract_by_path(self, obj: dict, path: str) -> Any:
        """Extract value from response using path like 'embeddings[0]' or 'embeddingsByType.*|embedding'."""
        try:
            # Handle alternate paths separated by |
            if "|" in path:
                paths = path.split("|")
                for alt_path in paths:
                    try:
                        return self._extract_single_path(obj, alt_path.strip())
                    except:
                        continue
                # If all paths fail, raise error with the first path
                return self._extract_single_path(obj, paths[0].strip())
            else:
                return self._extract_single_path(obj, path)
        except Exception as e:
            raise ValueError(f"Failed to extract embedding from response using path '{path}': {e}. Response keys: {list(obj.keys()) if isinstance(obj, dict) else type(obj)}")
    
    def _extract_single_path(self, obj: dict, path: str) -> Any:
        """Extract value from response using a single path."""
        if path.endswith(".*"):
            # Handle dynamic object access like "embeddingsByType.*"
            key = path[:-2]
            if key in obj and isinstance(obj[key], list):
                # Get first value from the dictionary
                values = list(obj[key][0].values())
                return values[0] if values else []            # Remove ".*"
            elif key in obj and isinstance(obj[key], dict):
                # Get first value from the dictionary
                values = list(obj[key].values())
                return values[0] if values else []
            else:
                raise KeyError(f"Key '{key}' not found or not a dictionary")
        elif "[" in path:
            # Handle array access like "embeddings[0]" or "embeddings[0].embedding"
            bracket_pos = path.find("[")
            key = path[:bracket_pos]
            remainder = path[bracket_pos:]
            
            # Extract index part
            if "]" not in remainder:
                raise ValueError(f"Invalid array access format in path: {path}")
            
            bracket_end = remainder.find("]")
            index_part = remainder[1:bracket_end]
            index = int(index_part)
            
            # Check if there's more path after the array index
            if bracket_end + 1 < len(remainder):
                # Handle paths like "embeddings[0].embedding"
                rest_path = remainder[bracket_end + 1:]
                if rest_path.startswith("."):
                    rest_path = rest_path[1:]
                
                # Get the array element and continue processing
                array_element = obj[key][index]
                # Recursively process the rest of the path
                if rest_path:
                    return self._get_by_path(array_element, rest_path) if "." in rest_path else array_element[rest_path]
                else:
                    return array_element
            else:
                # Simple array access like "embeddings[0]"
                return obj[key][index]
        else:
            # Simple key access; support dotted nested paths like "output.embeddings"
            if "." in path:
                current = obj
                for part in path.split("."):
                    if isinstance(current, dict) and part in current:
                        current = current[part]
                    else:
                        raise KeyError(f"Key '{part}' not found in path '{path}'")
                return current
            return obj[path]


def validate_user_parameters(system_payload: Dict[str, Any], user_params: Dict[str, Any]) -> None:
    """Validate user parameters don't conflict with system parameters."""

    system_fields = set(system_payload.keys())  # Top-level only
    user_fields = set(user_params.keys())       # Top-level only

    # "parameters" holds user-tunable model options.
    conflicts = system_fields.intersection(user_fields) - {"parameters"}

    if conflicts:
        conflict_list = sorted(list(conflicts))
        raise ValueError(
            f"Cannot override system-controlled parameters: {conflict_list}. "
            f"These parameters are automatically set based on your CLI inputs."
        )


def get_model_info(model_id: str) -> Optional[SupportedModel]:
    """Get model information from model ID."""
    return SupportedModel.from_model_id(model_id)


def validate_model_modality(model_id: str, modality: str) -> None:
    """Validate that model supports the requested modality."""
    model = get_model_info(model_id)
    if not model:
        raise ValueError(f"Unsupported model: {model_id}")
    
    if not model.supports_modality(modality):
        supported = ", ".join(model.capabilities.supported_modalities)
        raise ValueError(
            f"Model {model_id} does not support {modality} input. "
            f"Supported modalities: {supported}"
        )


def generate_vector_key(custom_key: Optional[str], use_object_key_name: bool, source_location: str, key_prefix: Optional[str] = None) -> str:
    """Generate vector key based on parameters and source location."""
    if custom_key:
        base_key = custom_key
    elif use_object_key_name:
        base_key = extract_key_from_source(source_location)
    else:
        base_key = str(uuid.uuid4())
    
    # Apply key prefix if provided
    if key_prefix:
        return f"{key_prefix}{base_key}"
    else:
        return base_key


def extract_key_from_source(source_location: str) -> str:
    """Extract key from source location (KS3 URI or local path)."""
    if source_location.startswith('ks3://'):
        # Extract filename from KS3 object key
        parts = source_location[6:].split('/', 1)  # Remove 'ks3://' and split
        object_key = parts[1] if len(parts) > 1 else parts[0]
        return Path(object_key).name  # Get filename from object key
    else:
        # Extract filename from local path
        return Path(source_location).name
