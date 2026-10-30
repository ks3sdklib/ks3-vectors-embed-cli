"""Helper utilities for reading source objects and building vector metadata."""

import base64
import os
from typing import Any, Dict, Tuple


def parse_ks3_uri(uri: str) -> Tuple[str, str]:
    """Parse a ks3:// URI into a (bucket, key) tuple.

    Args:
        uri: KS3 URI in format ks3://bucket/key.

    Returns:
        Tuple of (bucket_name, object_key).

    Raises:
        ValueError: If the URI is not a valid ks3:// URI (missing prefix,
            missing bucket, or missing key).
    """
    if not uri.startswith("ks3://"):
        raise ValueError(
            f"Invalid KS3 URI: '{uri}'. Expected format: ks3://bucket/key"
        )

    # Remove the ks3:// prefix
    path = uri[6:]  # len("ks3://") == 6
    parts = path.split("/", 1)

    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"Invalid KS3 URI: '{uri}'. Expected format: ks3://bucket/key"
        )

    return parts[0], parts[1]


def is_ks3_uri(path: str) -> bool:
    """Check if a path is a ks3:// URI.

    Args:
        path: Path string to check.

    Returns:
        True if path starts with ks3://.
    """
    return path.startswith("ks3://")


def is_http_url(path: str) -> bool:
    """Check if a path is an HTTP/HTTPS URL.

    Args:
        path: Path string to check.

    Returns:
        True if path starts with http:// or https://.
    """
    return path.startswith(("http://", "https://"))


def read_local_text(filepath: str) -> str:
    """Read text content from a local file path.

    Args:
        filepath: Local file path.

    Returns:
        File text content.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def read_text_from_url(url: str) -> str:
    """Read text content from an HTTP/HTTPS URL.

    Args:
        url: HTTP/HTTPS URL.

    Returns:
        Downloaded text content.

    Raises:
        RuntimeError: If download fails.
    """
    import urllib.request

    try:
        with urllib.request.urlopen(url) as response:
            return response.read().decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to download content from {url}: {e}") from e


def read_image_as_base64(filepath: str) -> str:
    """Read a local image file and return base64-encoded string.

    Args:
        filepath: Local image file path.

    Returns:
        Base64-encoded string of the image content.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image file not found: {filepath}")

    with open(filepath, "rb") as f:
        content = f.read()

    return base64.b64encode(content).decode("utf-8")


def get_mime_type(filepath: str) -> str:
    """Detect MIME type from an image file path.

    Args:
        filepath: File path or name.

    Returns:
        MIME type string ('image/jpeg' or 'image/png').
    """
    lower = filepath.lower()
    if lower.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if lower.endswith(".png"):
        return "image/png"
    return "image/jpeg"  # default


def create_source_metadata(
    content_type: str,
    data: Dict[str, Any],
    source_location: str,
) -> Dict[str, Any]:
    """Build standard KS3VECTORS-EMBED-SRC-* source metadata for a vector.

    Args:
        content_type: Content type ('text', 'image', 'video', 'multimodal').
        data: The processing input's data dict (file_path / text / multimodal).
        source_location: Source file path or URI.

    Returns:
        Metadata dict with standardized keys (only set when applicable).
    """
    metadata: Dict[str, Any] = {}

    if "file_path" in data:
        # File input (--text/--image/--video) - add content (text only), type, location
        content_type_formatted = content_type.upper().replace("_", "-")
        # For text files, also add the raw text content
        if content_type == "text" and "text" in data:
            metadata["KS3VECTORS-EMBED-SRC-CONTENT"] = data["text"]
        metadata["KS3VECTORS-EMBED-SRC-CONTENT-TYPE"] = content_type_formatted
        metadata["KS3VECTORS-EMBED-SRC-LOCATION"] = source_location
    elif content_type == "multimodal":
        # Multimodal input -- add content (text only), type, location
        multimodal_data = data.get("multimodal", {})
        content_type_formatted = content_type.upper()
        if multimodal_data.get("text", ""):
            metadata["KS3VECTORS-EMBED-SRC-CONTENT"] = multimodal_data.get("text", "")
        metadata["KS3VECTORS-EMBED-SRC-CONTENT-TYPE"] = content_type_formatted
        if multimodal_data.get("image", ""):
            metadata["KS3VECTORS-EMBED-SRC-LOCATION"] = multimodal_data.get("image", "")
        if multimodal_data.get("video", ""):
            metadata["KS3VECTORS-EMBED-SRC-LOCATION"] = multimodal_data.get("video", "")
    else:
        # Direct text input (--text-value) - add content + location
        if content_type == "text" and "text" in data:
            metadata["KS3VECTORS-EMBED-SRC-CONTENT"] = data["text"]
        content_type_formatted = content_type.upper()
        metadata["KS3VECTORS-EMBED-SRC-CONTENT-TYPE"] = content_type_formatted
        metadata["KS3VECTORS-EMBED-SRC-LOCATION"] = source_location

    return metadata
