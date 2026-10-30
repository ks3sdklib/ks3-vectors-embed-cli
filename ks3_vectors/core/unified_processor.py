"""Unified processing pipeline for embedding and vector storage."""

import base64
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ks3_vectors.utils.models import (
    get_model_info,
    SupportedModel,
    ProcessingInput,
    generate_vector_key,
)
from ks3_vectors.utils.multimodal_helpers import (
    create_source_metadata,
    get_mime_type,
    is_http_url,
    is_ks3_uri,
    parse_ks3_uri,
    read_image_as_base64,
    read_local_text,
    read_text_from_url,
)


@dataclass
class ProcessingResult:
    """Unified result structure."""
    vectors: List[Dict[str, Any]]  # List of vectors to store
    result_type: str  # "single" or "multiclip"
    raw_results: Optional[List[Dict[str, Any]]] = None  # Raw results for timing extraction


class UnifiedProcessor:
    """Unified processor that routes embedding requests by model."""

    def __init__(self, embedding_service, vector_service, ks3_client=None):
        """Initialize with embedding service, vector service, and optional KS3 object client."""
        self.embedding_service = embedding_service
        self.vector_service = vector_service
        self.ks3_client = ks3_client

    def process(self, model_id: str, processing_input: ProcessingInput,
                user_params: Dict[str, Any] = None,
                precomputed_dimensions: int = None,
                presign_url: bool = False,
                image_resize: int = None) -> ProcessingResult:
        """Process input into vectors, routing by model_id."""
        if precomputed_dimensions is None:
            raise ValueError("Unexpected error occurred. Index dimensions are not fetched.")

        user_params = user_params or {}
        model = get_model_info(model_id)

        # Step 1: Build content for schema application
        content = self._prepare_content(processing_input, precomputed_dimensions, user_params, presign_url, image_resize)

        # Step 2: Get embeddings (KS3 payload schema or OpenAI-compatible endpoint)
        if model is not None:
            raw_results = self._embed_schema(model, processing_input, content, user_params)
        else:
            raw_results = self._embed_openai_compatible(model_id, processing_input, content,
                                                        user_params, precomputed_dimensions)

        # Step 3: Process results into vectors
        vectors = self._prepare_vectors(raw_results, processing_input, model)

        # Step 4: Determine result type
        result_type = "multiclip" if len(vectors) > 1 else "single"

        return ProcessingResult(vectors=vectors, result_type=result_type, raw_results=raw_results)

    def _embed_schema(self, model: SupportedModel, processing_input: ProcessingInput,
                      content: Dict[str, Any], user_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Embed using the model's KS3-format payload schema."""
        final_payload = model.build_payload(processing_input.content_type, content, user_params)
        return self.embedding_service.embed(model, final_payload, model.capabilities.endpoint_path)

    def _embed_openai_compatible(self, model_id: str, processing_input: ProcessingInput,
                                 content: Dict[str, Any], user_params: Dict[str, Any],
                                 dimensions: int) -> List[Dict[str, Any]]:
        """Embed via the OpenAI-compatible endpoint."""
        # This endpoint supports text and image only.
        contents = content.get("contents", [])
        text_inputs = [c["text"] for c in contents if "text" in c]
        image_inputs = [c["image"] for c in contents if "image" in c]

        # User-specified dimensions override the index default.
        final_dimensions = user_params.get("dimensions", dimensions)
        extra_params = {k: v for k, v in user_params.items() if k not in ("dimensions",)}

        results: List[Dict[str, Any]] = []
        if text_inputs:
            embeddings = self.embedding_service.embed_texts(
                text_inputs, model_id, dimensions=final_dimensions, extra_params=extra_params or None
            )
            results.extend({"embedding": e} for e in embeddings)
        for img in image_inputs:
            emb = self.embedding_service.embed_image(img, model_id, dimensions=final_dimensions)
            results.append({"embedding": emb})
        return results

    def _prepare_content(self, processing_input: ProcessingInput, index_dimensions: int,
                         user_params: Dict[str, Any] = None,
                         presign_url: bool = False,
                         image_resize: int = None) -> Dict[str, Any]:
        """Prepare content dictionary with index dimensions and contents."""
        content = {"index": {"dimensions": index_dimensions}}

        if processing_input.content_type == "text":
            content["contents"] = [{"text": self._prepare_text_content(processing_input)}]
        elif processing_input.content_type == "image":
            content["contents"] = [{"image": self._prepare_image_content(processing_input, presign_url, image_resize)}]
        elif processing_input.content_type == "video":
            content["contents"] = [{"video": self._prepare_video_content(processing_input, presign_url)}]
        elif processing_input.content_type == "multimodal":
            content["contents"] = self._prepare_multimodal_content(processing_input, presign_url, image_resize)
        return content

    def _prepare_text_content(self, processing_input: ProcessingInput) -> str:
        """Return text content from direct input or a local/http file."""
        if "file_path" in processing_input.data and "text" not in processing_input.data:
            file_content = self._read_file_content(processing_input.data["file_path"])
            # Cache so metadata logic can access the text content
            processing_input.data["text"] = file_content
            return file_content
        return processing_input.data.get("text", "")

    def _prepare_image_content(self, processing_input: ProcessingInput, presign_url: bool = False,
                               image_resize: int = None) -> str:
        """Return image content as a public URL or base64 data URI."""
        if "file_path" in processing_input.data:
            file_path = processing_input.data["file_path"]
            if image_resize is not None and not is_ks3_uri(file_path):
                raise ValueError("--image-resize is only supported for ks3:// image inputs")
            if presign_url and is_ks3_uri(file_path):
                return self._generate_ks3_presign_url(file_path)
            if is_http_url(file_path):
                return file_path
            # Local image -> base64 data URI (KS3 accepts data:image/{format};base64,{data})
            return self._image_to_data_uri(file_path, image_resize)
        return processing_input.data.get("image", "")

    def _prepare_video_content(self, processing_input: ProcessingInput, presign_url: bool = False) -> str:
        """Return video content. KS3 requires a publicly accessible URL (no base64)."""
        file_path = processing_input.data.get("file_path", "")
        if presign_url and is_ks3_uri(file_path):
            return self._generate_ks3_presign_url(file_path)
        if is_http_url(file_path):
            return file_path
        if is_ks3_uri(file_path):
            raise ValueError("ks3:// video requires --presign-url.")
        raise ValueError("Local video files are not supported, use a URL or ks3:// URI.")

    def _prepare_multimodal_content(self, processing_input: ProcessingInput,
                                    presign_url: bool = False,
                                    image_resize: int = None) -> List[Dict[str, str]]:
        """Build a contents list mixing text/image/video blocks."""
        multimodal_data = processing_input.data.get("multimodal", {})
        contents: List[Dict[str, str]] = []

        text = multimodal_data.get("text", "")
        if text:
            contents.append({"text": text})
        image_path = multimodal_data.get("image", "")
        if image_path:
            if image_resize is not None and not is_ks3_uri(image_path):
                raise ValueError("--image-resize is only supported for ks3:// image inputs")
            if presign_url and is_ks3_uri(image_path):
                contents.append({"image": self._generate_ks3_presign_url(image_path)})
            elif is_http_url(image_path):
                contents.append({"image": image_path})
            else:
                contents.append({"image": self._image_to_data_uri(image_path, image_resize)})
        video_path = multimodal_data.get("video", "")
        if video_path:
            if presign_url and is_ks3_uri(video_path):
                contents.append({"video": self._generate_ks3_presign_url(video_path)})
            elif is_http_url(video_path):
                contents.append({"video": video_path})
            else:
                raise ValueError(
                    "Video input must be a publicly accessible URL. "
                    f"Local video files are not supported (got: {video_path!r})."
                )
        return contents

    def _read_file_content(self, file_path: str) -> str:
        """Read text file content from local, ks3://, or http(s) URL."""
        if is_ks3_uri(file_path):
            return self._read_ks3_object(file_path).decode("utf-8")
        elif is_http_url(file_path):
            return read_text_from_url(file_path)
        else:
            return read_local_text(file_path)

    def _image_to_data_uri(self, file_path: str, image_resize: int = None) -> str:
        """Read an image (local or ks3://) and return a base64 data URI."""
        if is_ks3_uri(file_path):
            image_bytes = self._read_ks3_object(file_path, image_resize)
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
        else:
            base64_image = read_image_as_base64(file_path)

        mime_type = get_mime_type(file_path)
        return f"data:{mime_type};base64,{base64_image}"

    def _read_ks3_object(self, ks3_path: str, image_resize: int = None) -> bytes:
        """Read raw bytes from a ks3:// URI."""
        if self.ks3_client is None:
            raise ValueError(
                "Reading ks3:// inputs requires a ks3_client, which was not provided."
            )
        bucket, key = parse_ks3_uri(ks3_path)
        if image_resize is not None:
            # Server-side resize via x-kss-process; fetch the scaled bytes directly.
            resp = self.ks3_client.make_request(
                "GET", bucket=bucket, key=key,
                query_args={"x-kss-process": f"image/resize,p_{image_resize}"},
            )
            if resp.status < 200 or resp.status >= 300:
                raise ValueError(f"Failed to read/resized ks3://{bucket}/{key}: HTTP {resp.status}")
            return resp.read()
        key_obj = self.ks3_client.get_bucket(bucket).get_key(key)
        if key_obj is None:
            raise ValueError(f"Object not found: ks3://{bucket}/{key}")
        return key_obj.get_contents_as_string()

    def _generate_ks3_presign_url(self, ks3_path: str) -> str:
        """Generate a pre-signed URL for a ks3:// object."""
        if self.ks3_client is None:
            raise ValueError(
                "Generating a pre-signed URL for ks3:// inputs requires a ks3_client, "
                "which was not provided."
            )
        bucket, key = parse_ks3_uri(ks3_path)
        return self.ks3_client.get_bucket(bucket).get_key(key).generate_url(900)

    def _prepare_vectors(self, raw_results: List[Dict[str, Any]],
                         processing_input: ProcessingInput,
                         model: Optional[SupportedModel]) -> List[Dict[str, Any]]:
        """Convert raw embedding results to vector storage format."""
        vectors = []

        for i, result in enumerate(raw_results):
            embedding = result.get("embedding", [])
            if not embedding:
                raise ValueError(
                    f"Missing required embedding in result {i+1}/{len(raw_results)}. Result: {result}"
                )

            # Generate vector key based on processing input preferences
            if processing_input.custom_key and len(raw_results) == 1:
                vector_key = generate_vector_key(
                    processing_input.custom_key, False,
                    processing_input.source_location, processing_input.key_prefix)
            elif processing_input.filename_as_key and len(raw_results) == 1:
                vector_key = generate_vector_key(
                    None, True, processing_input.source_location, processing_input.key_prefix)
            else:
                vector_key = generate_vector_key(
                    None, False, processing_input.source_location, processing_input.key_prefix)

            # Prepare metadata
            vector_metadata = processing_input.metadata.copy()
            vector_metadata.update(create_source_metadata(
                processing_input.content_type,
                processing_input.data,
                processing_input.source_location,
            ))

            # Create vector in KS3 vector format
            vector = {
                "key": vector_key,
                "data": {
                    "float32": embedding
                },
                "metadata": vector_metadata
            }

            vectors.append(vector)

        return vectors

    def store_vectors(self, vectors: List[Dict[str, Any]], vector_bucket_name: str,
                      index_name: str) -> List[str]:
        """Store vectors using batch operation."""
        if not vectors:
            return []

        self.vector_service.put_vectors_batch(vector_bucket_name, index_name, vectors)
        return [v["key"] for v in vectors]
