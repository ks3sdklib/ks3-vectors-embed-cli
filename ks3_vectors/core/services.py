"""Core services for KS3 Vectors operations."""

import json
import os
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

from rich.console import Console

from ks3_vectors.utils.config import create_ks3_vectors_client, get_user_agent


# Default embedding API base URL
DEFAULT_BASE_URL = "https://kspmas.ksyun.com/v1/"

# OpenAI-compatible embeddings endpoint path
OPENAI_EMBEDDINGS_PATH = "embeddings"


class EmbeddingAPIError(Exception):
    """Raised when the embedding API returns an error response."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Embedding API error (HTTP {status_code}): {message}")


class EmbeddingService:
    """HTTP embedding service using stdlib urllib."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        console: Optional[Console] = None,
        debug: bool = False,
        timeout: int = 120,
    ):
        self.base_url = (
            base_url
            or os.environ.get("KSC_BASE_URL")
            or DEFAULT_BASE_URL
        )
        if not self.base_url.endswith("/"):
            self.base_url += "/"
        self.api_key = api_key or os.environ.get("KSC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Embedding API key is required. Set the KSC_API_KEY "
                "environment variable or pass api_key explicitly."
            )
        self.console = console or Console()
        self.debug = debug
        self.timeout = timeout

        if self.debug and self.console:
            self.console.print(f"[dim] EmbeddingService initialized for: {self.base_url}[/dim]")

    def _debug_log(self, message: str):
        """Log debug message if debug mode is enabled."""
        if self.debug and self.console:
            self.console.print(f"[dim] {message}[/dim]")

    def _post(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send a JSON POST request and return the parsed response body.

        Args:
            url: Full request URL.
            payload: JSON-serializable request body.

        Returns:
            Parsed JSON response as a dict.

        Raises:
            EmbeddingAPIError: If the API returns a non-2xx status.
            ConnectionError: If the request fails at the network level.
        """
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        self._debug_log(f"POST {url}  payload_size={len(data)}")

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as exc:
            err_body = ""
            try:
                err_body = exc.read().decode("utf-8", errors="replace")
            except (OSError, ValueError):
                pass
            raise EmbeddingAPIError(exc.code, err_body or str(exc)) from exc
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Failed to connect to embedding API at {url}: {exc.reason}"
            ) from exc

    def embed(self, model, payload: Dict[str, Any], endpoint_path: str) -> List[Dict[str, Any]]:
        """Send a schema-built payload and extract embeddings via the model's schema."""
        url = f"{self.base_url}{endpoint_path}"
        body = self._post(url, payload)
        return model.extract_embedding(body)

    def embed_texts(
        self,
        texts: List[str],
        model_id: str,
        dimensions: Optional[int] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> List[List[float]]:
        """Embed text strings via the OpenAI-compatible /v1/embeddings endpoint."""
        payload: Dict[str, Any] = {"input": texts, "model": model_id}
        if dimensions is not None:
            payload["dimensions"] = dimensions
        if extra_params:
            payload.update(extra_params)

        result = self._post(f"{self.base_url}{OPENAI_EMBEDDINGS_PATH}", payload)
        data_items = sorted(result["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in data_items]

    def embed_image(
        self,
        image_base64: str,
        model_id: str,
        dimensions: Optional[int] = None,
    ) -> List[float]:
        """Embed an image via the OpenAI-compatible /v1/embeddings endpoint."""
        payload: Dict[str, Any] = {"input": [image_base64], "model": model_id}
        if dimensions is not None:
            payload["dimensions"] = dimensions

        result = self._post(f"{self.base_url}{OPENAI_EMBEDDINGS_PATH}", payload)
        return result["data"][0]["embedding"]


class KS3VectorService:
    """Service for KS3 vector-bucket operations."""

    def __init__(self, vectors_endpoint: str = None,
                 debug: bool = False, console=None):
        self.client = create_ks3_vectors_client(vectors_endpoint)
        self.debug = debug
        self.console = console

        if self.debug and self.console:
            self.console.print(f"[dim] KS3VectorService initialized[/dim]")
            self.console.print(f"[dim] User agent: {get_user_agent()}[/dim]")

    def _debug_log(self, message: str):
        """Log debug message if debug mode is enabled."""
        if self.debug and self.console:
            self.console.print(f"[dim] {message}[/dim]")

    def put_vectors_batch(self, bucket_name: str, index_name: str,
                         vectors: List[Dict[str, Any]]) -> List[str]:
        """Put multiple vectors into a KS3 vector index."""
        start_time = time.time()
        self._debug_log(f"Starting put_vectors_batch operation")
        self._debug_log(f"Bucket: {bucket_name}, Index: {index_name}")
        self._debug_log(f"Batch size: {len(vectors)} vectors")

        self.client.put_vectors(vectors, vector_bucket_name=bucket_name,
                               index_name=index_name)

        elapsed_time = time.time() - start_time
        self._debug_log(f"KS3 put_vectors completed in {elapsed_time:.2f} seconds")

        vector_ids = [vector["key"] for vector in vectors]
        self._debug_log(f"Batch stored successfully with {len(vector_ids)} vectors")
        return vector_ids

    def query_vectors(self, bucket_name: str, index_name: str,
                      query_embedding: List[float], top_k: int = 5,
                      filter: Optional[Dict[str, Any]] = None,
                      return_metadata: bool = True,
                      return_distance: bool = True) -> List[Dict[str, Any]]:
        """Query vectors from a KS3 vector index."""
        result = self.client.query_vectors(
            {"float32": query_embedding}, top_k,
            vector_bucket_name=bucket_name, index_name=index_name,
            filter=filter, return_metadata=return_metadata,
            return_distance=return_distance,
        )

        results = []
        for vector in result.vectors:
            results.append({
                'vectorId': vector.key,
                'similarity': vector.distance,
                'metadata': vector.metadata,
            })
        return results

    def get_index(self, bucket_name: str, index_name: str):
        """Get index info (including dimension) from a KS3 vector index."""
        result = self.client.get_index(vector_bucket_name=bucket_name,
                                       index_name=index_name)
        return result.index
