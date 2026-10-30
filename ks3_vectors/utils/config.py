"""KS3 configuration utilities."""

import os

from ks3.connection import Connection
from ks3.vectors import VectorsClient
from ks3.signers import V4Signer

from ks3_vectors.__version__ import __version__


def _region_from_endpoint(endpoint: str) -> str:
    """Derive the region (uppercased) from the vector-bucket endpoint for V4 signing, defaulting to BEIJING on failure."""
    prefix = "ks3vectors-cn-"
    idx = endpoint.find(prefix)
    if idx == -1:
        return "BEIJING"
    rest = endpoint[idx + len(prefix):]
    region = rest.split(".")[0]
    if region.endswith("-internal"):
        region = region[: -len("-internal")]
    return region.upper()


def create_ks3_vectors_client(endpoint: str = None):
    """Create a KS3 vector-bucket client (ks3sdk VectorsClient) for put/query against the ks3vectors service."""
    endpoint = endpoint or os.environ.get("KS3_VECTORS_ENDPOINT") or "ks3vectors-cn-beijing.ksyuncs.com"
    missing = [v for v in ("KS3_ACCESS_KEY_ID", "KS3_ACCESS_KEY_SECRET") if not os.environ.get(v)]
    if missing:
        raise ValueError(
            "KS3 vector operations require these environment variables (missing: "
            f"{', '.join(missing)}): KS3_ACCESS_KEY_ID, KS3_ACCESS_KEY_SECRET"
        )

    # KS3_SECURITY_TOKEN is optional (STS temporary credentials only)
    security_token = os.environ.get("KS3_SECURITY_TOKEN") or None
    region = _region_from_endpoint(endpoint)
    return VectorsClient(
        os.environ["KS3_ACCESS_KEY_ID"],
        os.environ["KS3_ACCESS_KEY_SECRET"],
        host=endpoint,
        security_token=security_token,
        is_secure=False,
        signer=V4Signer(region=region),
    )


def create_ks3_client(endpoint: str = None):
    """Create a KS3 object-storage client (ks3sdk Connection) for reading ks3:// URIs."""
    host = endpoint or os.environ.get("KS3_ENDPOINT") or "ks3-cn-beijing.ksyuncs.com"
    missing = [v for v in ("KS3_ACCESS_KEY_ID", "KS3_ACCESS_KEY_SECRET") if not os.environ.get(v)]
    if missing:
        raise ValueError(
            "Reading ks3:// inputs requires these environment variables (missing: "
            f"{', '.join(missing)}): KS3_ACCESS_KEY_ID, KS3_ACCESS_KEY_SECRET"
        )

    # KS3_SECURITY_TOKEN is optional (STS temporary credentials only)
    security_token = os.environ.get("KS3_SECURITY_TOKEN") or None
    return Connection(
        os.environ["KS3_ACCESS_KEY_ID"],
        os.environ["KS3_ACCESS_KEY_SECRET"],
        host=host,
        security_token=security_token,
    )


def get_user_agent():
    """
    Get the CLI user agent string for logging/debugging.

    Returns:
        User agent string for display
    """
    return f"ks3-vectors-embed-cli/{__version__}"
