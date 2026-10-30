# KS3 Vectors Embed CLI

KS3 Vectors Embed CLI is a command-line tool for KingsoftCloud KS3 vector buckets. It vectorizes text, images, and videos into a KS3 vector index and performs similarity search through simple commands.

## Supported Commands

**ks3-vectors-embed put**: Embed text, file content, or KS3 objects and store them as vectors in a KS3 vector index.
You can create and ingest vector embeddings into a KS3 vector index using a single put command. You specify the data input you want to create an embedding for, an embedding model ID, your KS3 vector bucket name, and KS3 vector index name. The command supports several input formats including text data, a local text or image file, a KS3 text or image object or prefix. The command generates embeddings using the dimensions configured in your KS3 vector index properties. If you are ingesting embeddings for several objects in a KS3 prefix or local file path, it automatically uses batch processes to maximize throughput.

**Note**: Each file is processed as a single embedding. Document chunking is not currently supported.

**ks3-vectors-embed query**: Embed a query input and search for similar vectors in a KS3 vector index.
You can perform similarity queries for vector embeddings in your KS3 vector index using a single query command. You specify your query input, an embedding model ID, the vector bucket name, and vector index name. The command accepts several types of query inputs like a text string, an image file, or a single KS3 text or image object. The command generates embeddings for your query using the input embeddings model and then performs a similarity search to find the most relevant matches. You can control the number of results returned, apply metadata filters to narrow your search, and choose whether to include similarity distance in the results for comprehensive analysis.

## Installation and Configuration

### Prerequisites

Make sure the following are ready before using this tool:

- Python 3.9 or higher.
- KS3 service enabled, with AccessKey ID and AccessKey Secret obtained.
- KingsoftCloud Starflow Platform enabled, with an API Key obtained.
- A vector bucket and a vector index created.

> **Important**: The vector index dimension must match the embedding model's output dimension, otherwise writes will fail.

### Quick Install (Recommended)

```bash
pip install ks3-vectors-embed-cli
```

### Development Install

```bash
git clone https://github.com/ks3sdklib/ks3-vectors-embed-cli.git
cd ks3-vectors-embed-cli
pip install -e .
```

### Verify installation

```bash
ks3-vectors-embed --version
# Output: ks3-vectors-embed, version 0.1.0
```

### Configure credentials

Configure credentials as environment variables. The CLI reads them automatically, so you don't need to pass them in every command.

```bash
# KS3 access key (shared by object reads and vector buckets, required)
export KS3_ACCESS_KEY_ID="AK"
export KS3_ACCESS_KEY_SECRET="SK"

# Object storage and vector bucket endpoints (optional, defaults below)
export KS3_ENDPOINT="ks3-cn-beijing.ksyuncs.com"
export KS3_VECTORS_ENDPOINT="ks3vectors-cn-beijing.ksyuncs.com"

# Embedding service (API Key required, endpoint optional, defaults below)
export KSC_API_KEY="APIKey"
export KSC_BASE_URL="https://kspmas.ksyun.com/v1/"
```

Environment variables:

| Variable | Description | Required |
|---|---|---|
| `KS3_ACCESS_KEY_ID` | KS3 access key ID, shared by object reads and vector buckets | Yes |
| `KS3_ACCESS_KEY_SECRET` | KS3 access key secret, shared by object reads and vector buckets | Yes |
| `KS3_SECURITY_TOKEN` | Temporary security token, only for STS temporary credentials | No |
| `KS3_ENDPOINT` | Object storage endpoint, default `ks3-cn-beijing.ksyuncs.com` | No |
| `KS3_VECTORS_ENDPOINT` | Vector bucket endpoint, default `ks3vectors-cn-beijing.ksyuncs.com` | No |
| `KSC_API_KEY` | Embedding service API Key | Yes |
| `KSC_BASE_URL` | Embedding service endpoint, default `https://kspmas.ksyun.com/v1/` | No |

### Uninstall

Uninstall the tool with:

```bash
pip uninstall ks3-vectors-embed-cli
```

## Quick Start

### put

```bash
# Embed text and store it
ks3-vectors-embed put \
  --vector-bucket-name <bucket> \
  --index-name <index> \
  --model-id qwen3-vl-embedding \
  --text-value "hello world"

# Embed a local text file
ks3-vectors-embed put \
  --vector-bucket-name <bucket> \
  --index-name <index> \
  --model-id qwen3-vl-embedding \
  --text ./docs/readme.txt

# Embed a KS3 text object
ks3-vectors-embed put \
  --vector-bucket-name <bucket> \
  --index-name <index> \
  --model-id qwen3-vl-embedding \
  --text ks3://bucket/docs/readme.txt

# Batch from a local glob
ks3-vectors-embed put \
  --vector-bucket-name <bucket> \
  --index-name <index> \
  --model-id qwen3-vl-embedding \
  --text "./docs/*.txt" \
  --max-workers 8

# Embed an image: local file (auto base64)
ks3-vectors-embed put \
  --vector-bucket-name <bucket> \
  --index-name <index> \
  --model-id qwen3-vl-embedding \
  --image ./cat.jpg

# Server-side resize a ks3:// image before embedding (put only, 50%)
ks3-vectors-embed put \
  --vector-bucket-name <bucket> \
  --index-name <index> \
  --model-id qwen3-vl-embedding \
  --image ks3://bucket/imgs/cat.png \
  --image-resize 50
```

### query

```bash
# Search for similar vectors
ks3-vectors-embed query \
  --vector-bucket-name <bucket> \
  --index-name <index> \
  --model-id qwen3-vl-embedding \
  --text-value "hello world" \
  --top-k 3

# Query with a metadata filter
ks3-vectors-embed query \
  --vector-bucket-name <bucket> \
  --index-name <index> \
  --model-id qwen3-vl-embedding \
  --text-value "hello world" \
  --filter '{"category": {"$eq": "docs"}}' \
  --top-k 5
```
