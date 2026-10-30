import json
from typing import Optional, Dict, Any

import click
from rich.console import Console
from rich.table import Table
from ks3_vectors.core.unified_processor import UnifiedProcessor
from ks3_vectors.core.services import KS3VectorService, EmbeddingService
from ks3_vectors.utils.config import create_ks3_client
from ks3_vectors.utils.models import get_model_info, validate_model_modality, validate_user_parameters, prepare_processing_input, determine_content_type
from ks3_vectors.utils.multimodal_helpers import is_ks3_uri


def _validate_query_inputs(text_value, text, image, video, multi_images, model, model_id):
    """Validate query input parameters."""
    inputs = [text_value, text, image, video, multi_images]
    provided_inputs = [
        inp for inp in inputs
        if inp is not None and (not hasattr(inp, "__len__") or len(inp) > 0)
    ]

    if len(provided_inputs) == 0:
        raise click.ClickException(
            "No query input provided. Use one of: --text-value, --text, --image, or --video"
        )

    if not multi_images or len(multi_images) == 0:
        multi_images = None

    # Text-only models do not support image/video input.
    if model is None and (image or video):
        raise click.ClickException(
            f"Model {model_id} does not support image/video input. "
            f"For multimodal input, use qwen3-vl-embedding."
        )

    # Multimodal input is allowed only for models that support it.
    supports_multi = model is not None and model.supports_multimodal_input()
    is_multimodal_input = supports_multi and sum(
        1 for inp in [text_value, text, image, video, multi_images] if inp is not None
    ) >= 2

    if len(provided_inputs) > 1 and not is_multimodal_input:
        raise click.ClickException(
            "Multiple query inputs provided. Use only one input type, except for multimodal queries with supported models (--text-value + --image)"
        )

    return is_multimodal_input


def _needs_ks3_client(text, image, video):
    """True if any single-file input is a ks3:// URI (object read needs a client)."""
    return any(isinstance(p, str) and is_ks3_uri(p) for p in (text, image, video))


def _format_query_results(results: Dict[str, Any], output_format: str, console: Console):
    """Format and display query results."""
    if output_format == "table":
        _display_results_table(results, console)
    else:
        console.print_json(data=results)


def _display_results_table(results: Dict[str, Any], console: Console):
    """Display query results in table format."""
    table = Table(title="Query Results")
    table.add_column("Rank", style="cyan")
    table.add_column("Vector Key", style="green")
    table.add_column("Distance", style="yellow")
    table.add_column("Metadata", style="blue")

    query_results = results.get("results", [])
    for i, result in enumerate(query_results, 1):
        key = result.get("key", "N/A")
        distance = f"{result.get('distance', 0):.4f}" if result.get('distance') is not None else "N/A"
        metadata = json.dumps(result.get("metadata", {}), indent=2) if result.get("metadata") else "None"

        table.add_row(str(i), key, distance, metadata)

    console.print(table)

    # Display summary
    summary = results.get("summary", {})
    console.print(f"\nQuery Summary:")
    console.print(f"  Model: {summary.get('model', 'N/A')}")
    console.print(f"  Results Found: {summary.get('resultsFound', 0)}")
    console.print(f"  Query Dimensions: {summary.get('queryDimensions', 'N/A')}")


@click.command()
@click.option('--vector-bucket-name', required=True, help='KS3 vector bucket name')
@click.option('--index-name', required=True, help='Vector index name')
@click.option('--model-id', required=True, help='Embedding model ID')
@click.option('--text-value', help='Direct text query string')
@click.option('--text', help='Text file path (local file or ks3 URI)')
@click.option('--image', help='Image file path (local, ks3 URI, or public URL)')
@click.option('--video', help='Video public URL (local files not supported)')
@click.option('--top-k', default=5, type=int, help='Number of results to return (default: 5)')
@click.option('--filter', 'filter_expr', help='Filter expression for results (JSON format with operators, e.g., \'{"$and": [{"category": "docs"}, {"version": "1.0"}]}\')')
@click.option('--return-distance/--no-return-distance', default=True, help='Return similarity distances in results (default: true)')
@click.option('--return-metadata/--no-return-metadata', default=True, help='Return metadata in results (default: true)')
@click.option('--embedding-inference-params', help='JSON string with model-specific inference parameters')
@click.option('--output', type=click.Choice(['table', 'json']), default='json', help='Output format (default: json)')
@click.pass_context
def embed_query(ctx, vector_bucket_name, index_name, model_id, text_value, text, image, video,
                top_k, filter_expr, return_distance, return_metadata,
                embedding_inference_params, output):
    """Embed query input and search for similar vectors.

    \b
    QUERY INPUT TYPES:
    • Direct text: --text-value "search for this text"
    • Local text file: --text /path/to/query.txt
    • Local image file: --image /path/to/image.jpg
    • Public image/video URL: --image/--video https://...
    • Multimodal (supported models only): --text-value "..." --image ...

    \b
    MODELS:
    • qwen3-vl-embedding supports text, image, and video input.
    • Any other model ID uses the OpenAI-compatible /v1/embeddings endpoint (text only).

    \b
    FILTERING:
    • Use JSON format with KS3 Vectors API operators
    • Single condition: --filter '{"category": {"$eq": "documentation"}}'
    • Multiple conditions (AND): --filter '{"$and": [{"category": "docs"}, {"version": "1.0"}]}'
    • Multiple conditions (OR): --filter '{"$or": [{"category": "docs"}, {"category": "guides"}]}'
    • Comparison operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin
    """
    console = Console()

    multi_images = None

    try:
        # model is None means the OpenAI-compatible endpoint (text only).
        model = get_model_info(model_id)

        # Validate inputs
        is_multimodal = _validate_query_inputs(text_value, text, image, video, multi_images, model, model_id)

        # Determine content type for model validation
        content_type = determine_content_type(text_value, text, image, video, multi_images, is_multimodal)

        # Validate model capabilities (supported models only)
        if model is not None:
            if is_multimodal:
                if not model.supports_multimodal_input():
                    raise click.ClickException(f"Model {model_id} does not support multimodal input (text + image)")
            else:
                validate_model_modality(model_id, content_type)

        # Parse user parameters
        user_params = {}
        if embedding_inference_params:
            try:
                user_params = json.loads(embedding_inference_params)
            except json.JSONDecodeError as e:
                raise click.ClickException(f"Invalid JSON in --embedding-inference-params: {e}")

        # Validate user parameters against system-controlled parameters.
        if user_params and model is not None:
            try:
                system_keys = model.get_system_keys(content_type)
                system_payload = {key: None for key in system_keys}
                validate_user_parameters(system_payload, user_params)
            except ValueError as e:
                raise click.ClickException(str(e))

        # OpenAI-compatible endpoint: protect input and model from override.
        if user_params and model is None:
            conflicts = {"input", "model"}.intersection(user_params.keys())
            if conflicts:
                raise click.ClickException(
                    f"Cannot override system-controlled parameters: {sorted(conflicts)}. "
                    f"These parameters are automatically set based on your CLI inputs."
                )

        # Parse filter expression
        metadata_filter = None
        if filter_expr:
            try:
                metadata_filter = json.loads(filter_expr)
            except json.JSONDecodeError as e:
                raise click.ClickException(f"Invalid JSON in --filter: {e}")

        debug = ctx.obj.get('debug', False)
        embedding_service = EmbeddingService(base_url=ctx.obj.get('base_url'), console=console, debug=debug)
        vector_service = KS3VectorService(
            vectors_endpoint=ctx.obj.get('vectors_endpoint'),
            debug=debug, console=console)

        # Create a KS3 object-storage client when a query input reads ks3://
        ks3_client = create_ks3_client(ctx.obj.get('endpoint')) if _needs_ks3_client(text, image, video) else None

        # Create UnifiedProcessor
        processor = UnifiedProcessor(embedding_service, vector_service, ks3_client=ks3_client)

        # Fetch index dimensions once at the top level
        try:
            index_info = vector_service.get_index(vector_bucket_name, index_name)
            index_dimensions = index_info.dimension
            if not index_dimensions:
                raise click.ClickException(f"Could not determine dimensions for index {index_name}")
        except Exception as e:
            raise click.ClickException(f"Failed to get index information: {str(e)}")

        # Create ProcessingInput (no metadata, keys, or object naming for queries)
        processing_input = prepare_processing_input(text_value, text, image, video, multi_images, is_multimodal, None, None, False, None)

        # Auto-presign ks3:// video so users need no --presign-url for queries.
        query_presign = bool(video and is_ks3_uri(video))
        with console.status("[bold green]Generating query embedding..."):
            result = processor.process(
                model_id, processing_input, user_params, index_dimensions, query_presign
            )

        # Extract query embedding
        if hasattr(result, 'vectors') and result.vectors:
            # Get the embedding from the first vector
            first_vector = result.vectors[0]
            if "embedding" in first_vector:
                query_embedding = first_vector["embedding"]
            elif "data" in first_vector and "float32" in first_vector["data"]:
                query_embedding = first_vector["data"]["float32"]
            else:
                raise click.ClickException(f"No embedding found in result. Available keys: {list(first_vector.keys())}")
        else:
            raise click.ClickException("Failed to generate query embedding - no vectors returned")

        # Perform vector similarity search
        with console.status("[bold green]Searching for similar vectors..."):
            search_results = vector_service.query_vectors(
                bucket_name=vector_bucket_name,
                index_name=index_name,
                query_embedding=query_embedding,
                top_k=top_k,
                filter=metadata_filter,
                return_metadata=return_metadata,
                return_distance=return_distance
            )

        # Format results
        formatted_results = {
            "results": [
                {
                    "key": result.get("vectorId", ""),
                    "distance": result.get("similarity", 0.0),
                    "metadata": result.get("metadata", {})
                }
                for result in search_results
            ],
            "summary": {
                "queryType": content_type,
                "model": model_id,
                "index": index_name,
                "resultsFound": len(search_results),
                "queryDimensions": len(query_embedding),
            }
        }

        # Add distances if requested (already included in results)
        if not return_distance:
            for result in formatted_results["results"]:
                result.pop("distance", None)

        # Display results
        _format_query_results(formatted_results, output, console)

    except Exception as e:
        raise click.ClickException(str(e))
