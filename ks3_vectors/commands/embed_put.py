import os
import json
import click
from rich.progress import Progress, SpinnerColumn, TextColumn

from ks3_vectors.core.services import KS3VectorService, EmbeddingService
from ks3_vectors.core.unified_processor import UnifiedProcessor
from ks3_vectors.utils.config import create_ks3_client
from ks3_vectors.utils.models import get_model_info, validate_user_parameters, prepare_processing_input, determine_content_type
from ks3_vectors.utils.multimodal_helpers import is_ks3_uri
from ks3_vectors.core.streaming_batch_orchestrator import StreamingBatchOrchestrator


def _create_progress_context(console):
    """Create progress context for operations."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    )


def _validate_inputs(text_value, text, image, video, multi_images, model, model_id, key, filename_as_key):
    """Validate input parameters."""
    inputs_provided = sum(bool(x) for x in [text_value, text, image, video, multi_images])

    if inputs_provided == 0:
        raise click.ClickException("At least one input must be provided: --text-value, --text, --image, or --video")

    # Text-only models do not support image/video input.
    if model is None and (image or video or multi_images):
        raise click.ClickException(
            f"Model {model_id} does not support image/video input. "
            f"For multimodal input, use qwen3-vl-embedding."
        )

    # Check mutual exclusivity of key parameters
    if key and filename_as_key:
        raise click.ClickException("Cannot use both --key and --filename-as-key. Choose one.")

    # --filename-as-key not allowed with --text-value (no file/object to extract name from)
    if filename_as_key and text_value:
        raise click.ClickException("--filename-as-key is not supported with --text-value (no file or object to extract name from)")

    # Key parameters not supported for video (multi-vector output)
    if (key or filename_as_key) and video:
        raise click.ClickException("--key and --filename-as-key are not supported for video inputs (multi-vector output)")

    # Multimodal input is allowed only for models that support it.
    supports_multi = model is not None and model.supports_multimodal_input()
    is_multimodal_input = supports_multi and sum(
        1 for inp in [text_value, text, image, video, multi_images] if inp is not None
    ) >= 2

    if inputs_provided > 1 and not is_multimodal_input:
        raise click.ClickException("Only one input type can be specified at a time, except for multimodal input with supported models")

    return is_multimodal_input


def _needs_ks3_client(text, image, video):
    """True if any single-file input is a ks3:// URI (object read needs a client)."""
    return any(isinstance(p, str) and is_ks3_uri(p) for p in (text, image, video))


@click.command()
@click.option('--vector-bucket-name', required=True, help='KS3 vector bucket name')
@click.option('--index-name', required=True, help='Vector index name')
@click.option('--model-id', required=True, help='Embedding model ID')
@click.option('--text-value', help='Direct text input to embed')
@click.option('--text', help='Text file path (local or ks3 URI)')
@click.option('--image', help='Image file path (local, ks3 URI, or public URL)')
@click.option('--image-resize', type=click.IntRange(10, 100), help='Scale ks3:// images to this percentage (10-100, server-side resize)')
@click.option('--video', help='Video public URL (local files not supported)')
@click.option('--embedding-inference-params', help='Embedding inference parameters (JSON)')
@click.option('--key', help='Custom vector key (auto-generated UUID if not provided)')
@click.option('--key-prefix', help='Prefix to prepend to all vector keys')
@click.option('--filename-as-key', is_flag=True, help='Use filename as vector key')
@click.option('--metadata', help='Additional metadata (JSON)')
@click.option('--max-workers', type=int, default=4, help='Maximum parallel workers for batch processing')
@click.option('--batch-size', type=click.IntRange(1, 500), default=500, help='Number of vectors per KS3 Vector put_vectors call (1-500, default: 500)')
@click.option('--output', type=click.Choice(['json', 'table']), default='json', help='Output format')
@click.option('--presign-url', is_flag=True, help='For ks3:// image/video inputs, serve a pre-signed URL to the embedding service instead of downloading the object')
@click.pass_context
def embed_put(ctx, vector_bucket_name, index_name, model_id, text_value, text, image,
              image_resize, video, embedding_inference_params, key, key_prefix, filename_as_key,
              metadata, max_workers, batch_size, output, presign_url):
    """Embed content and store vectors. Unknown model IDs use the OpenAI-compatible endpoint."""

    console = ctx.obj['console']
    debug = ctx.obj.get('debug', False)

    multi_images = None

    # model is None means the OpenAI-compatible endpoint (text only).
    model = get_model_info(model_id)

    # Parse parameters
    user_params = {}
    if embedding_inference_params:
        try:
            user_params = json.loads(embedding_inference_params)
        except json.JSONDecodeError:
            raise click.ClickException("Invalid JSON in --embedding-inference-params parameter")

    metadata_dict = {}
    if metadata:
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            raise click.ClickException("Invalid JSON in --metadata parameter")

    # Validate user parameters against system-controlled parameters.
    if user_params and model is not None:
        try:
            content_type = determine_content_type(text_value, text, image, video, multi_images)
            system_keys = model.get_system_keys(content_type)
            system_payload = {key: None for key in system_keys}  # Dummy values for validation
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

    # Validate inputs
    is_multimodal = _validate_inputs(text_value, text, image, video, multi_images, model, model_id, key, filename_as_key)

    try:
        # Initialize services
        embedding_service = EmbeddingService(
            base_url=ctx.obj.get('base_url'), console=console, debug=debug)
        vector_service = KS3VectorService(
            vectors_endpoint=ctx.obj.get('vectors_endpoint'),
            debug=debug, console=console)

        # Create a KS3 object-storage client when an input reads ks3://
        ks3_client = create_ks3_client(ctx.obj.get('endpoint')) if _needs_ks3_client(text, image, video) else None

        # Create unified processor
        processor = UnifiedProcessor(embedding_service, vector_service, ks3_client=ks3_client)

        # Fetch index dimensions once at the top level
        try:
            index_info = vector_service.get_index(vector_bucket_name, index_name)
            index_dimensions = index_info.dimension
            if not index_dimensions:
                raise click.ClickException(f"Could not determine dimensions for index {index_name}")
        except Exception as e:
            raise click.ClickException(f"Failed to get index information: {str(e)}")

        # Prepare processing input
        processing_input = prepare_processing_input(
            text_value, text, image, video, multi_images, is_multimodal, metadata_dict, key, filename_as_key, key_prefix
        )

        # Check if processing_input is None
        if processing_input is None:
            raise click.ClickException("No valid input provided. Please specify one of: --text-value, --text, --image, or --video")

        # Check for wildcard patterns (streaming batch processing)
        if (processing_input.content_type in ["text", "image", "video"] and
            "file_path" in processing_input.data):
            file_path = processing_input.data["file_path"]
            if '*' in file_path or ('?' in file_path and not file_path.startswith('http')):
                return _process_streaming_batch(
                    file_path, processing_input.content_type, vector_bucket_name, index_name,
                    model_id, metadata_dict, user_params,
                    processor, console, output, max_workers, batch_size, index_dimensions, processing_input.filename_as_key, processing_input.key_prefix, presign_url, image_resize
                )

        with _create_progress_context(console) as progress:
            # Process with unified pipeline
            progress.add_task("Processing input...", total=None)
            result = processor.process(
                model_id, processing_input, user_params, index_dimensions, presign_url, image_resize
            )

            # Store vectors with batch_size handling
            progress.add_task(f"Storing {len(result.vectors)} vector(s)...", total=None)

            # Handle batch_size for single file processing too
            vector_count = len(result.vectors)

            if vector_count <= batch_size:
                stored_keys = processor.store_vectors(result.vectors, vector_bucket_name, index_name)
            else:
                stored_keys = []
                for i in range(0, vector_count, batch_size):
                    chunk = result.vectors[i:i + batch_size]
                    chunk_keys = processor.store_vectors(chunk, vector_bucket_name, index_name)
                    stored_keys.extend(chunk_keys)

        # Prepare output
        if result.result_type == "multiclip":
            output_result = {
                'type': 'multiclip',
                'bucket': vector_bucket_name,
                'index': index_name,
                'model': model_id,
                'contentType': processing_input.content_type,
                'totalVectors': len(stored_keys),
                'keys': stored_keys
            }
        else:
            output_result = {
                'key': stored_keys[0],
                'bucket': vector_bucket_name,
                'index': index_name,
                'model': model_id,
                'contentType': processing_input.content_type,
                'embeddingDimensions': index_dimensions,
                'metadata': result.vectors[0]['metadata']
            }

        if output == "table":
            _display_put_table(output_result, console)
        else:
            console.print_json(data=output_result)

    except Exception as e:
        raise click.ClickException(str(e))


if __name__ == '__main__':
    embed_put()


def _process_streaming_batch(file_path, content_type, vector_bucket_name, index_name,
                             model_id, metadata_dict, user_params,
                             processor, console, output, max_workers, batch_size, index_dimensions, filename_as_key, key_prefix, presign_url=False, image_resize=None):
    """Process wildcard pattern using efficient streaming batch orchestrator."""

    try:
        # Create streaming batch orchestrator
        streaming_orchestrator = StreamingBatchOrchestrator(processor, max_workers, batch_size)

        console.print(f"Starting streaming batch processing: {file_path}")

        # Process using streaming approach (no pre-loading of file paths)
        batch_result = streaming_orchestrator.process_streaming_batch(
            file_path, content_type, vector_bucket_name, index_name, model_id,
            metadata_dict, user_params, index_dimensions, filename_as_key, key_prefix, presign_url, image_resize
        )

        # Display results
        result_dict = {
            "type": "streaming_batch",
            "bucket": vector_bucket_name,
            "index": index_name,
            "model": model_id,
            "contentType": content_type,
            "totalFiles": batch_result.processed_count + batch_result.failed_count,
            "processedFiles": batch_result.processed_count,
            "failedFiles": batch_result.failed_count,
            "totalVectors": len(batch_result.processed_keys),  # Count actual vectors
            "vectorKeys": batch_result.processed_keys[:10] if batch_result.processed_keys else []  # Show first 10
        }

        if batch_result.errors:
            result_dict["errors"] = batch_result.errors[:10]  # Show first 10 errors

        if output == "table":
            _display_batch_table(result_dict, console)
        else:
            console.print_json(data=result_dict)

        # Print display limit messages after output
        if len(batch_result.processed_keys) > 10:
            console.print(f"[dim]Note: Showing first 10 of {len(batch_result.processed_keys)} vector keys[/dim]")

        if batch_result.errors and len(batch_result.errors) > 10:
            console.print(f"[dim]Note: Showing first 10 of {len(batch_result.errors)} errors[/dim]")

        return result_dict if output == "json" else None

    except Exception as e:
        console.print(f"[red]Streaming batch processing failed: {str(e)}[/red]")
        raise click.ClickException(f"Streaming batch processing failed: {str(e)}")


def _display_put_table(result, console):
    """Display single-file put results in table format."""
    from rich.table import Table

    if result.get("type") == "multiclip":
        table = Table(title="Put Results")
        table.add_column("Key", style="cyan")
        table.add_column("Bucket", style="green")
        table.add_column("Index", style="green")
        table.add_column("Content Type", style="green")
        table.add_column("Total Vectors", style="yellow")
        for key in result["keys"]:
            table.add_row(key, result["bucket"], result["index"], result["contentType"], str(result["totalVectors"]))
        console.print(table)
        console.print(f"\n[green]Successfully wrote {result['totalVectors']} vector(s).[/green]")
    else:
        table = Table(title="Put Results")
        table.add_column("Key", style="cyan")
        table.add_column("Bucket", style="green")
        table.add_column("Index", style="green")
        table.add_column("Content Type", style="green")
        table.add_column("Dimensions", style="yellow")
        table.add_row(
            result["key"], result["bucket"], result["index"],
            result["contentType"], str(result["embeddingDimensions"])
        )
        console.print(table)
        console.print(f"\n[green]Successfully wrote 1 vector.[/green]")


def _display_batch_table(result, console):
    """Display batch results in table format."""
    from rich.table import Table

    table = Table(title="Batch Processing Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Files", str(result["totalFiles"]))
    table.add_row("Processed Files", str(result["processedFiles"]))
    table.add_row("Failed Files", str(result["failedFiles"]))
    table.add_row("Total Vectors", str(result["totalVectors"]))
    table.add_row("Model", result["model"])
    table.add_row("Content Type", result["contentType"])

    console.print(table)
