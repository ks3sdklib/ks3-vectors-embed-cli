"""Streaming batch orchestrator for efficient processing of large file sets."""

import os
import glob
import threading
from typing import List, Dict, Any, Generator, Tuple
from pathlib import Path
from dataclasses import dataclass
from ks3_vectors.core.unified_processor import UnifiedProcessor, ProcessingInput
from ks3_vectors.utils.multimodal_helpers import is_ks3_uri


@dataclass
class BatchResult:
    """Result of batch processing operation."""
    processed_count: int
    failed_count: int
    processed_keys: List[str]
    errors: List[str] = None


class StreamingBatchOrchestrator:
    """Efficient streaming batch processor that handles files in chunks without loading all paths into memory."""
    
    def __init__(self, unified_processor: UnifiedProcessor, max_workers: int = 4, batch_size: int = 500):
        self.unified_processor = unified_processor
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.batch_lock = threading.Lock()
        
    def process_streaming_batch(self, file_pattern: str, content_type: str,
                                vector_bucket_name: str, index_name: str, model_id: str,
                                metadata: Dict[str, Any],
                                user_params: Dict[str, Any] = None,
                                index_dimensions: int = None, filename_as_key: bool = False, key_prefix: str = None,
                                presign_url: bool = False, image_resize: int = None) -> BatchResult:
        """Process files using streaming approach - no memory loading of all file paths."""

        # Detect processing strategy
        strategy = self._detect_processing_strategy(file_pattern)

        # Use pre-fetched index dimensions (no API call needed)
        if index_dimensions is None:
            raise ValueError("Failed to fetch index dimensions from KS3 vector index. Cannot proceed without actual index dimensions.")

        batch_context = {
            'model_id': model_id,
            'index_dimensions': index_dimensions,
            'user_params': user_params or {},
            'batch_size': self.batch_size,
            'filename_as_key': filename_as_key,
            'key_prefix': key_prefix,
            'presign_url': presign_url,
            'image_resize': image_resize
        }

        # Process using appropriate streaming method
        if strategy == "ks3_streaming":
            return self._process_ks3_streaming(file_pattern, content_type, vector_bucket_name,
                                               index_name, metadata, batch_context)
        else:  # local_streaming
            return self._process_local_streaming(file_pattern, content_type, vector_bucket_name,
                                               index_name, metadata, batch_context)
    
    def _detect_processing_strategy(self, file_pattern: str) -> str:
        """Check whether it is for KS3 or Local files."""
        if is_ks3_uri(file_pattern) and '*' in file_pattern:
            return "ks3_streaming"
        elif '*' in file_pattern or '?' in file_pattern:
            return "local_streaming"
        else:
            raise ValueError(f"Invalid pattern for batch processing: {file_pattern}. Pattern must contain wildcards (* or ?) for batch processing.")
    
    def _process_ks3_streaming(self, ks3_pattern: str, content_type: str,
                               vector_bucket_name: str, index_name: str,
                               metadata: Dict[str, Any], batch_context: Dict[str, Any]) -> BatchResult:
        """Stream KS3 objects directly without pre-loading all paths."""

        if self.unified_processor.ks3_client is None:
            raise ValueError(
                "Batch processing of ks3:// patterns requires a ks3_client, which was not provided."
            )

        # Parse KS3 pattern
        ks3_path = ks3_pattern[6:]  # Remove 'ks3://'
        if ks3_path.endswith('/*'):
            ks3_path = ks3_path[:-1]  # Remove only '*', keep '/'
        elif ks3_path.endswith('*'):
            ks3_path = ks3_path[:-1]  # Remove '*'
            # Add trailing slash if not present to ensure directory-level filtering
            if not ks3_path.endswith('/'):
                ks3_path += '/'

        bucket, prefix = ks3_path.split('/', 1) if '/' in ks3_path else (ks3_path, '')

        iter_keys = self.unified_processor.ks3_client.get_bucket(bucket).list(prefix=prefix, delimiter="")

        total_processed = 0
        total_failed = 0
        all_processed_keys = []
        all_errors = []

        chunk_number = 0
        for chunk_files in self._stream_ks3_chunks(iter_keys, bucket):
            if not chunk_files:
                continue

            chunk_number += 1
            print(f"Processing chunk {chunk_number}...")

            # Process chunk
            processed, failed, keys, errors = self._process_chunk(
                chunk_files, content_type, vector_bucket_name, index_name, metadata, batch_context
            )

            total_processed += processed
            total_failed += failed
            all_processed_keys.extend(keys)
            if errors:
                all_errors.extend(errors)

            print(f"Completed chunk {chunk_number}: {processed} processed, {failed} failed")

        return BatchResult(
            processed_count=total_processed,
            failed_count=total_failed,
            processed_keys=all_processed_keys,
            errors=all_errors if all_errors else None
        )
    
    def _stream_ks3_chunks(self, keys_iter, bucket: str) -> Generator[List[str], None, None]:
        """Generator that yields chunks of KS3 object paths."""
        chunk = []
        text_extensions = {'txt', 'md', 'json', 'csv', 'log'}
        image_extensions = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp'}
        video_extensions = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}
        multi_images_extensions = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp'}
        all_extensions = text_extensions | image_extensions | video_extensions | multi_images_extensions

        for key_obj in keys_iter:
            key = key_obj.name
            # Skip directories (e.g. the placeholder for an empty "folder")
            if key.endswith('/'):
                continue

            # Check file extension
            extension = key.lower().split('.')[-1] if '.' in key else ''
            if extension in all_extensions:
                chunk.append(f"ks3://{bucket}/{key}")

                # Yield chunk when it reaches target size
                if len(chunk) >= self.batch_size:
                    yield chunk
                    chunk = []

        # Yield remaining files
        if chunk:
            yield chunk
    
    def _process_local_streaming(self, local_pattern: str, content_type: str,
                                vector_bucket_name: str, index_name: str, 
                                metadata: Dict[str, Any], batch_context: Dict[str, Any]) -> BatchResult:
        """Stream local files directly without pre-loading all paths."""
        
        total_processed = 0
        total_failed = 0
        all_processed_keys = []
        all_errors = []
        
        print(f"Starting streaming processing of {local_pattern}")
        
        chunk_number = 0
        for chunk_files in self._stream_local_chunks(local_pattern):
            if not chunk_files:
                continue
                
            chunk_number += 1
            print(f"Processing chunk {chunk_number}...")
            
            # Process chunk
            processed, failed, keys, errors = self._process_chunk(
                chunk_files, content_type, vector_bucket_name, index_name, metadata, batch_context
            )
            
            total_processed += processed
            total_failed += failed
            all_processed_keys.extend(keys)
            if errors:
                all_errors.extend(errors)
            
            print(f"Completed chunk {chunk_number}: {processed} processed, {failed} failed")
        
        return BatchResult(
            processed_count=total_processed,
            failed_count=total_failed,
            processed_keys=all_processed_keys,
            errors=all_errors if all_errors else None
        )
    
    def _stream_local_chunks(self, pattern: str) -> Generator[List[str], None, None]:
        """Generator that yields chunks of local file paths using iterator."""
        chunk = []
        text_extensions = {'txt', 'md', 'json', 'csv', 'log'}
        image_extensions = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp'}
        video_extensions = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}
        multi_images_extensions = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp'}
        all_extensions = text_extensions | image_extensions | video_extensions | multi_images_extensions
        
        # Use iglob for memory-efficient iteration
        for file_path in glob.iglob(pattern, recursive=True):
            if os.path.isfile(file_path):
                path_obj = Path(file_path)
                extension = path_obj.suffix.lower()[1:]  # Remove the dot
                
                if extension in all_extensions:
                    chunk.append(file_path)
                    
                    # Yield chunk when it reaches target size
                    if len(chunk) >= self.batch_size:
                        yield chunk
                        chunk = []
        
        # Yield remaining files
        if chunk:
            yield chunk
    
    def _process_chunk(self, chunk_files: List[str], content_type: str,
                      vector_bucket_name: str, index_name: str,
                      metadata: Dict[str, Any], batch_context: Dict[str, Any]) -> Tuple[int, int, List[str], List[str]]:
        """Process a chunk of files using UnifiedProcessor."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        processed_files_count = 0  # Count files, not vectors
        failed_files_count = 0     # Count files, not vectors
        processed_keys = []
        errors = []

        total_files = len(chunk_files)
        print(f"Processing {total_files} {content_type} files with {self.max_workers} concurrent workers...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {}

            # Submit all files for processing
            for file_path in chunk_files:
                processing_input = ProcessingInput(
                    content_type, {"file_path": file_path}, file_path, metadata,
                    filename_as_key=batch_context['filename_as_key'],
                    key_prefix=batch_context['key_prefix']
                )

                future = executor.submit(
                    self.unified_processor.process,
                    batch_context['model_id'],
                    processing_input,
                    batch_context['user_params'],
                    batch_context['index_dimensions'],
                    batch_context['presign_url'],
                    batch_context['image_resize']
                )
                future_to_file[future] = file_path

            # Collect results as they complete
            completed_files = 0
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                completed_files += 1

                try:
                    result = future.result()

                    if result and result.vectors:
                        # Store vectors from this file, chunking if needed
                        try:
                            file_vector_count = len(result.vectors)
                            batch_size = batch_context['batch_size']

                            if file_vector_count <= batch_size:
                                # Small file - single API call
                                self.unified_processor.vector_service.put_vectors_batch(
                                    vector_bucket_name, index_name, result.vectors
                                )
                            else:
                                # Large file - chunk into batch_size pieces
                                for i in range(0, file_vector_count, batch_size):
                                    chunk = result.vectors[i:i + batch_size]
                                    self.unified_processor.vector_service.put_vectors_batch(
                                        vector_bucket_name, index_name, chunk
                                    )

                            processed_files_count += 1  # Count files, not vectors
                            processed_keys.extend([v.get("key", "") for v in result.vectors])

                            print(f"[{completed_files}/{total_files}] Stored {file_vector_count} vectors from {file_path}")

                        except Exception as e:
                            failed_files_count += 1
                            errors.append(f"Storage failed for {file_path}: {str(e)}")

                    else:
                        failed_files_count += 1
                        errors.append(f"No vectors generated for {file_path}")

                except Exception as e:
                    failed_files_count += 1
                    errors.append(f"Error processing {file_path}: {str(e)}")

        return processed_files_count, failed_files_count, processed_keys, errors
