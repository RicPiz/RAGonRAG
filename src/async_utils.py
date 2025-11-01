"""
Async utilities for concurrent operations in the RAG system.
"""

import asyncio
import os
import time
from typing import List, Dict, Any, Callable, Coroutine, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import aiohttp
import aiofiles

# Import core dependencies
try:
    from .config import get_config
    from .logger import get_logger, log_performance
except ImportError:
    from config import get_config
    from logger import get_logger, log_performance

logger = get_logger("async_utils")


class AsyncHTTPClient:
    """Async HTTP client for API calls."""

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.config = get_config()
        self._session_active = False

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=self.config.embedding.timeout)
        self.session = aiohttp.ClientSession(timeout=timeout)
        self._session_active = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session and not self.session.closed:
            await self.session.close()
        self._session_active = False

    def _check_session(self) -> None:
        """Check if session is properly initialized and active."""
        if not self.session:
            raise RuntimeError("HTTP client not initialized. Use as async context manager.")

        if not self._session_active:
            raise RuntimeError("HTTP client session is not active. Use as async context manager.")

        if self.session.closed:
            raise RuntimeError("HTTP client session has been closed.")

    async def post(self, url: str, headers: Dict[str, str], json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make async POST request with error parsing."""
        self._check_session()

        for attempt in range(self.config.embedding.max_retries):
            try:
                async with self.session.post(url, headers=headers, json=json_data) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:  # Rate limit
                        wait_time = 2 ** attempt
                        logger.warning(f"Rate limited, waiting {wait_time}s", attempt=attempt)
                        await asyncio.sleep(wait_time)
                        continue
                    elif response.status == 401:
                        # Authentication error
                        error_body = await response.text()
                        logger.error("Authentication failed", status=response.status, error=error_body)
                        raise RuntimeError("OpenAI API authentication failed. Check your API key.")
                    else:
                        # Parse error response body
                        try:
                            error_data = await response.json()
                            error_msg = error_data.get('error', {}).get('message', 'Unknown error')
                            logger.error("API request failed", status=response.status, error=error_msg)
                            raise RuntimeError(f"OpenAI API error ({response.status}): {error_msg}")
                        except:
                            error_text = await response.text()
                            logger.error("API request failed", status=response.status, error=error_text)
                            response.raise_for_status()
            except Exception as e:
                if attempt == self.config.embedding.max_retries - 1:
                    logger.error("HTTP request failed after retries",
                               url=url, exception=e)
                    raise
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)

        raise RuntimeError("Max retries exceeded")


class AsyncEmbeddingGenerator:
    """Async wrapper for embedding generation with batching and concurrency."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.model = model
        self.config = get_config()
        self.url = "https://api.openai.com/v1/embeddings"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts with batching."""
        if not texts:
            return []
        
        start_time = time.perf_counter()
        batch_size = self.config.embedding.batch_size
        all_embeddings = []
        
        async with AsyncHTTPClient() as client:
            # Process in batches
            batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
            
            # Create tasks for concurrent processing
            tasks = []
            for batch in batches:
                task = self._process_batch(client, batch)
                tasks.append(task)
            
            # Execute batches concurrently
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect results and handle exceptions
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error("Batch processing failed", exception=result)
                    raise result
                all_embeddings.extend(result)
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        log_performance("generate_embeddings", duration_ms, 
                       num_texts=len(texts), num_batches=len(batches))
        
        return all_embeddings
    
    async def _process_batch(self, client: AsyncHTTPClient, batch: List[str]) -> List[List[float]]:
        """Process a single batch of texts."""
        json_data = {
            "model": self.model,
            "input": batch
        }

        response = await client.post(self.url, self.headers, json_data)

        # Ensure we return float32 embeddings
        embeddings = [item["embedding"] for item in response["data"]]

        logger.debug("Processed embedding batch", batch_size=len(batch))
        return embeddings


class AsyncFileLoader:
    """Async file loading utilities."""
    
    @staticmethod
    async def load_text_file(file_path: str) -> str:
        """Load text file asynchronously."""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            logger.debug("Loaded text file", path=file_path, size=len(content))
            return content
        except Exception as e:
            logger.error("Failed to load file", path=file_path, exception=e)
            raise
    
    @staticmethod
    async def load_multiple_files(file_paths: List[str]) -> Dict[str, str]:
        """Load multiple files concurrently."""
        start_time = time.perf_counter()
        
        tasks = []
        for path in file_paths:
            task = AsyncFileLoader.load_text_file(path)
            tasks.append((path, task))
        
        results = {}
        for path, task in tasks:
            try:
                content = await task
                results[path] = content
            except Exception as e:
                logger.error("Failed to load file in batch", path=path, exception=e)
                # Continue with other files
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        log_performance("load_multiple_files", duration_ms, 
                       num_files=len(file_paths), loaded_files=len(results))
        
        return results


class AsyncTaskExecutor:
    """Execute CPU-bound tasks asynchronously using thread pool with proper context management."""

    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="rag-async")
        self._closed = False
        logger.info("Initialized async task executor", max_workers=self.max_workers)

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.aclose()

    def _check_not_closed(self):
        """Check if executor is still open."""
        if self._closed:
            raise RuntimeError("AsyncTaskExecutor is closed")

    async def run_in_thread(self, func: Callable, *args, **kwargs) -> Any:
        """Run a function in a thread pool."""
        self._check_not_closed()

        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, lambda: func(*args, **kwargs))
        except Exception as e:
            logger.error("Thread execution failed", function=func.__name__, exception=e)
            raise

    async def run_multiple(self, tasks: List[tuple], continue_on_error: bool = False) -> List[Any]:
        """Run multiple functions concurrently in thread pool.

        Args:
            tasks: List of (function, args, kwargs) tuples
            continue_on_error: If True, continue processing other tasks on error
        """
        self._check_not_closed()

        if not tasks:
            return []

        start_time = time.perf_counter()

        futures = []
        for func, args, kwargs in tasks:
            future = self.run_in_thread(func, *args, **kwargs)
            futures.append(future)

        results = await asyncio.gather(*futures, return_exceptions=True)

        # Handle exceptions
        valid_results = []
        exceptions = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                func_name = tasks[i][0].__name__
                logger.error("Task failed", function=func_name, exception=result)
                if continue_on_error:
                    valid_results.append(None)
                    exceptions.append(result)
                else:
                    raise result
            else:
                valid_results.append(result)

        duration_ms = (time.perf_counter() - start_time) * 1000
        log_performance("run_multiple_tasks", duration_ms,
                       num_tasks=len(tasks), num_exceptions=len(exceptions))

        return valid_results

    async def aclose(self):
        """Async close method."""
        if not self._closed:
            self.executor.shutdown(wait=False)
            self._closed = True
            logger.info("Task executor closed")

    def close(self):
        """Sync close method for backward compatibility."""
        if not self._closed:
            self.executor.shutdown(wait=True)
            self._closed = True
            logger.info("Task executor closed")


class AsyncBatchProcessor:
    """Process items in batches with concurrency control."""
    
    def __init__(self, batch_size: int = 10, max_concurrent_batches: int = 3):
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.semaphore = asyncio.Semaphore(max_concurrent_batches)
    
    async def process_batches(self, 
                            items: List[Any], 
                            processor: Callable[[List[Any]], Coroutine]) -> List[Any]:
        """Process items in batches with concurrency control.
        
        Args:
            items: Items to process
            processor: Async function that processes a batch of items
        """
        if not items:
            return []
        
        start_time = time.perf_counter()
        
        # Create batches
        batches = [items[i:i + self.batch_size] 
                  for i in range(0, len(items), self.batch_size)]
        
        # Process batches with concurrency control
        async def process_with_semaphore(batch):
            async with self.semaphore:
                return await processor(batch)
        
        tasks = [process_with_semaphore(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results and handle exceptions
        all_results = []
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error("Batch processing failed", exception=result)
                raise result
            if isinstance(result, list):
                all_results.extend(result)
            else:
                all_results.append(result)
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        log_performance("process_batches", duration_ms, 
                       num_items=len(items), num_batches=len(batches))
        
        return all_results


# Global instances
_task_executor: Optional[AsyncTaskExecutor] = None


def get_async_executor() -> AsyncTaskExecutor:
    """Get the global async task executor."""
    global _task_executor
    if _task_executor is None:
        _task_executor = AsyncTaskExecutor()
    return _task_executor


async def cleanup_async_resources():
    """Clean up async resources."""
    global _task_executor
    if _task_executor:
        await _task_executor.aclose()
        _task_executor = None


async def ainit_async_resources():
    """Initialize async resources."""
    global _task_executor
    if _task_executor is None:
        _task_executor = AsyncTaskExecutor()
    return _task_executor


# Utility functions for common async patterns
async def run_with_timeout(coro: Coroutine, timeout: float) -> Any:
    """Run a coroutine with timeout."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.error("Operation timed out", timeout=timeout)
        raise


async def retry_async(coro_func: Callable[[], Coroutine],
                     max_retries: int = 3,
                     delay: float = 1.0,
                     backoff_factor: float = 2.0) -> Any:
    """Retry an async operation with exponential backoff."""
    last_exception = None

    for attempt in range(max_retries):
        try:
            return await coro_func()
        except Exception as e:
            last_exception = e
            if attempt == max_retries - 1:
                break

            wait_time = delay * (backoff_factor ** attempt)
            logger.warning(f"Retry attempt {attempt + 1}",
                         wait_time=wait_time, exception=str(e))
            await asyncio.sleep(wait_time)

    logger.error("All retry attempts failed", max_retries=max_retries)
    raise last_exception


class AsyncResourceManager:
    """Comprehensive async resource manager for the RAG system."""

    def __init__(self):
        self.http_client = AsyncHTTPClient()
        self.task_executor = AsyncTaskExecutor()
        self.resources_initialized = False

    async def __aenter__(self):
        """Initialize all async resources."""
        await ainit_async_resources()
        self.resources_initialized = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up all async resources."""
        await cleanup_async_resources()
        self.resources_initialized = False

    async def get_http_client(self) -> AsyncHTTPClient:
        """Get the HTTP client with proper session management."""
        return self.http_client

    async def get_task_executor(self) -> AsyncTaskExecutor:
        """Get the task executor."""
        return self.task_executor

    async def execute_with_resources(self, coro: Coroutine) -> Any:
        """Execute a coroutine with full resource management."""
        if not self.resources_initialized:
            raise RuntimeError("AsyncResourceManager must be used as async context manager")

        return await coro