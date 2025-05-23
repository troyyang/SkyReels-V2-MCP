import aiohttp
import asyncio
import json
import logging
import os
import re
import time
import base64
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from urllib.parse import urlparse
from uuid import uuid4

from mcp import ClientSession
from mcp.client.sse import sse_client

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('skyreels_client.log')
    ]
)
logger = logging.getLogger('skyreels-client')

class SkyreelsClient:
    """Client for interacting with the skyreels server with file upload and improved sequential processing."""

    def __init__(self, base_url: str = "http://localhost:8080", download_dir: str = "./downloads"):
        """Initialize the Skyreels client.

        Args:
            base_url: Base URL of the skyreels server.
            download_dir: Directory where downloaded files will be saved.
        """
        self.base_url = base_url
        self.download_dir = Path(download_dir)
        # Create download directory if it doesn't exist
        self.download_dir.mkdir(parents=True, exist_ok=True)

        # For tracking task progress
        self._current_task_id = None
        self._processing_lock = asyncio.Lock()
        self._uploaded_files = {}  # Track uploaded files

    async def upload_file(self, file_path: str, timeout: int = 300) -> Optional[str]:
        """Upload a file to the server.

        Args:
            file_path: Path to the local file to upload.
            timeout: Timeout for the upload operation.

        Returns:
            The uploaded filename on the server, or None if upload failed.
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return None

            if not file_path.is_file():
                logger.error(f"Path is not a file: {file_path}")
                return None

            # Read file content
            with open(file_path, 'rb') as f:
                file_content = f.read()

            # Encode content as base64
            content_b64 = base64.b64encode(file_content).decode('utf-8')

            # Guess MIME type
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if mime_type is None:
                mime_type = "application/octet-stream"

            logger.info(f"Uploading file: {file_path.name} ({len(file_content)} bytes, {mime_type})")

            # Connect to server and upload
            async with sse_client(self.base_url + "/sse", sse_read_timeout=timeout) as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()

                    # Call upload-file tool
                    result = await session.call_tool("upload-file", {
                        "filename": file_path.name,
                        "content": content_b64,
                        "mime_type": mime_type
                    })

                    # Extract uploaded filename from response
                    uploaded_filename = None
                    for item in result.content:
                        if hasattr(item, 'text') and "uploaded successfully as" in item.text:
                            # Extract filename from response text
                            import re
                            match = re.search(r"uploaded successfully as '([^']+)'", item.text)
                            if match:
                                uploaded_filename = match.group(1)
                                break

                    if uploaded_filename:
                        logger.info(f"File uploaded successfully as: {uploaded_filename}")
                        # Cache the mapping for future use
                        self._uploaded_files[file_path.name] = uploaded_filename
                        return uploaded_filename
                    else:
                        logger.error("Could not determine uploaded filename from server response")
                        return None

        except Exception as e:
            logger.error(f"Error uploading file {file_path}: {str(e)}", exc_info=True)
            return None

    async def generate_video_df(
            self,
            prompt: str,
            image_path: Optional[str] = None,
            model_id: str = "Skywork/SkyReels-V2-DF-1.3B-540P",
            resolution: str = "540P",
            num_frames: int = 97,
            end_image: Optional[str] = None,
            video_path: str = '',
            ar_step: int = 0,
            causal_attention: bool = False,
            causal_block_size: int = 1,
            base_num_frames: int = 97,
            overlap_history: Optional[int] = None,
            addnoise_condition: int = 0,
            guidance_scale: float = 6.0,
            shift: float = 8.0,
            inference_steps: int = 30,
            use_usp: bool = False,
            offload: bool = False,
            fps: int = 24,
            seed: Optional[int] = None,
            prompt_enhancer: bool = False,
            teacache: bool = False,
            teacache_thresh: float = 0.2,
            use_ret_steps: bool = False,
            timeout: int = 3600
    ) -> List[Dict]:
        """Generate a video with comprehensive parameters matching df_generator.py.

        Args:
            prompt: The prompt to generate the video from.
            image_path: Path to local image file to upload and use as starting frame.
            model_id: Model identifier for the diffusion model.
            resolution: Output resolution ("540P" or "720P").
            num_frames: Number of frames to generate.
            end_image: Path to ending image (optional).
            video_path: Path to input video for extension (optional).
            ar_step: Auto-regressive step size.
            causal_attention: Whether to use causal attention.
            causal_block_size: Block size for causal attention.
            base_num_frames: Base number of frames for processing.
            overlap_history: Number of overlapping frames for video extension.
            addnoise_condition: Noise conditioning parameter.
            guidance_scale: Scale for classifier-free guidance.
            shift: Shift parameter for diffusion.
            inference_steps: Number of denoising steps.
            use_usp: Whether to use USP (Unified Sequence Parallelism).
            offload: Whether to offload model to CPU when not in use.
            fps: Frames per second for output video.
            seed: Random seed for reproducibility.
            prompt_enhancer: Whether to enhance the prompt automatically.
            teacache: Whether to use TEACache optimization.
            teacache_thresh: Threshold for TEACache.
            use_ret_steps: Whether to use return steps in TEACache.
            timeout: Maximum time to wait for video generation in seconds.

        Returns:
            List of file information for generated videos.
        """
        task_id = str(uuid4())[:8]

        # Wait for any previous tasks and acquire the lock
        async with self._processing_lock:
            self._current_task_id = task_id
            logger.info(f"Task {task_id}: Starting video generation with comprehensive parameters")
            logger.info(f"Task {task_id}: Prompt: {prompt[:100]}...")  # Log first 100 chars of prompt

            start_time = time.time()

            try:
                # Upload image file if provided
                uploaded_image_filename = None
                if image_path:
                    logger.info(f"Task {task_id}: Uploading image file: {image_path}")
                    uploaded_image_filename = await self.upload_file(image_path)
                    if not uploaded_image_filename:
                        logger.error(f"Task {task_id}: Failed to upload image file: {image_path}")
                        return []

                # Prepare the parameters dictionary
                params = {
                    "prompt": prompt,
                    "model_id": model_id,
                    "resolution": resolution,
                    "num_frames": num_frames,
                    "end_image": end_image,
                    "video_path": video_path,
                    "ar_step": ar_step,
                    "causal_attention": causal_attention,
                    "causal_block_size": causal_block_size,
                    "base_num_frames": base_num_frames,
                    "overlap_history": overlap_history,
                    "addnoise_condition": addnoise_condition,
                    "guidance_scale": guidance_scale,
                    "shift": shift,
                    "inference_steps": inference_steps,
                    "use_usp": use_usp,
                    "offload": offload,
                    "fps": fps,
                    "seed": seed,
                    "prompt_enhancer": prompt_enhancer,
                    "teacache": teacache,
                    "teacache_thresh": teacache_thresh,
                    "use_ret_steps": use_ret_steps
                }

                # Add uploaded image filename
                if uploaded_image_filename:
                    params["image"] = uploaded_image_filename

                # Remove None values to avoid sending null parameters
                params = {k: v for k, v in params.items() if v is not None}

                # Connect to the server with appropriate timeout
                async with sse_client(self.base_url + "/sse", sse_read_timeout=timeout) as streams:
                    async with ClientSession(*streams) as session:
                        await session.initialize()
                        logger.info(f"Task {task_id}: Connected to server")

                        # Call the skyreels-df tool with all parameters
                        logger.info(f"Task {task_id}: Sending generation request with parameters")
                        logger.info(f"Task {task_id}: Video generation started. This may take several minutes...")

                        result = await session.call_tool("skyreels-df", params)

                        elapsed_time = time.time() - start_time
                        logger.info(f"Task {task_id}: Video generation completed in {elapsed_time:.2f} seconds")

                        # Process the result and download files
                        downloaded_files = await self._process_generation_result(result, task_id)

                        total_elapsed = time.time() - start_time
                        logger.info(f"Task {task_id}: Total processing time: {total_elapsed:.2f} seconds")
                        return downloaded_files

            except asyncio.TimeoutError:
                logger.error(f"Task {task_id}: Timeout after {timeout} seconds waiting for video generation")
                return []

            except Exception as e:
                logger.error(f"Task {task_id}: Error during video generation: {str(e)}", exc_info=True)
                return []

            finally:
                self._current_task_id = None

    async def generate_video(
            self,
            prompt: str,
            image_path: Optional[str] = None,
            model_id: str = "Skywork/SkyReels-V2-I2V-1.3B-540P",
            resolution: str = "540P",
            num_frames: int = 97,
            guidance_scale=6.0,
            shift=8.0,
            inference_steps=30,
            use_usp=False,
            offload=False,
            fps=24,
            seed=None,
            prompt_enhancer=False,
            teacache=False,
            teacache_thresh=0.2,
            use_ret_steps=False,
            timeout: int = 3600
    ) -> List[Dict]:
        """Generate a video using the V2 I2V model with image upload support.

        Args:
            prompt: The prompt to generate the video from.
            image_path: Path to local image file to upload and use for I2V generation.
            model_id: Model identifier for the diffusion model.
            resolution: Output resolution ("540P" or "720P").
            num_frames: Number of frames to generate.
            guidance_scale: Scale for classifier-free guidance.
            shift: Shift parameter for diffusion.
            inference_steps: Number of denoising steps.
            use_usp: Whether to use USP (Unified Sequence Parallelism).
            offload: Whether to offload model to CPU when not in use.
            fps: Frames per second for output video.
            seed: Random seed for reproducibility.
            prompt_enhancer: Whether to enhance the prompt automatically.
            teacache: Whether to use TEACache optimization.
            teacache_thresh: Threshold for TEACache.
            use_ret_steps: Whether to use return steps in TEACache.
            timeout: Maximum time to wait for video generation in seconds.

        Returns:
            List of file information for generated videos.
        """
        task_id = str(uuid4())[:8]

        # Wait for any previous tasks and acquire the lock
        async with self._processing_lock:
            self._current_task_id = task_id
            logger.info(f"Task {task_id}: Starting V2 video generation with image upload support")
            logger.info(f"Task {task_id}: Prompt: {prompt[:100]}...")  # Log first 100 chars of prompt

            start_time = time.time()

            try:
                # Upload image file if provided
                uploaded_image_filename = None
                if image_path:
                    logger.info(f"Task {task_id}: Uploading image file: {image_path}")
                    uploaded_image_filename = await self.upload_file(image_path)
                    if not uploaded_image_filename:
                        logger.error(f"Task {task_id}: Failed to upload image file: {image_path}")
                        return []

                # Prepare the parameters dictionary
                params = {
                    "prompt": prompt,
                    "model_id": model_id,
                    "resolution": resolution,
                    "num_frames": num_frames,
                    "guidance_scale": guidance_scale,
                    "shift": shift,
                    "inference_steps": inference_steps,
                    "use_usp": use_usp,
                    "offload": offload,
                    "fps": fps,
                    "seed": seed,
                    "prompt_enhancer": prompt_enhancer,
                    "teacache": teacache,
                    "teacache_thresh": teacache_thresh,
                    "use_ret_steps": use_ret_steps
                }

                # Add uploaded image filename
                if uploaded_image_filename:
                    params["image_path"] = uploaded_image_filename

                # Remove None values to avoid sending null parameters
                params = {k: v for k, v in params.items() if v is not None}

                # Connect to the server with appropriate timeout
                async with sse_client(self.base_url + "/sse", sse_read_timeout=timeout) as streams:
                    async with ClientSession(*streams) as session:
                        await session.initialize()
                        logger.info(f"Task {task_id}: Connected to server")

                        # Call the skyreels-v2 tool with all parameters
                        logger.info(f"Task {task_id}: Sending generation request with parameters")
                        logger.info(f"Task {task_id}: Video generation started. This may take several minutes...")

                        result = await session.call_tool("skyreels-v2", params)

                        elapsed_time = time.time() - start_time
                        logger.info(f"Task {task_id}: Video generation completed in {elapsed_time:.2f} seconds")

                        # Process the result and download files
                        downloaded_files = await self._process_generation_result(result, task_id)

                        total_elapsed = time.time() - start_time
                        logger.info(f"Task {task_id}: Total processing time: {total_elapsed:.2f} seconds")
                        return downloaded_files

            except asyncio.TimeoutError:
                logger.error(f"Task {task_id}: Timeout after {timeout} seconds waiting for video generation")
                return []

            except Exception as e:
                logger.error(f"Task {task_id}: Error during video generation: {str(e)}", exc_info=True)
                return []

            finally:
                self._current_task_id = None

    async def list_uploaded_files(self) -> List[Dict]:
        """List files uploaded to the server.

        Returns:
            List of uploaded files with their metadata.
        """
        try:
            async with sse_client(self.base_url + "/sse", sse_read_timeout=30) as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()
                    response = await session.list_resources()

                    uploaded_files = []
                    for resource in response.resources:
                        uploaded_files.append({
                            "uri": str(resource.uri),
                            "name": resource.name,
                            "description": resource.description,
                            "mime_type": resource.mimeType
                        })

                    return uploaded_files
        except Exception as e:
            logger.error(f"Error listing uploaded files: {str(e)}", exc_info=True)
            return []

    async def _process_generation_result(self, result: Any, task_id: str) -> List[Dict]:
        """Process the result from video generation and download files.

        Args:
            result: The result returned from the server.
            task_id: The ID of the current task.

        Returns:
            List of dictionaries with information about the downloaded files.
        """
        downloaded_files = []
        logger.info(f"Task {task_id}: Processing generation result")

        # Check if we have content in the result
        if hasattr(result, 'content') and result.content:
            file_count = 0
            total_files = sum(1 for item in result.content if hasattr(item, 'resource') and item.resource)

            for item in result.content:
                if hasattr(item, 'resource') and item.resource:
                    resource = item.resource
                    file_count += 1

                    # Extract URI and file information
                    if hasattr(resource, 'uri'):
                        uri = resource.uri
                        mime_type = getattr(resource, 'mimeType', 'application/octet-stream')

                        logger.info(f"Task {task_id}: Processing file {file_count}/{total_files}: {uri}")

                        # Download the file
                        file_info = await self._download_file(uri, mime_type, task_id)
                        if file_info:
                            downloaded_files.append(file_info)
        else:
            logger.warning(f"Task {task_id}: No content found in generation result")

        logger.info(f"Task {task_id}: Successfully processed {len(downloaded_files)} files")
        return downloaded_files

    async def _download_file(self, uri: str, mime_type: str, task_id: str) -> Optional[Dict]:
        """Download a file from the given URI."""
        try:
            # Ensure uri is a string
            uri_str = str(uri)
            parsed_url = urlparse(uri_str)
            path = parsed_url.path
            filename = os.path.basename(path)

            # Generate a safe filename
            clean_filename = re.sub(r'[^\w\-_. ]', '', filename)
            if not clean_filename:
                clean_filename = f"file_{int(time.time())}"

            # Add extension if missing
            name, ext = os.path.splitext(clean_filename)
            if not ext:
                ext = self._get_extension_from_mime_type(mime_type)
                clean_filename = f"{name}{ext}"

            # Add timestamp to prevent conflicts
            timestamp = int(time.time())
            final_filename = f"{name}_{timestamp}{ext}"
            file_path = self.download_dir / final_filename

            # For HTTP/HTTPS downloads
            if parsed_url.scheme in ["http", "https"]:
                timeout = aiohttp.ClientTimeout(total=300)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    logger.info(f"Task {task_id}: Downloading {uri_str}")
                    download_start = time.time()

                    async with session.get(uri_str) as response:
                        if response.status == 200:
                            # Get actual content type if available
                            content_type = response.headers.get('Content-Type', mime_type)
                            if content_type != mime_type:
                                logger.info(f"Task {task_id}: Server reported content type {content_type}, "
                                            f"using instead of {mime_type}")
                                mime_type = content_type

                            # Stream the download
                            total_size = 0
                            with open(file_path, 'wb') as f:
                                async for chunk in response.content.iter_chunked(1024 * 1024):  # 1MB chunks
                                    f.write(chunk)
                                    total_size += len(chunk)

                            download_time = time.time() - download_start
                            size_mb = total_size / (1024 * 1024)
                            logger.info(f"Task {task_id}: Downloaded {size_mb:.2f}MB in {download_time:.2f}s")

                            return {
                                "source_uri": uri_str,
                                "local_path": str(file_path),
                                "mime_type": mime_type,
                                "filename": final_filename,
                                "file_size_bytes": total_size,
                                "task_id": task_id
                            }
                        else:
                            logger.error(f"Task {task_id}: Download failed with status {response.status}")
                            return None

            # For file:// URIs (local testing)
            elif parsed_url.scheme == "file":
                source_path = parsed_url.path
                if os.name == 'nt' and source_path.startswith('/'):
                    source_path = source_path[1:]

                if os.path.exists(source_path):
                    with open(source_path, 'rb') as src, open(file_path, 'wb') as dest:
                        dest.write(src.read())
                    logger.info(f"Task {task_id}: Copied local file to {file_path}")

                    # Get file size
                    file_size = os.path.getsize(file_path)

                    return {
                        "source_uri": uri_str,
                        "local_path": str(file_path),
                        "mime_type": mime_type,
                        "filename": final_filename,
                        "file_size_bytes": file_size,
                        "task_id": task_id
                    }
                else:
                    logger.error(f"Task {task_id}: Source file not found: {source_path}")
                    return None

            else:
                logger.error(f"Task {task_id}: Unsupported URI scheme: {parsed_url.scheme}")
                return None

        except Exception as e:
            logger.error(f"Task {task_id}: Error downloading file: {str(e)}", exc_info=True)
            return None

    def _get_extension_from_mime_type(self, mime_type: str) -> str:
        """Get a file extension from a MIME type."""
        mime_to_ext = {
            'video/mp4': '.mp4',
            'video/quicktime': '.mov',
            'video/x-msvideo': '.avi',
            'video/webm': '.webm',
            'video/x-matroska': '.mkv',
            'image/jpeg': '.jpg',
            'image/png': '.png',
            'image/gif': '.gif',
            'application/pdf': '.pdf',
            'application/json': '.json',
            'text/plain': '.txt',
        }
        return mime_to_ext.get(mime_type.lower(), '.bin')

    async def list_available_tools(self) -> List[Dict]:
        """List all available tools on the server.

        Returns:
            A list of available tools with their descriptions and input schemas.
        """
        # Use a reasonable timeout for tool listing
        async with sse_client(self.base_url + "/sse", sse_read_timeout=30) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                response = await session.list_tools()
                return response.tools

    def get_current_task_id(self) -> Optional[str]:
        """Get the ID of the currently executing task, if any.

        Returns:
            The current task ID or None if no task is running.
        """
        return self._current_task_id

    def get_uploaded_files(self) -> Dict[str, str]:
        """Get mapping of original filenames to uploaded filenames.

        Returns:
            Dictionary mapping original filenames to server filenames.
        """
        return self._uploaded_files.copy()


async def create_video_for_df():
    """Example usage of the Skyreels client."""
    # Create client with custom download directory
    client = SkyreelsClient(download_dir="./downloads")

    try:
        # List available tools
        tools = await client.list_available_tools()
        print(f"Available tools: {tools}")

        # Define some prompts for video generation
        prompts = [
            "A young person sits alone in a dimly lit room, staring at their reflection in a rain-streaked window.A cat chases a butterfly in a sunlit garden with colorful flowers.An astronaut floats in space with Earth visible in the background.",
            # "A cat chases a butterfly in a sunlit garden with colorful flowers.",
            # "An astronaut floats in space with Earth visible in the background."
        ]

        # Process prompts sequentially
        for i, prompt in enumerate(prompts):
            print(f"\n[{i + 1}/{len(prompts)}] Processing prompt: {prompt}")

            # Generate video and wait for completion
            downloaded_files = await client.generate_video_df(prompt,
                                        resolution="540P",
                                        ar_step=5,
                                        causal_block_size=5,
                                        base_num_frames=97,
                                        num_frames=377,
                                        overlap_history=17,
                                        addnoise_condition=20,
                                        offload=True,
                            )

            if downloaded_files:
                print(f"Successfully downloaded {len(downloaded_files)} files:")
                for file_info in downloaded_files:
                    print(f"  - {file_info['filename']} ({file_info['mime_type']}) saved to {file_info['local_path']}")
            else:
                print("No files were generated or downloaded for this prompt.")

    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)



async def create_video_for_i2v():
    """Example usage of the Skyreels client with file upload."""
    # Create client with custom download directory
    client = SkyreelsClient(download_dir="./downloads")

    try:
        # List available tools
        tools = await client.list_available_tools()
        print(f"Available tools: {[tool.name for tool in tools]}")

        # Example: Upload an image file first
        image_path = "swan.png"  # Replace with your actual image path
        uploaded_filename = None
        if Path(image_path).exists():
            print(f"\nUploading image: {image_path}")
            uploaded_filename = await client.upload_file(image_path)
            if uploaded_filename:
                print(f"Image uploaded successfully as: {uploaded_filename}")
            else:
                print("Failed to upload image")
                return
        else:
            print(f"Image file not found: {image_path}")
            print("Proceeding without image upload...")
            image_path = None

        # List uploaded files
        uploaded_files = await client.list_uploaded_files()
        print(f"\nUploaded files: {[f['name'] for f in uploaded_files]}")

        # # Define prompts for video generation
        prompts = [
            "A serene lake surrounded by towering mountains, with a few swans gracefully gliding across the water and sunlight dancing on the surface.",
        ]

        # Process prompts sequentially
        for i, prompt in enumerate(prompts):
            print(f"\n[{i + 1}/{len(prompts)}] Processing prompt: {prompt}")

            # Generate video with uploaded image (V2 model for I2V)
            downloaded_files = await client.generate_video(
                prompt=prompt,
                image_path=image_path,  # This will be uploaded automatically if provided
                model_id="Skywork/SkyReels-V2-I2V-1.3B-540P",
                resolution="540P",
                num_frames=97,
                guidance_scale=5.0,
                shift=3.0,
                fps=24,
                offload=True,
                teacache=True,
                use_ret_steps=True,
                teacache_thresh=0.3,
            )

            if downloaded_files:
                print(f"Successfully downloaded {len(downloaded_files)} files:")
                for file_info in downloaded_files:
                    print(f"  - {file_info['filename']} ({file_info['mime_type']}) saved to {file_info['local_path']}")
            else:
                print("No files were generated or downloaded for this prompt.")

    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(create_video_for_df())
    asyncio.run(create_video_for_i2v())