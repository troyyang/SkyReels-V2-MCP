import logging
import os
import time
import base64
import mimetypes
from pathlib import Path
from typing import List, Dict, Union, Optional, Any

import anyio
import uvicorn
from mcp import McpError, ErrorData, types
from pydantic import AnyUrl, BaseModel
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.shared.exceptions import McpError
from mcp.types import (
    EmptyResult,
    ErrorData,
    InitializeResult,
    ReadResourceResult,
    TextContent,
    TextResourceContents,
    Tool,
    EmbeddedResource,
    BlobResourceContents,
)

from pydantic import AnyUrl

from df_generator import generate_video as df_runner
from video_generator import create_video_from_prompt_or_image as v2_runner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('skyreels_server.log')
    ]
)
logger = logging.getLogger('skyreels-server')

SERVER_NAME = "skyreels-server"
DEFAULT_OUTPUT_DIR = os.path.join(os.getcwd(), "output")


class ServerConfig(BaseModel):
    """Server configuration model."""
    output_directory: str = DEFAULT_OUTPUT_DIR
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "info"
    enable_cors: bool = True
    cors_origins: List[str] = ["*"]
    uploads_directory: str = os.path.join(DEFAULT_OUTPUT_DIR, "uploads")


class SkyReelsServer(Server):
    """SkyReels server implementation with file upload support."""

    def __init__(self, config: ServerConfig = None):
        """Initialize the SkyReels server.

        Args:
            config: Server configuration. If None, default configuration is used.
        """
        super().__init__(SERVER_NAME)

        self.config = config or ServerConfig()

        # Ensure output and uploads directories exist
        self.output_dir = Path(self.config.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.uploads_dir = Path(self.config.uploads_directory)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Server initialized with output directory: {self.output_dir}")
        logger.info(f"Server initialized with uploads directory: {self.uploads_dir}")
        logger.info(f"Available tools: {self._get_available_tools()}")

        self._setup_handlers()

    def _setup_handlers(self):
        """Set up the server handlers."""

        @self.read_resource()
        async def handle_read_resource(uri: AnyUrl) -> Union[str, bytes]:
            """Handle resource reading requests.

            Args:
                uri: The URI of the resource to read.

            Returns:
                The resource content.

            Raises:
                McpError: If the resource cannot be found or read.
            """
            logger.info(f"Read resource request: {uri}")

            try:
                if uri.scheme == "file":
                    # Handle file:// URIs
                    path = uri.path
                    if os.name == 'nt' and path.startswith('/'):
                        path = path[1:]  # Remove leading slash on Windows

                    if not os.path.exists(path):
                        raise FileNotFoundError(f"File not found: {path}")

                    # Determine if it's a text or binary file
                    with open(path, 'rb') as f:
                        content = f.read()

                    logger.info(f"Successfully read resource: {uri}")
                    return content

                elif uri.scheme in ["http", "https"]:
                    # This would require implementing HTTP client functionality
                    # For now, we'll just raise an error
                    raise NotImplementedError(f"HTTP/HTTPS resource reading not implemented")

                else:
                    raise ValueError(f"Unsupported URI scheme: {uri.scheme}")

            except Exception as e:
                logger.error(f"Error reading resource {uri}: {str(e)}", exc_info=True)
                raise McpError(
                    error=ErrorData(
                        code=404,
                        message=f"Resource not found or cannot be read: {uri}. Error: {str(e)}"
                    )
                )

        @self.list_resources()
        async def handle_list_resources() -> List[types.Resource]:
            """Handle resource listing requests."""
            logger.info("Resource listing requested")
            resources = []

            # List files in uploads directory
            if self.uploads_dir.exists():
                for file_path in self.uploads_dir.iterdir():
                    if file_path.is_file():
                        mime_type, _ = mimetypes.guess_type(str(file_path))
                        if mime_type is None:
                            mime_type = "application/octet-stream"

                        resources.append(types.Resource(
                            uri=AnyUrl(f"file://{file_path.absolute()}"),
                            name=file_path.name,
                            description=f"Uploaded file: {file_path.name}",
                            mimeType=mime_type
                        ))

            return resources

        @self.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """Handle tool listing requests.

            Returns:
                List of available tools.
            """
            logger.info("Tool listing requested")
            return self._get_available_tools()

        @self.call_tool()
        async def handle_call_tool(name: str, arguments: Dict) -> List[
            Union[types.TextContent, types.ImageContent, types.EmbeddedResource]]:
            """Handle tool call requests."""
            logger.info(f"Tool call request: {name} with arguments: {arguments}")

            try:
                if name == "upload-file":
                    return await self._handle_file_upload(arguments)

                elif name == "skyreels-df":
                    if "prompt" not in arguments:
                        raise ValueError("Missing required argument 'prompt'")

                    # Extract valid parameters for df_runner
                    from inspect import signature
                    df_runner_params = signature(df_runner).parameters
                    filtered_args = {k: v for k, v in arguments.items() if k in df_runner_params}

                    # Server defaults for key parameters (can be overridden by client)
                    server_defaults = {
                        'model_id': "Skywork/SkyReels-V2-DF-1.3B-540P",
                    }

                    # Merge parameters (client args take precedence over server defaults)
                    merged_args = {**server_defaults, **filtered_args}

                    # Enforce server-controlled directory
                    merged_args['save_dir'] = str(self.output_dir / "df")

                    # Handle image path if provided
                    if 'image' in merged_args and merged_args['image']:
                        image_path = self._resolve_uploaded_file_path(merged_args['image'])
                        if image_path:
                            merged_args['image'] = image_path
                        else:
                            logger.warning(f"Could not resolve uploaded image: {merged_args['image']}")

                    logger.info(f"Generating video with merged parameters: {merged_args}")
                    result = df_runner(**merged_args)

                    filename = os.path.basename(result)
                    logger.info(f"Video generation completed successfully: {filename}")
                    file_uri = f"http://{self.config.host}:{self.config.port}/files/df/{filename}"
                    return [
                        EmbeddedResource(
                            type="resource",
                            resource=TextResourceContents(
                                uri=AnyUrl(file_uri),
                                mimeType="video/mp4",
                                text=f"Generated video for prompt: {merged_args['prompt']}"
                            )
                        )
                    ]

                elif name == "skyreels-v2":
                    if "prompt" not in arguments:
                        raise ValueError("Missing required argument 'prompt'")

                    # Extract valid parameters for v2_runner
                    from inspect import signature
                    v2_runner_params = signature(v2_runner).parameters
                    filtered_args = {k: v for k, v in arguments.items() if k in v2_runner_params}

                    # Server defaults for key parameters (can be overridden by client)
                    server_defaults = {
                        'model_id': "Skywork/SkyReels-V2-I2V-1.3B-540P",
                    }

                    # Merge parameters (client args take precedence over server defaults)
                    merged_args = {**server_defaults, **filtered_args}

                    # Enforce server-controlled directory
                    merged_args['save_dir'] = str(self.output_dir / "video_out")

                    # Handle image path if provided
                    if 'image_path' in merged_args and merged_args['image_path']:
                        image_path = self._resolve_uploaded_file_path(merged_args['image_path'])
                        if image_path:
                            merged_args['image_path'] = image_path
                        else:
                            # Fallback to default swan.png if uploaded file not found
                            logger.warning(
                                f"Could not resolve uploaded image: {merged_args['image_path']}, using default")
                            merged_args['image_path'] = 'swan.png'
                    else:
                        merged_args['image_path'] = 'swan.png'

                    logger.info(f"Generating video with merged parameters: {merged_args}")
                    result = v2_runner(**merged_args)

                    filename = os.path.basename(result)
                    logger.info(f"Video generation completed successfully: {filename}")
                    file_uri = f"http://{self.config.host}:{self.config.port}/files/video_out/{filename}"
                    return [
                        EmbeddedResource(
                            type="resource",
                            resource=TextResourceContents(
                                uri=AnyUrl(file_uri),
                                mimeType="video/mp4",
                                text=f"Generated video for prompt: {merged_args['prompt']}"
                            )
                        )
                    ]
                else:
                    raise ValueError(f"Unknown tool: {name}")

            except Exception as e:
                logger.error(f"Error calling tool {name}: {str(e)}", exc_info=True)
                raise McpError(
                    error=ErrorData(
                        code=500,
                        message=f"Error calling tool {name}: {str(e)}"
                    )
                )

    async def _handle_file_upload(self, arguments: Dict) -> List[Union[types.TextContent, types.EmbeddedResource]]:
        """Handle file upload requests.

        Args:
            arguments: Dictionary containing file upload parameters

        Returns:
            List containing upload confirmation
        """
        try:
            if "filename" not in arguments or "content" not in arguments:
                raise ValueError("Missing required arguments 'filename' and 'content'")

            filename = arguments["filename"]
            content_b64 = arguments["content"]
            mime_type = arguments.get("mime_type", "application/octet-stream")

            # Decode base64 content
            try:
                file_content = base64.b64decode(content_b64)
            except Exception as e:
                raise ValueError(f"Invalid base64 content: {str(e)}")

            # Create safe filename
            safe_filename = self._create_safe_filename(filename)
            file_path = self.uploads_dir / safe_filename

            # Write file to uploads directory
            with open(file_path, 'wb') as f:
                f.write(file_content)

            logger.info(f"File uploaded successfully: {safe_filename} ({len(file_content)} bytes)")

            # Return upload confirmation
            return [
                types.TextContent(
                    type="text",
                    text=f"File '{filename}' uploaded successfully as '{safe_filename}'. Size: {len(file_content)} bytes."
                ),
                EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri=AnyUrl(f"file://{file_path.absolute()}"),
                        mimeType=mime_type,
                        text=f"Uploaded file: {safe_filename}"
                    )
                )
            ]

        except Exception as e:
            logger.error(f"Error handling file upload: {str(e)}", exc_info=True)
            raise McpError(
                error=ErrorData(
                    code=500,
                    message=f"Error uploading file: {str(e)}"
                )
            )

    def _create_safe_filename(self, filename: str) -> str:
        """Create a safe filename by removing dangerous characters and adding timestamp."""
        import re
        from datetime import datetime

        # Remove path separators and dangerous characters
        safe_name = re.sub(r'[^\w\-_.]', '_', filename)

        # Add timestamp to prevent conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(safe_name)

        return f"{name}_{timestamp}{ext}"

    def _resolve_uploaded_file_path(self, filename: str) -> Optional[str]:
        """Resolve uploaded file path from filename.

        Args:
            filename: The filename to resolve

        Returns:
            Full path to the uploaded file if found, None otherwise
        """
        # Check if it's already a full path
        if os.path.isabs(filename) and os.path.exists(filename):
            return filename

        # Check in uploads directory
        upload_path = self.uploads_dir / filename
        if upload_path.exists():
            return str(upload_path)

        # Check for files with timestamp suffix (in case of conflicts)
        for file_path in self.uploads_dir.iterdir():
            if file_path.is_file() and filename in file_path.name:
                return str(file_path)

        return None

    def _get_available_tools(self) -> List[Tool]:
        """Get list of available tools.

        Returns:
            List of tools.
        """
        return [
            types.Tool(
                name="upload-file",
                description="Upload a file to the server for use in video generation",
                inputSchema={
                    "type": "object",
                    "required": ["filename", "content"],
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "The name of the file being uploaded.",
                        },
                        "content": {
                            "type": "string",
                            "description": "Base64 encoded file content.",
                        },
                        "mime_type": {
                            "type": "string",
                            "description": "MIME type of the file (optional).",
                        }
                    },
                },
            ),
            types.Tool(
                name="skyreels-df",
                description="Generate a video based on a text prompt using the DF model",
                inputSchema={
                    "type": "object",
                    "required": ["prompt"],
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The text prompt describing the video to generate.",
                        },
                        "image": {
                            "type": "string",
                            "description": "Path or filename of an uploaded image to use as starting frame (optional).",
                        }
                    },
                },
            ),
            types.Tool(
                name="skyreels-v2",
                description="Generate a video based on a text prompt using the V2 I2V model",
                inputSchema={
                    "type": "object",
                    "required": ["prompt"],
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The text prompt describing the video to generate.",
                        },
                        "image_path": {
                            "type": "string",
                            "description": "Path or filename of an uploaded image to use for image-to-video generation (optional).",
                        }
                    },
                },
            )
        ]


def make_server_app(config: ServerConfig = None) -> Starlette:
    """Create Starlette app with SSE transport."""
    config = config or ServerConfig()

    # Set up CORS middleware if enabled
    middleware = []
    if config.enable_cors:
        middleware.append(
            Middleware(
                CORSMiddleware,
                allow_origins=config.cors_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        )

    # Create SSE transport and server
    sse = SseServerTransport("/messages")
    server = SkyReelsServer(config)

    async def handle_sse(request: Request):
        """Handle SSE connections."""
        logger.info(f"New SSE connection from {request.client}")
        async with sse.connect_sse(
                request.scope, request.receive, request._send
        ) as streams:
            await server.run(
                streams[0], streams[1], server.create_initialization_options()
            )
            return Response(status_code=200)

    async def handle_health_check(request: Request) -> JSONResponse:
        """Handle health check requests."""
        return JSONResponse({
            "status": "ok",
            "server": SERVER_NAME,
            "version": "2.0.0",
            "tools": [tool.name for tool in server._get_available_tools()],
            "uploads_directory": str(server.uploads_dir),
            "output_directory": str(server.output_dir),
        })

    # Create Starlette app with static file serving
    app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Route("/health", endpoint=handle_health_check),
            Mount("/messages", app=sse.handle_post_message),
            Mount("/files", app=StaticFiles(directory=config.output_directory), name="files"),
            Mount("/uploads", app=StaticFiles(directory=config.uploads_directory), name="uploads"),
        ],
        middleware=middleware,
    )

    return app


def run_server(config: ServerConfig = None) -> None:
    """Run the server.

    Args:
        config: Server configuration. If None, default configuration is used.
    """
    config = config or ServerConfig()

    app = make_server_app(config)

    log_level = config.log_level.lower()
    if log_level not in ["critical", "error", "warning", "info", "debug", "trace"]:
        log_level = "info"

    server = uvicorn.Server(
        config=uvicorn.Config(
            app=app,
            host=config.host,
            port=config.port,
            log_level=log_level
        )
    )

    logger.info(f"Starting server on {config.host}:{config.port}")
    logger.info(f"Output directory: {config.output_directory}")
    logger.info(f"Uploads directory: {config.uploads_directory}")

    server.run()

    # Give server time to start
    while not server.started:
        logger.info("Waiting for server to start...")
        time.sleep(0.5)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SkyReels Video Generation Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind the server to")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to store generated videos")
    parser.add_argument("--uploads-dir", default=os.path.join(DEFAULT_OUTPUT_DIR, "uploads"),
                        help="Directory to store uploaded files")
    parser.add_argument("--log-level", default="info", choices=["critical", "error", "warning", "info", "debug"],
                        help="Logging level")
    parser.add_argument("--disable-cors", action="store_true", help="Disable CORS")

    args = parser.parse_args()

    config = ServerConfig(
        output_directory=args.output_dir,
        uploads_directory=args.uploads_dir,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        enable_cors=not args.disable_cors,
    )

    run_server(config)