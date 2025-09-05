import os
import uuid
import shutil
import asyncio
import aiofiles
import tarfile
import tempfile
import requests
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import subprocess
import logging
from typing import List
import json
from datetime import datetime, timezone
from huggingface_hub import HfApi
import time
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from sse_starlette.sse import EventSourceResponse
import paramiko
from typing import Dict
import re
from fastapi.templating import Jinja2Templates

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
templates = Jinja2Templates(directory="templates")


# Configuration
UPLOAD_DIR = Path("uploads")
EMBEDDING_DIR = Path("embeddings")
SNAPSHOT_DIR = Path("snapshots")
WASM_DIR = Path("wasm")
MODEL_DIR = Path("models")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB default
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "default"
VECTOR_SIZE = 1536  # For gte-Qwen2-1.5B model
HF_DATASET_NAME=os.getenv("HF_DATASET_NAME", "thenocode/gaia-console")

# DigitalOcean config
DO_TOKEN = os.getenv("DO_TOKEN")
SSH_KEY_ID = os.getenv("DO_SSH_KEY_ID")  # must be uploaded to DO account already
DO_REGION = os.getenv("DO_REGION", "nyc3")
DO_SIZE = os.getenv("DO_SIZE", "s-2vcpu-4gb")
SSH_PRIVATE_KEY = os.getenv("DO_SSH_PRIVATE_KEY", "do_gaia_console")
SSH_PASSPHRASE = os.getenv("DO_SSH_PASSPHRASE")  # passphrase for encrypted private key
SSH_USER = "root"

# In-memory stores (replace with DB / Redis in prod)
DEPLOYMENTS: Dict[int, dict] = {}
LOG_STREAMS: Dict[int, asyncio.Queue] = {}
LOGS: Dict[int, List[str]] = {}

# Create directories
for directory in [UPLOAD_DIR, EMBEDDING_DIR, SNAPSHOT_DIR, WASM_DIR, MODEL_DIR]:
    directory.mkdir(exist_ok=True)

# Global Qdrant client
qdrant_client = None
wasmedge_available = False
progress_events = {}

app = FastAPI()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Get Hugging Face token from environment
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    logger.warning("HF_TOKEN environment variable not set. Hugging Face uploads will not work.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI"""
    # Startup: Initialize services
    global qdrant_client, wasmedge_available, hf_access_verified, dataset_verified
    
    # Initialize variables
    qdrant_client = None
    wasmedge_available = False
    hf_access_verified = False
    dataset_verified = False
    
    try:
        qdrant_client = await initialize_qdrant()
        wasmedge_available = await check_wasmedge()
        hf_access_verified = await verify_huggingface_access()
        dataset_verified = await verify_huggingface_dataset()
        
        if not hf_access_verified and HF_TOKEN:
            logger.warning("Hugging Face token exists but write access could not be verified")
        
        if not dataset_verified and HF_DATASET_NAME:
            logger.warning(f"Target dataset {HF_DATASET_NAME} could not be verified")
        
        logger.info("Application started successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        # Continue anyway - the app can still run with fallback functionality
    
    yield
    
    # Shutdown: Clean up resources
    logger.info("Shutting down application...")
    
    try:
        # Clean up Qdrant client if it exists
        if qdrant_client is not None:
            try:
                # Check if the client has a close method and it's awaitable
                if hasattr(qdrant_client, 'close') and callable(getattr(qdrant_client, 'close')):
                    # Check if it's an async method
                    if asyncio.iscoroutinefunction(qdrant_client.close):
                        await qdrant_client.close()
                    else:
                        qdrant_client.close()
                    logger.info("Qdrant client closed successfully")
                else:
                    logger.info("Qdrant client doesn't have a close method")
            except Exception as e:
                logger.warning(f"Error closing Qdrant client: {e}")
        
        # Clean up Qdrant collection if it exists
        try:
            await cleanup_qdrant_collection()
        except Exception as e:
            logger.warning(f"Error cleaning up Qdrant collection: {e}")
            
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    
    logger.info("Application shutdown complete")

app = FastAPI(title="Gaia Node Knowledge Base Generator", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

async def check_wasmedge():
    """Check if WasmEdge is available and install it if not"""
    try:
        # Check if wasmedge is in PATH
        result = subprocess.run(["wasmedge", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"WasmEdge found: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    # WasmEdge not found, try to install it
    logger.warning("WasmEdge not found. Attempting to install...")
    try:
        install_script = """
        curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install_v2.sh | bash -s
        export PATH="$HOME/.wasmedge/bin:$PATH"
        """
        
        result = subprocess.run(install_script, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            # Add WasmEdge to PATH for current process
            wasmedge_path = Path.home() / ".wasmedge" / "bin"
            os.environ["PATH"] = f"{wasmedge_path}:{os.environ['PATH']}"
            
            # Verify installation
            result = subprocess.run(["wasmedge", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"WasmEdge installed successfully: {result.stdout.strip()}")
                return True
            else:
                logger.error("WasmEdge installation failed")
                return False
        else:
            logger.error(f"WasmEdge installation failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error installing WasmEdge: {e}")
        return False

async def initialize_qdrant():
    """Initialize Qdrant connection with retry logic"""
    max_retries = 5
    retry_delay = 3  # seconds
    
    for attempt in range(max_retries):
        try:
            # Create client with proper authentication
            client_config = {
                "url": QDRANT_URL,
                "prefer_grpc": False
            }
            
            # Add API key if provided
            if QDRANT_API_KEY:
                client_config["api_key"] = QDRANT_API_KEY
            
            client = QdrantClient(**client_config)
            
            # Test connection with a simple operation
            client.get_collections()
            
            logger.info(f"Connected to Qdrant at {QDRANT_URL}")
            return client
            
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Failed to connect to Qdrant (attempt {attempt + 1}/{max_retries}): {e}")
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"Failed to connect to Qdrant after {max_retries} attempts: {e}")
                # Create a mock client for development
                logger.warning("Creating in-memory Qdrant client for development")
                return QdrantClient(":memory:")
    
    return QdrantClient(":memory:")  # Fallback to in-memory

async def verify_huggingface_access():
    """Verify that the token has write access"""
    try:
        if not HF_TOKEN:
            return False
            
        api = HfApi(token=HF_TOKEN)
        
        # Try to get your user info to verify token works
        user_info = api.whoami()
        username = user_info.get('name', 'unknown')
        logger.info(f"Hugging Face user: {username}")
        
        # Try to create a test repo to verify write access
        test_repo = f"{username}/test_write_access_{int(time.time())}"
        api.create_repo(repo_id=test_repo, repo_type="dataset", private=True, exist_ok=False)
        api.delete_repo(repo_id=test_repo, repo_type="dataset")
        
        logger.info("✓ Token has write access confirmed")
        return True
        
    except Exception as e:
        logger.error(f"Token access verification failed: {e}")
        return False

async def verify_huggingface_dataset():
    """Verify that the target dataset exists and is accessible"""
    try:
        if not HF_TOKEN or not HF_DATASET_NAME:
            return False
            
        api = HfApi(token=HF_TOKEN)
        
        # Try to access the dataset
        repo_info = api.repo_info(repo_id=HF_DATASET_NAME, repo_type="dataset")
        logger.info(f"✓ Dataset accessible: {HF_DATASET_NAME}")
        
        # Try to check if snapshots folder exists by listing files
        try:
            files = api.list_repo_files(repo_id=HF_DATASET_NAME, repo_type="dataset")
            snapshots_exists = any(f.startswith("snapshots/") for f in files)
            if snapshots_exists:
                logger.info("✓ Snapshots folder exists in dataset")
            else:
                logger.info("ℹ Snapshots folder doesn't exist yet - will be created on first upload")
        except:
            logger.info("ℹ Could not check snapshots folder - will attempt to create it")
        
        return True
        
    except Exception as e:
        logger.error(f"Dataset verification failed: {HF_DATASET_NAME} - {e}")
        return False

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Return the main HTML interface"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Gaia Node Knowledge Base Snapshot Generator</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script>
            tailwind.config = {
                theme: {
                    extend: {
                        colors: {
                            primary: {
                                50: '#eff6ff',
                                100: '#dbeafe',
                                500: '#3b82f6',
                                600: '#2563eb',
                                700: '#1d4ed8',
                                900: '#1e3a8a',
                            }
                        }
                    }
                }
            }
        </script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <style>
            body {
                font-family: 'Inter', sans-serif;
                background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            }
            
            .card {
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            
            .feature-card {
                transition: all 0.3s ease;
                border: 1px solid #e5e7eb;
            }
            
            .feature-card:hover {
                transform: translateY(-4px);
                box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
            }
            
            .gradient-bg {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            
            .drop-zone-gradient {
                background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
                border: 2px dashed #d1d5db;
            }
            
            .drop-zone-gradient.dragover {
                background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
                border-color: #3b82f6;
            }
            .toast {
                position: fixed;
                bottom: 20px;
                right: 20px;
                background: #1f2937;
                color: white;
                padding: 12px 16px;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                z-index: 1000;
                animation: slideIn 0.3s ease-out;
            }
            .deployment-status {
                transition: all 0.3s ease;
            }

            .deployment-status.success {
                background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
                border-color: #34d399;
            }

            .deployment-status.error {
                background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
                border-color: #f87171;
            }
            @keyframes slideIn {
                from {
                    transform: translateX(100%);
                    opacity: 0;
                }
                to {
                    transform: translateX(0);
                    opacity: 1;
                }
            }
        </style>
    </head>
    <body class="min-h-screen">
        <div class="min-h-screen flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
            <div class="max-w-4xl w-full space-y-8">
                <!-- Header -->
                <div class="text-center">
                    <div class="flex items-center justify-center mb-6">
                        <img src="/static/logo.png" alt="Gaia Node" class="h-12 w-auto">
                    </div>
                    <h1 class="text-3xl font-bold text-gray-900 mb-2">Knowledge Snapshot Generator</h1>
                    <p class="text-gray-600">Create intelligent knowledge snapshots for your Gaia Node deployment</p>
                </div>

                <!-- Main Upload Card -->
                <div class="bg-white rounded-xl shadow-lg p-6 card">
                    <div id="wasmWarning" class="bg-yellow-50 border-l-4 border-yellow-400 p-4 mb-6 rounded hidden">
                        <div class="flex">
                            <div class="flex-shrink-0">
                                <i class="fas fa-exclamation-triangle text-yellow-400"></i>
                            </div>
                            <div class="ml-3">
                                <p class="text-sm text-yellow-700">
                                    <strong>Performance Note:</strong> WasmEdge is not available. Using fallback embedding generation.
                                </p>
                            </div>
                        </div>
                    </div>

                    <div class="drop-zone-gradient rounded-lg p-8 text-center cursor-pointer transition-all duration-200 mb-6" id="dropZone">
                        <div class="text-primary-600 text-4xl mb-3">
                            <i class="fas fa-cloud-upload-alt"></i>
                        </div>
                        <p class="text-gray-600 font-medium">Drag & drop your files here</p>
                        <p class="text-sm text-gray-500 mt-1">or click to browse (TXT, MD, PDF, CSV)</p>
                        <p class="text-xs text-gray-400 mt-2">Max 10MB per file</p>
                        <input type="file" id="fileInput" multiple class="hidden">
                    </div>

                    <div id="fileList" class="space-y-3 mb-6"></div>

                    <button id="processBtn" class="w-full bg-primary-600 hover:bg-primary-700 text-white py-3 px-4 rounded-lg font-semibold shadow-md hover:shadow-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center" disabled>
                        <i class="fas fa-bolt mr-2"></i>
                        Generate Snapshot
                    </button>

                    <!-- Progress Section -->
                    <div id="progressSection" class="hidden mt-6">
                        <div class="bg-gray-50 rounded-lg p-6">
                            <h3 class="text-lg font-semibold text-gray-800 mb-4">Processing Progress</h3>
                            <div class="w-full bg-gray-200 rounded-full h-2 mb-4">
                                <div id="progressFill" class="bg-primary-600 h-2 rounded-full transition-all duration-500" style="width: 0%"></div>
                            </div>
                            <p id="progressText" class="text-gray-700 mb-4">Starting snapshot generation...</p>
                            <div class="bg-white rounded-lg p-4 max-h-48 overflow-y-auto">
                                <div id="stepDetails" class="space-y-1 text-sm text-gray-600"></div>
                            </div>
                        </div>
                    </div>

                    <!-- Result Section -->
                    <div id="resultSection" class="hidden mt-6">
                        <div class="bg-green-50 border border-green-200 rounded-lg p-6">
                            <div class="flex items-start">
                                <div class="flex-shrink-0">
                                    <i class="fas fa-check-circle text-green-500 text-xl"></i>
                                </div>
                                <div class="ml-3 flex-1">
                                    <h3 class="text-lg font-semibold text-green-800 mb-2">✅ Snapshot Created Successfully!</h3>
                                    <p class="text-green-700 mb-4">Your Gaia Node snapshot is ready for deployment.</p>
                                    
                                    <!-- Snapshot URL -->
                                    <div class="mb-6">
                                        <h4 class="font-medium text-gray-900 mb-2">Snapshot URL:</h4>
                                        <div class="bg-gray-100 p-3 rounded-lg">
                                            <a id="snapshotUrl" target="_blank" class="text-primary-600 hover:text-primary-800 break-all font-mono text-sm"></a>
                                        </div>
                                    </div>

                                    <!-- Usage Instructions -->
                                    <div class="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
                                        <h4 class="font-semibold text-blue-800 mb-3">Steps to use this snapshot with your Gaia node:</h4>
                                        
                                        <div class="space-y-4">
                                            <!-- Step 1 -->
                                            <div class="flex items-start">
                                                <div class="flex-shrink-0 w-6 h-6 bg-blue-600 rounded-full flex items-center justify-center mt-0.5">
                                                    <span class="text-white text-xs font-bold">1</span>
                                                </div>
                                                <div class="ml-3">
                                                    <p class="text-sm font-medium text-gray-900 mb-1">Install the Gaia CLI (if not already installed):</p>
                                                    <div class="bg-gray-900 text-green-400 p-3 rounded-lg font-mono text-sm overflow-x-auto">
                                                        curl -sSfL 'https://github.com/GaiaNet-AI/gaianet-node/releases/latest/download/install.sh' | bash
                                                    </div>
                                                </div>
                                            </div>

                                            <!-- Step 2 -->
                                            <div class="flex items-start">
                                                <div class="flex-shrink-0 w-6 h-6 bg-blue-600 rounded-full flex items-center justify-center mt-0.5">
                                                    <span class="text-white text-xs font-bold">2</span>
                                                </div>
                                                <div class="ml-3">
                                                    <p class="text-sm font-medium text-gray-900 mb-1">Update the node's configuration:</p>
                                                    <div class="bg-gray-900 text-green-400 p-3 rounded-lg font-mono text-sm overflow-x-auto" id="configCommand">
                                                        <!-- This will be populated by JavaScript -->
                                                    </div>
                                                </div>
                                            </div>

                                            <!-- Step 3 -->
                                            <div class="flex items-start">
                                                <div class="flex-shrink-0 w-6 h-6 bg-blue-600 rounded-full flex items-center justify-center mt-0.5">
                                                    <span class="text-white text-xs font-bold">3</span>
                                                </div>
                                                <div class="ml-3">
                                                    <p class="text-sm font-medium text-gray-900 mb-1">Initialize and run the node:</p>
                                                    <div class="bg-gray-900 text-green-400 p-3 rounded-lg font-mono text-sm overflow-x-auto space-y-2">
                                                        <div>gaianet init</div>
                                                        <div>gaianet start</div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    <!-- Copy buttons -->
                                    <div class="flex space-x-3">
                                        <button onclick="copySnapshotUrl()" class="flex items-center px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg text-sm">
                                            <i class="fas fa-copy mr-2"></i> Copy URL
                                        </button>
                                        <button onclick="copyConfigCommand()" class="flex items-center px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg text-sm">
                                            <i class="fas fa-terminal mr-2"></i> Copy Config Command
                                        </button>
                                        <button id="deploy-do" class="hidden flex items-center px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm">
                                            <i class="fas fa-cloud mr-2"></i> Deploy to DigitalOcean
                                        </button>
                                    </div>

                                    <div id="deploymentStatus" class="hidden mt-4">
                                        <div class="bg-blue-50 border border-blue-200 rounded-lg p-4">
                                            <div class="flex items-center">
                                                <div class="flex-shrink-0">
                                                    <i class="fas fa-sync-alt fa-spin text-blue-500"></i>
                                                </div>
                                                <div class="ml-3">
                                                    <p class="text-sm font-medium text-blue-800">Deployment in progress...</p>
                                                    <p class="text-sm text-blue-600">Your Gaia node is being deployed. This may take 5-10 minutes.</p>
                                                    <p class="text-xs text-blue-500 mt-1">Droplet ID: <span id="currentDropletId"></span></p>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Error Section -->
                    <div id="errorSection" class="hidden mt-6">
                        <div class="bg-red-50 border border-red-200 rounded-lg p-4">
                            <div class="flex">
                                <div class="flex-shrink-0">
                                    <i class="fas fa-exclamation-circle text-red-400"></i>
                                </div>
                                <div class="ml-3">
                                    <p id="errorMessage" class="text-sm text-red-700"></p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Information Section -->
                <div class="bg-white rounded-xl shadow-lg p-6 card">
                    <h2 class="text-2xl font-bold text-gray-900 mb-6 text-center">How It Works</h2>
                    
                    <div class="grid md:grid-cols-3 gap-6 mb-8">
                        <div class="feature-card bg-white p-6 rounded-lg border">
                            <div class="text-center mb-4">
                                <div class="w-12 h-12 bg-primary-100 rounded-full flex items-center justify-center mx-auto">
                                    <i class="fas fa-file-upload text-primary-600 text-xl"></i>
                                </div>
                            </div>
                            <h3 class="font-semibold text-gray-900 mb-2 text-center">1. Upload Files</h3>
                            <p class="text-gray-600 text-sm text-center">Upload your knowledge files (TXT, MD, PDF, CSV) with support for multiple formats</p>
                        </div>

                        <div class="feature-card bg-white p-6 rounded-lg border">
                            <div class="text-center mb-4">
                                <div class="w-12 h-12 bg-primary-100 rounded-full flex items-center justify-center mx-auto">
                                    <i class="fas fa-brain text-primary-600 text-xl"></i>
                                </div>
                            </div>
                            <h3 class="font-semibold text-gray-900 mb-2 text-center">2. AI Processing</h3>
                            <p class="text-gray-600 text-sm text-center">Uses gte-Qwen2-1.5B model to generate high-quality embeddings with WasmEdge acceleration</p>
                        </div>

                        <div class="feature-card bg-white p-6 rounded-lg border">
                            <div class="text-center mb-4">
                                <div class="w-12 h-12 bg-primary-100 rounded-full flex items-center justify-center mx-auto">
                                    <i class="fas fa-rocket text-primary-600 text-xl"></i>
                                </div>
                            </div>
                            <h3 class="font-semibold text-gray-900 mb-2 text-center">3. Deploy</h3>
                            <p class="text-gray-600 text-sm text-center">Get your snapshot URL and deploy instantly to your Gaia Node infrastructure</p>
                        </div>
                    </div>

                    <!-- Model Information -->
                    <div class="bg-gray-50 rounded-lg p-6 mb-6">
                        <h3 class="text-lg font-semibold text-gray-900 mb-4">Embedding Model</h3>
                        <div class="flex items-start space-x-4">
                            <div class="flex-shrink-0">
                                <div class="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                                    <i class="fas fa-robot text-white"></i>
                                </div>
                            </div>
                            <div>
                                <h4 class="font-medium text-gray-900">gte-Qwen2-1.5B-instruct</h4>
                                <p class="text-sm text-gray-600 mt-1">
                                    State-of-the-art embedding model with 1536-dimensional vectors, optimized for semantic search and knowledge retrieval.
                                </p>
                                <ul class="text-sm text-gray-600 mt-2 space-y-1">
                                    <li><i class="fas fa-check-circle text-green-500 mr-2"></i>1536-dimensional embeddings</li>
                                    <li><i class="fas fa-check-circle text-green-500 mr-2"></i>Optimized for semantic search</li>
                                    <li><i class="fas fa-check-circle text-green-500 mr-2"></i>Multi-format document support</li>
                                </ul>
                            </div>
                        </div>
                    </div>

                    <!-- Best Practices -->
                    <div class="bg-blue-50 rounded-lg p-6">
                        <h3 class="text-lg font-semibold text-gray-900 mb-4">Best Practices</h3>
                        <div class="space-y-3">
                            <div class="flex items-start">
                                <div class="flex-shrink-0">
                                    <i class="fas fa-lightbulb text-yellow-500 mt-1"></i>
                                </div>
                                <div class="ml-3">
                                    <p class="text-sm font-medium text-gray-900">File Preparation</p>
                                    <p class="text-sm text-gray-600">Clean your documents and ensure proper formatting for optimal embedding quality</p>
                                </div>
                            </div>
                            <div class="flex items-start">
                                <div class="flex-shrink-0">
                                    <i class="fas fa-file-alt text-blue-500 mt-1"></i>
                                </div>
                                <div class="ml-3">
                                    <p class="text-sm font-medium text-gray-900">Supported Formats</p>
                                    <p class="text-sm text-gray-600">TXT (plain text), MD (Markdown), PDF (converted to text), CSV (tabular data)</p>
                                </div>
                            </div>
                            <div class="flex items-start">
                                <div class="flex-shrink-0">
                                    <i class="fas fa-database text-green-500 mt-1"></i>
                                </div>
                                <div class="ml-3">
                                    <p class="text-sm font-medium text-gray-900">Batch Processing</p>
                                    <p class="text-sm text-gray-600">Process multiple related files together for better contextual understanding</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Footer -->
                <div class="text-center text-gray-600 text-sm pt-6">
                    <p>Powered by <a href="https://gaianet.ai" target="_blank" class="text-primary-600 hover:text-primary-800 font-medium">Gaia Network</a> - Open Source AI Infrastructure</p>
                    <p class="mt-1">Version 1.0.0 | <a href="https://github.com/gaia-network" target="_blank" class="text-primary-600 hover:text-primary-800">GitHub</a> | <a href="https://docs.gaianet.ai" target="_blank" class="text-primary-600 hover:text-primary-800">Documentation</a></p>
                </div>
            </div>
        </div>

        <script>
            const dropZone = document.getElementById('dropZone');
            const fileInput = document.getElementById('fileInput');
            const fileList = document.getElementById('fileList');
            const processBtn = document.getElementById('processBtn');
            const progressSection = document.getElementById('progressSection');
            const progressFill = document.getElementById('progressFill');
            const progressText = document.getElementById('progressText');
            const resultSection = document.getElementById('resultSection');
            const snapshotUrl = document.getElementById('snapshotUrl');
            const errorSection = document.getElementById('errorSection');
            const errorMessage = document.getElementById('errorMessage');
            const wasmWarning = document.getElementById('wasmWarning');
            const deployButton = document.getElementById('deploy-do');

            let files = [];

            // Check if WasmEdge is available
            fetch('/check-wasm')
                .then(response => response.json())
                .then(data => {
                    if (!data.available) {
                        wasmWarning.classList.remove('hidden');
                    }
                });

            // Drag and drop handlers
            dropZone.addEventListener('click', () => fileInput.click());

            dropZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                dropZone.classList.add('dragover');
            });

            dropZone.addEventListener('dragleave', () => {
                dropZone.classList.remove('dragover');
            });

            dropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                dropZone.classList.remove('dragover');
                handleFiles(e.dataTransfer.files);
            });

            fileInput.addEventListener('change', () => {
                handleFiles(fileInput.files);
            });

            deployButton.addEventListener('click', async function() {
                const snapshotUrl = document.getElementById('snapshotUrl').textContent;
                if (!snapshotUrl) {
                    showError('No snapshot URL available for deployment');
                    return;
                }
                
                deployButton.disabled = true;
                deployButton.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i> Deploying...';
                
                // Show deployment status
                showDeploymentStatus('Starting...');
                
                try {
                    const response = await fetch(`/deploy?snapshot_url=${encodeURIComponent(snapshotUrl)}&user_id=user_${Date.now()}`, {
                        method: "POST"
                    });
                    
                    if (!response.ok) {
                        throw new Error('Deployment failed to start');
                    }
                    
                    const data = await response.json();
                    const dropletId = data.droplet_id;
                    
                    showToast('Deployment started! Droplet ID: ' + dropletId);
                    updateDeploymentStatus('Droplet created. Starting installation...', 'info');
                    showDeploymentStatus(dropletId);
                    
                    // Open deployment status page
                    const statusWindow = window.open(`/deployment-status?droplet_id=${dropletId}`, '_blank');
                    
                    // Poll for deployment completion
                    const pollInterval = setInterval(async () => {
                        try {
                            const response = await fetch(`/status/${dropletId}`);
                            const statusData = await response.json();
                            
                            // Update status message
                            if (statusData.status) {
                                updateDeploymentStatus(`Status: ${statusData.status}${statusData.ip ? ', IP: ' + statusData.ip : ''}`, 'info');
                            }
                            
                            if (statusData.gaia_url) {
                                // Deployment completed successfully!
                                clearInterval(pollInterval);
                                
                                deployButton.disabled = false;
                                deployButton.innerHTML = '<i class="fas fa-check mr-2"></i> Deployment Complete';
                                deployButton.classList.remove('bg-blue-600', 'hover:bg-blue-700');
                                deployButton.classList.add('bg-green-600', 'hover:bg-green-700');
                                
                                updateDeploymentStatus('Deployment completed successfully!', 'success');
                                
                                // Show the Gaia URL
                                const gaiaUrl = statusData.gaia_url;
                                snapshotUrl.href = gaiaUrl;
                                snapshotUrl.textContent = gaiaUrl;
                                
                                // Create open button
                                const openButton = document.createElement('button');
                                openButton.className = 'ml-4 px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg';
                                openButton.innerHTML = '<i class="fas fa-external-link-alt mr-2"></i> Open Gaia Node';
                                openButton.onclick = () => window.open(gaiaUrl, '_blank');
                                
                                deployButton.parentNode.appendChild(openButton);
                                
                                // Also update the status window if it's still open
                                if (statusWindow && !statusWindow.closed) {
                                    statusWindow.location.reload();
                                }
                            }
                            
                        } catch (error) {
                            console.error('Error polling deployment status:', error);
                        }
                    }, 5000); // Check every 5 seconds
                    
                    // Set a timeout to stop polling after 15 minutes
                    setTimeout(() => {
                        clearInterval(pollInterval);
                        if (deployButton.disabled) {
                            deployButton.disabled = false;
                            deployButton.innerHTML = '<i class="fas fa-cloud mr-2"></i> Deploy to DigitalOcean';
                            updateDeploymentStatus('Deployment timed out after 15 minutes. Check the status window for details.', 'error');
                        }
                    }, 15 * 60 * 1000); // 15 minutes
                    
                } catch (error) {
                    showError('Deployment failed: ' + error.message);
                    deployButton.disabled = false;
                    deployButton.innerHTML = '<i class="fas fa-cloud mr-2"></i> Deploy to DigitalOcean';
                    hideDeploymentStatus();
                }
            });
            
            // Show deploy button when snapshot is ready
            function showDeployButton() {
                deployButton.classList.remove('hidden');
            }
            
            function handleFiles(fileList) {
                for (let i = 0; i < fileList.length; i++) {
                    const file = fileList[i];
                    
                    // Check file size (10MB limit)
                    if (file.size > 10 * 1024 * 1024) {
                        showError(`File ${file.name} exceeds 10MB limit`);
                        continue;
                    }
                    
                    // Check file type
                    const ext = file.name.split('.').pop().toLowerCase();
                    if (!['txt', 'md', 'pdf', 'csv'].includes(ext)) {
                        showError(`File type ${ext} not supported. Please use TXT, MD, PDF, or CSV files.`);
                        continue;
                    }
                    
                    // Add to files list
                    if (!files.some(f => f.name === file.name && f.size === file.size)) {
                        files.push(file);
                        addFileToList(file);
                    }
                }
                
                updateProcessButton();
            }

            function addFileToList(file) {
                const fileItem = document.createElement('div');
                fileItem.className = 'flex items-center justify-between bg-gray-50 p-3 rounded-lg';
                fileItem.innerHTML = `
                    <div class="flex items-center space-x-3">
                        <span class="text-primary-600">
                            <i class="fas fa-file"></i>
                        </span>
                        <div>
                            <div class="font-medium text-gray-800">${file.name}</div>
                            <div class="text-sm text-gray-500">${formatFileSize(file.size)}</div>
                        </div>
                    </div>
                    <button class="text-red-500 hover:text-red-700" onclick="removeFile('${file.name}', ${file.size})">
                        <i class="fas fa-times"></i>
                    </button>
                `;
                fileList.appendChild(fileItem);
            }

            function formatFileSize(bytes) {
                if (bytes < 1024) return bytes + ' B';
                else if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
                else return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
            }

            function removeFile(name, size) {
                files = files.filter(f => !(f.name === name && f.size === size));
                
                // Rebuild file list
                fileList.innerHTML = '';
                files.forEach(addFileToList);
                
                updateProcessButton();
            }

            function updateProcessButton() {
                processBtn.disabled = files.length === 0;
                processBtn.innerHTML = files.length > 0 ? 
                    `<i class="fas fa-bolt mr-2"></i> Generate Snapshot (${files.length} file${files.length > 1 ? 's' : ''})` : 
                    '<i class="fas fa-bolt mr-2"></i> Generate Snapshot';
            }

            function showError(message) {
                errorMessage.textContent = message;
                errorSection.classList.remove('hidden');
                setTimeout(() => {
                    errorSection.classList.add('hidden');
                }, 5000);
            }

            function showDeploymentStatus(dropletId) {
                const statusElement = document.getElementById('deploymentStatus');
                const dropletIdElement = document.getElementById('currentDropletId');
                
                dropletIdElement.textContent = dropletId;
                statusElement.classList.remove('hidden');
                
                // Scroll to status
                statusElement.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }

            // Function to hide deployment status
            function hideDeploymentStatus() {
                const statusElement = document.getElementById('deploymentStatus');
                statusElement.classList.add('hidden');
            }

            // Function to update deployment status message
            function updateDeploymentStatus(message, type = 'info') {
                const statusElement = document.getElementById('deploymentStatus');
                const iconElement = statusElement.querySelector('.fa-sync-alt');
                const messageElement = statusElement.querySelector('.text-blue-800');
                const detailsElement = statusElement.querySelector('.text-blue-600');
                
                if (type === 'success') {
                    iconElement.className = 'fas fa-check-circle text-green-500';
                    statusElement.className = 'mt-4 bg-green-50 border border-green-200 rounded-lg p-4';
                    messageElement.className = 'text-sm font-medium text-green-800';
                    detailsElement.className = 'text-sm text-green-600';
                } else if (type === 'error') {
                    iconElement.className = 'fas fa-exclamation-circle text-red-500';
                    statusElement.className = 'mt-4 bg-red-50 border border-red-200 rounded-lg p-4';
                    messageElement.className = 'text-sm font-medium text-red-800';
                    detailsElement.className = 'text-sm text-red-600';
                }
                
                detailsElement.textContent = message;
            }

            // Process files
            processBtn.addEventListener('click', async () => {
                progressSection.classList.remove('hidden');
                resultSection.classList.add('hidden');
                errorSection.classList.add('hidden');
                processBtn.disabled = true;
                processBtn.classList.add('processing');
                
                const session_id = 'session_' + Date.now();
                
                const formData = new FormData();
                files.forEach(file => formData.append('files', file));
                formData.append('session_id', session_id);
                
                try {
                    const eventSource = new EventSource(`/process-stream?session_id=${session_id}`);
                    eventSource.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        updateProgress(data.percent, data.message, data.step);
                        
                        if (data.percent === 100) {
                            eventSource.close();
                            if (data.snapshot_url) {
                                snapshotUrl.href = data.snapshot_url;
                                snapshotUrl.textContent = data.snapshot_url;
                                resultSection.classList.remove('hidden');
                                updateConfigCommand(data.snapshot_url);
                                window.showDeployButton();
                            }
                            processBtn.disabled = false;
                            processBtn.classList.remove('processing');
                        }
                    };
                    
                    eventSource.onerror = function(error) {
                        console.error('EventSource failed:', error);
                        eventSource.close();
                        showError('Connection error during processing');
                        processBtn.disabled = false;
                        processBtn.classList.remove('processing');
                    };
                    
                    const response = await fetch('/process', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (!response.ok) {
                        const error = await response.text();
                        throw new Error(error);
                    }
                    
                } catch (error) {
                    showError('Error processing files: ' + error.message);
                    processBtn.disabled = false;
                    processBtn.classList.remove('processing');
                }
            });

            function updateProgress(percent, message, step) {
                progressFill.style.width = percent + '%';
                progressText.textContent = message;
                
                if (step) {
                    const stepElement = document.createElement('div');
                    stepElement.innerHTML = `<span class="text-gray-400">${new Date().toLocaleTimeString()}</span>: ${step}`;
                    document.getElementById('stepDetails').appendChild(stepElement);
                    document.getElementById('stepDetails').scrollTop = document.getElementById('stepDetails').scrollHeight;
                }
            }

            // Function to update the config command with the actual URL
            function updateConfigCommand(snapshotUrl) {
                const configCommandElement = document.getElementById('configCommand');
                configCommandElement.innerHTML = `
                    gaianet config --snapshot <span class="text-yellow-300">${snapshotUrl}</span><br>
                    gaianet config --embedding-url <span class="text-yellow-300">https://huggingface.co/gaianet/gte-Qwen2-1.5B-instruct-GGUF/resolve/main/gte-Qwen2-1.5B-instruct-f16.gguf</span><br>
                    gaianet config embedding-ctx-size <span class="text-yellow-300">8192</span>
                `;
            }

            function copySnapshotUrl() {
                const url = document.getElementById('snapshotUrl').textContent;
                navigator.clipboard.writeText(url).then(() => {
                    showToast('Snapshot URL copied to clipboard!');
                }).catch(err => {
                    console.error('Failed to copy: ', err);
                });
            }

            function copyConfigCommand() {
                const snapshotUrl = document.getElementById('snapshotUrl').textContent;
                const configCommand = `gaianet config --snapshot ${snapshotUrl}
gaianet config --embedding-url https://huggingface.co/gaianet/gte-Qwen2-1.5B-instruct-GGUF/resolve/main/gte-Qwen2-1.5B-instruct-f16.gguf
gaianet config embedding-ctx-size 8192`;

                navigator.clipboard.writeText(configCommand).then(() => {
                    showToast('Config commands copied to clipboard!');
                }).catch(err => {
                    console.error('Failed to copy: ', err);
                });
            }

            function showToast(message) {
                const toast = document.createElement('div');
                toast.className = 'fixed bottom-4 right-4 bg-gray-800 text-white px-4 py-2 rounded-lg shadow-lg z-50';
                toast.textContent = message;
                document.body.appendChild(toast);
                
                setTimeout(() => {
                    toast.remove();
                }, 3000);
            }
            async function checkDeploymentStatus(dropletId) {
                try {
                    const response = await fetch(`/status/${dropletId}`);
                    const data = await response.json();
                    
                    if (data.gaia_url) {
                        // Deployment completed successfully!
                        deployButton.disabled = false;
                        deployButton.innerHTML = '<i class="fas fa-check mr-2"></i> Deployment Complete';
                        deployButton.classList.remove('bg-blue-600', 'hover:bg-blue-700');
                        deployButton.classList.add('bg-green-600', 'hover:bg-green-700');
                        
                        // Show the Gaia URL
                        const gaiaUrlElement = document.createElement('div');
                        gaiaUrlElement.className = 'mt-4 p-4 bg-green-50 border border-green-200 rounded-lg';
                        gaiaUrlElement.innerHTML = `
                            <h4 class="font-semibold text-green-800 mb-2">🚀 Your Gaia Node is Ready!</h4>
                            <p class="text-green-700 mb-2">Open your node and start chatting:</p>
                            <a href="${data.gaia_url}" target="_blank" class="text-primary-600 hover:text-primary-800 font-medium break-all">
                                ${data.gaia_url}
                            </a>
                            <button onclick="copyGaiaUrl('${data.gaia_url}')" class="ml-2 px-3 py-1 bg-green-600 hover:bg-green-700 text-white rounded text-sm">
                                <i class="fas fa-copy mr-1"></i> Copy URL
                            </button>
                        `;
                        
                        // Insert after the deploy button
                        deployButton.parentNode.insertBefore(gaiaUrlElement, deployButton.nextSibling);
                        
                        // Also update the result section with the URL
                        snapshotUrl.href = data.gaia_url;
                        snapshotUrl.textContent = data.gaia_url;
                        
                        return true;
                    }
                    
                    return false;
                    
                } catch (error) {
                    console.error('Error checking deployment status:', error);
                    return false;
                }
            }

            // Add this function to copy the Gaia URL
            function copyGaiaUrl(url) {
                navigator.clipboard.writeText(url).then(() => {
                    showToast('Gaia URL copied to clipboard!');
                }).catch(err => {
                    console.error('Failed to copy: ', err);
                });
            }

            window.removeFile = removeFile;
            window.copySnapshotUrl = copySnapshotUrl;
            window.copyConfigCommand = copyConfigCommand;
            window.showDeployButton = showDeployButton;
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/deployment-status")
async def deployment_status(request: Request):
    return templates.TemplateResponse("deployment-status.html", {"request": request})
    
@app.get("/check-wasm")
async def check_wasm():
    """Check if WasmEdge is available"""
    return {"available": wasmedge_available}

@app.post("/process")
async def process_files(
    files: List[UploadFile] = File(...),
    session_id: str = Form(None)  # Add session_id parameter
):
    try:
        # Generate session_id if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Initialize progress tracking
        progress_events[session_id] = {"percent": 0, "message": "Starting...", "step": "Initializing"}
        
        # Check if Qdrant is available
        if qdrant_client is None:
            raise HTTPException(status_code=500, detail="Qdrant database not available")
        
        # Update progress
        progress_events[session_id] = {"percent": 5, "message": "Saving files...", "step": "Saving uploaded files"}
        
        session_dir = UPLOAD_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded files
        saved_files = []
        for file in files:
            if file.size > MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail=f"File {file.filename} exceeds 10MB limit")
            
            file_path = session_dir / file.filename
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            saved_files.append(file_path)
        
        # Update progress
        progress_events[session_id] = {"percent": 10, "message": "Processing files...", "step": "Processing file types"}
        
        # Process files based on type
        processed_files = []
        for file_path in saved_files:
            ext = file_path.suffix.lower()
            
            if ext == '.pdf':
                progress_events[session_id] = {"percent": 15, "message": "Converting PDF...", "step": f"Converting {file_path.name} to Markdown"}
                md_path = file_path.with_suffix('.md')
                await convert_pdf_to_md(file_path, md_path)
                processed_files.append(('md', md_path))
            else:
                processed_files.append((ext[1:], file_path))
        
        # Update progress
        progress_events[session_id] = {"percent": 20, "message": "Creating collection...", "step": "Setting up Qdrant collection"}
        # Always use "default" collection name as required by gaianet
        global COLLECTION_NAME
        COLLECTION_NAME = "default"

        # Create the collection
        try:
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection '{COLLECTION_NAME}' with vector size {VECTOR_SIZE}")
        except Exception as e:
            logger.warning(f"Collection may already exist or creation failed: {e}")
            # Continue anyway - the collection might already exist
        # === END OF ADDED SECTION ===
        
        # Update progress for embedding generation
        total_embeddings = 0
        successful_files = 0
        failed_files = []

        for i, (file_type, file_path) in enumerate(processed_files):
            progress_percent = 20 + (i * 50 / len(processed_files))
            progress_events[session_id] = {
                "percent": progress_percent, 
                "message": f"Processing {file_path.name}...", 
                "step": f"Generating embeddings for {file_path.name}"
            }
            
            try:
                if wasmedge_available:
                    embeddings_count = await generate_embeddings_wasmedge(file_type, file_path)
                else:
                    embeddings_count = await generate_embeddings_fallback(file_type, file_path)
                total_embeddings += embeddings_count
                successful_files += 1
                logger.info(f"Successfully processed {file_path} with {embeddings_count} embeddings")
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                failed_files.append(f"{file_path.name}: {str(e)}")
        
        # Update progress
        progress_events[session_id] = {"percent": 70, "message": "Creating snapshot...", "step": "Creating Qdrant snapshot"}
        
        # Create snapshot
        snapshot_name = f"snapshot-{session_id}"
        snapshot_file = await create_qdrant_snapshot(snapshot_name)
        
        # Update progress
        progress_events[session_id] = {"percent": 80, "message": "Compressing...", "step": "Compressing snapshot"}
        
        # Compress snapshot
        compressed_snapshot = await compress_snapshot(snapshot_file)
        
        # Update progress
        progress_events[session_id] = {"percent": 90, "message": "Uploading to Hugging Face...", "step": "Uploading to Hugging Face"}
        
        # Upload to Hugging Face
        snapshot_url = await upload_to_huggingface(compressed_snapshot, snapshot_name)
        
        # Update progress
        progress_events[session_id] = {"percent": 95, "message": "Cleaning up...", "step": "Cleaning up temporary files"}
        
        # Clean up ALL local files
        await cleanup_all_files(session_dir, compressed_snapshot, snapshot_file)
        await cleanup_qdrant_collection()
        
        # Final progress update
        progress_events[session_id] = {
            "percent": 100, 
            "message": "Complete!", 
            "step": "Process completed successfully",
            "snapshot_url": snapshot_url
        }
        
        # Prepare response
        response_data = {
            "status": "success",
            "snapshot_url": snapshot_url,
            "message": f"Processed {successful_files} files with {total_embeddings} embeddings",
            "deploy_endpoint": f"/deploy?snapshot_url={snapshot_url}&user_id={session_id}"
        }

        if failed_files:
            response_data["warnings"] = f"{len(failed_files)} files failed: {', '.join(failed_files)}"

        return JSONResponse(response_data)
    
    except Exception as e:
        logger.error(f"Error processing files: {e}")
        # Clean up on error too
        if 'session_dir' in locals():
            await cleanup_all_files(session_dir, None, None)
        # Also clean up Qdrant collection on error
        await cleanup_qdrant_collection()
        raise HTTPException(status_code=500, detail=str(e))

async def convert_pdf_to_md(pdf_path: Path, md_path: Path):
    """Convert PDF to Markdown using markitdown"""
    try:
        result = subprocess.run([
            "markitdown", str(pdf_path), "-o", str(md_path)
        ], capture_output=True, text=True, check=True)
        
        logger.info(f"PDF to MD conversion: {result.stdout}")
        if result.stderr:
            logger.warning(f"PDF to MD conversion warnings: {result.stderr}")
            
        return md_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting PDF to MD: {e.stderr}")
        raise HTTPException(status_code=500, detail=f"PDF conversion failed: {e.stderr}")

async def generate_embeddings_wasmedge(file_type: str, file_path: Path) -> int:
    """Generate embeddings using WasmEdge and the appropriate WASM script"""
    try:
        # Determine the appropriate WASM script and parameters
        if file_type == 'csv':
            wasm_script = WASM_DIR / "csv_embed.wasm"
            args = [str(file_path), "--ctx_size", "8192"]
        elif file_type == 'md':
            wasm_script = WASM_DIR / "markdown_embed.wasm"
            args = [str(file_path), "--heading_level", "1", "--ctx_size", "8192"]
        elif file_type == 'txt':
            wasm_script = WASM_DIR / "paragraph_embed.wasm"
            args = [str(file_path), "-c", "8192"]
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_type}")
        
        # Check if WASM script exists
        if not wasm_script.exists():
            raise HTTPException(status_code=500, detail=f"WASM script not found: {wasm_script}")
        
        # Check if model exists
        model_path = MODEL_DIR / "gte-Qwen2-1.5B-instruct-f16.gguf"
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            raise HTTPException(status_code=500, detail=f"Model file not found: {model_path.name}")
        
        # Build WasmEdge command
        cmd = [
            "wasmedge",
            "--dir", f".:{str(file_path.parent.absolute())}",
            "--dir", f"models:{MODEL_DIR.absolute()}",
            "--nn-preload", f"embedding:GGML:AUTO:{model_path.name}",  # Use just filename
            str(wasm_script.absolute()),
            "embedding",  # Model name
            COLLECTION_NAME,  # Collection name
            str(VECTOR_SIZE),  # Vector size
            *args
        ]

        
        logger.info(f"Running WasmEdge command: {' '.join(cmd)}")
        
        # Run WasmEdge with timeout
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=Path.cwd(), timeout=300)
        
        logger.info(f"WasmEdge execution completed: {result.stdout[:200]}...")
        if result.stderr:
            logger.warning(f"WasmEdge execution warnings: {result.stderr}")
        
        # Parse output to get number of embeddings created
        lines = result.stdout.split('\n')
        for line in lines:
            if "embeddings created" in line.lower() or "vectors created" in line.lower():
                try:
                    parts = line.split()
                    count = int(parts[0])
                    return count
                except (ValueError, IndexError):
                    pass
        
        # If we can't parse the count, estimate based on file size
        return estimate_embedding_count(file_path, file_type)
            
    except subprocess.TimeoutExpired:
        logger.error("WasmEdge execution timed out after 5 minutes")
        raise HTTPException(status_code=500, detail="Embedding generation timed out")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running WasmEdge: {e.stderr}")
        # Fallback to simple embeddings
        return await generate_embeddings_fallback(file_type, file_path)
    except Exception as e:
        logger.error(f"Unexpected error in embedding generation: {e}")
        # Fallback to simple embeddings
        return await generate_embeddings_fallback(file_type, file_path)

async def generate_embeddings_fallback(file_type: str, file_path: Path) -> int:
    """Fallback embedding generation using simple method with batching"""
    try:
        # Read file content
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
        
        # Extract texts based on file type
        texts = []
        if file_type == 'txt':
            paragraphs = content.split('\n\n')
            texts = [para.strip() for para in paragraphs if para.strip()]
        elif file_type == 'md':
            sections = content.split('\n# ')
            texts = [section.strip() for section in sections if section.strip()]
        elif file_type == 'csv':
            lines = content.split('\n')
            texts = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
        
        # Generate simple embeddings
        points = []
        for i, text in enumerate(texts):
            embedding = [0.0] * VECTOR_SIZE
            for j, char in enumerate(text[:VECTOR_SIZE]):
                embedding[j % VECTOR_SIZE] = (embedding[j % VECTOR_SIZE] + ord(char)) / 255.0
            
            # Normalize
            norm = (sum(e**2 for e in embedding)) ** 0.5
            if norm > 0:
                embedding = [e / norm for e in embedding]
            
            points.append(PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "text": text,
                    "file_type": file_type,
                    "file_name": file_path.name,
                    "session_id": str(uuid.uuid4())
                }
            ))
        
        # Store in Qdrant with batching to avoid timeouts
        if points:
            batch_size = 100  # Process in smaller batches
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        qdrant_client.upsert(
                            collection_name=COLLECTION_NAME, 
                            points=batch
                        )
                        logger.info(f"Successfully upserted batch {i//batch_size + 1}")
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            logger.warning(f"Qdrant upsert failed (attempt {attempt + 1}/{max_retries}): {e}")
                            await asyncio.sleep(2)
                        else:
                            logger.error(f"Qdrant upsert failed after {max_retries} attempts: {e}")
                            # Save this batch to file
                            await save_embeddings_to_file(batch, file_path, f"_batch_{i//batch_size + 1}")
        
        logger.info(f"Generated {len(points)} fallback embeddings for {file_type} file")
        return len(points)
        
    except Exception as e:
        logger.error(f"Error in fallback embedding generation: {e}")
        raise HTTPException(status_code=500, detail=f"Fallback embedding generation failed: {str(e)}")

async def save_embeddings_to_file(points: List[PointStruct], file_path: Path, suffix: str = ""):
    """Save embeddings to file when Qdrant is unavailable"""
    try:
        # Create embeddings directory if it doesn't exist
        EMBEDDING_DIR.mkdir(exist_ok=True)
        
        # Create a filename based on the original file
        embedding_file = EMBEDDING_DIR / f"{file_path.stem}_embeddings{suffix}.json"
        
        # Convert points to serializable format
        embeddings_data = []
        for point in points:
            embeddings_data.append({
                "id": point.id,
                "vector": point.vector,
                "payload": point.payload
            })
        
        # Save to file
        async with aiofiles.open(embedding_file, 'w') as f:
            await f.write(json.dumps(embeddings_data, indent=2))
        
        logger.info(f"Saved {len(points)} embeddings to file: {embedding_file}")
        
    except Exception as e:
        logger.error(f"Error saving embeddings to file: {e}")

def estimate_embedding_count(file_path: Path, file_type: str) -> int:
    """Estimate the number of embeddings based on file content"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if file_type == 'txt':
            # Count paragraphs
            paragraphs = content.split('\n\n')
            return len([p for p in paragraphs if p.strip()])
        elif file_type == 'md':
            # Count sections (headers)
            sections = content.split('\n# ')
            return len([s for s in sections if s.strip()])
        elif file_type == 'csv':
            # Count rows
            lines = content.split('\n')
            return len([l for l in lines if l.strip() and not l.startswith('#')])
        else:
            return 1  # Default estimate
    except:
        return 1  # Fallback estimate


async def create_qdrant_snapshot(snapshot_name: str) -> Path:
    """Create a Qdrant snapshot - ensure it's always a proper snapshot file"""
    try:
        logger.info(f"Creating snapshot '{snapshot_name}' for collection '{COLLECTION_NAME}'")

        # Always try to create a proper Qdrant snapshot first
        if QDRANT_API_KEY:  # We're using cloud Qdrant
            try:
                return await create_proper_cloud_snapshot(snapshot_name)
            except Exception as cloud_error:
                logger.error(f"Cloud snapshot failed, trying direct API: {cloud_error}")
                return await create_snapshot_via_direct_api(snapshot_name)
        else:  # Local Qdrant
            return await create_snapshot_via_direct_api(snapshot_name)

    except Exception as e:
        logger.error(f"Error creating Qdrant snapshot: {e}")
        raise HTTPException(status_code=500, detail=f"Snapshot creation failed: {str(e)}")

async def create_proper_cloud_snapshot(snapshot_name: str) -> Path:
    """Create snapshot using Qdrant client with proper error handling"""
    try:
        logger.info(f"Creating proper cloud snapshot for '{COLLECTION_NAME}'")
        
        # Create snapshot using Qdrant client
        snapshot_info = qdrant_client.create_snapshot(
            collection_name=COLLECTION_NAME
        )
        
        logger.info(f"Snapshot created: {snapshot_info}")
        
        # Get the actual snapshot name
        actual_snapshot_name = snapshot_info.name
        logger.info(f"Snapshot name: {actual_snapshot_name}")
        
        # Wait a moment for snapshot to be ready
        await asyncio.sleep(5)
        
        # Download using direct API
        return await download_snapshot_via_api(actual_snapshot_name)
        
    except Exception as e:
        logger.error(f"Proper cloud snapshot failed: {e}")
        raise

async def download_snapshot_via_api(snapshot_name: str) -> Path:
    """Download snapshot using wget-style approach as shown in Qdrant docs"""
    try:
        logger.info(f"Downloading snapshot '{snapshot_name}' using wget approach")
        
        download_url = f"{QDRANT_URL}/collections/{COLLECTION_NAME}/snapshots/{snapshot_name}"
        headers = {}
        if QDRANT_API_KEY:
            headers["api-key"] = QDRANT_API_KEY
        
        logger.info(f"Download URL: {download_url}")
        
        # Use the exact same approach as wget command in documentation
        # Wait a bit for the snapshot to be fully ready
        await asyncio.sleep(10)
        
        # Try multiple times with proper headers
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Download attempt {attempt + 1}/{max_retries}")
                
                # Use stream=True and proper timeout
                response = requests.get(
                    download_url, 
                    headers=headers, 
                    stream=True, 
                    timeout=60,  # Longer timeout for large files
                    verify=True  # Ensure SSL verification
                )
                
                # Check if we got a successful response
                if response.status_code == 200:
                    # Create the snapshot file
                    snapshot_file = SNAPSHOT_DIR / snapshot_name
                    
                    # Download with progress
                    total_size = int(response.headers.get('content-length', 0))
                    logger.info(f"Content-length: {total_size} bytes")
                    
                    # Download the file
                    downloaded_size = 0
                    with open(snapshot_file, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded_size += len(chunk)
                    
                    # Verify the download
                    if snapshot_file.exists() and snapshot_file.stat().st_size > 0:
                        file_size = snapshot_file.stat().st_size
                        logger.info(f"Successfully downloaded {file_size} bytes")
                        
                        # For cloud snapshots, we expect large files
                        if file_size > 1024:  # Reasonable minimum
                            logger.info(f"Snapshot downloaded successfully: {snapshot_file}")
                            return snapshot_file
                        else:
                            logger.warning(f"File too small ({file_size} bytes), might be incomplete")
                            snapshot_file.unlink()  # Remove the small file
                            if attempt < max_retries - 1:
                                await asyncio.sleep(10)
                                continue
                    else:
                        raise Exception("Downloaded file is empty or doesn't exist")
                
                elif response.status_code == 404:
                    logger.warning(f"Snapshot not found (404), might need more time to be ready")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(15)
                        continue
                    else:
                        raise Exception("Snapshot not found after multiple attempts")
                
                else:
                    response.raise_for_status()
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(10)
                else:
                    raise
        
        raise Exception("All download attempts failed")
            
    except Exception as e:
        logger.error(f"Snapshot download failed: {e}")
        # Try one more approach as a last resort
        return await download_snapshot_fallback(snapshot_name)

async def download_snapshot_fallback(snapshot_name: str) -> Path:
    """Fallback download method using subprocess with wget/curl"""
    try:
        logger.info(f"Trying fallback download with wget/curl for '{snapshot_name}'")
        
        download_url = f"{QDRANT_URL}/collections/{COLLECTION_NAME}/snapshots/{snapshot_name}"
        snapshot_file = SNAPSHOT_DIR / snapshot_name
        
        # Try wget first (as shown in documentation)
        try:
            cmd = [
                'wget', download_url,
                '--header', f'api-key: {QDRANT_API_KEY}',
                '-O', str(snapshot_file),
                '--timeout=60',
                '--tries=3'
            ]
            
            logger.info(f"Running wget command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0 and snapshot_file.exists() and snapshot_file.stat().st_size > 0:
                logger.info(f"wget download successful: {snapshot_file}")
                return snapshot_file
            else:
                logger.warning(f"wget failed: {result.stderr}")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.info("wget not available or timed out")
        
        # Try curl as alternative
        try:
            cmd = [
                'curl', download_url,
                '-H', f'api-key: {QDRANT_API_KEY}',
                '-o', str(snapshot_file),
                '--connect-timeout', '30',
                '--max-time', '120'
            ]
            
            logger.info(f"Running curl command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0 and snapshot_file.exists() and snapshot_file.stat().st_size > 0:
                logger.info(f"curl download successful: {snapshot_file}")
                return snapshot_file
            else:
                logger.warning(f"curl failed: {result.stderr}")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.info("curl not available or timed out")
        
        raise Exception("All fallback download methods failed")
        
    except Exception as e:
        logger.error(f"Fallback download also failed: {e}")
        raise

async def is_valid_qdrant_snapshot(snapshot_file: Path) -> bool:
    """Simple validation - just check file exists and has reasonable size"""
    try:
        if not snapshot_file.exists():
            return False
            
        file_size = snapshot_file.stat().st_size
        # Qdrant snapshots should be at least a few KB
        return file_size > 1024
        
    except Exception:
        return False

async def is_tar_file(file_path: Path) -> bool:
    """Check if a file is a valid tar archive"""
    try:
        with tarfile.open(file_path, 'r:*') as tar:
            return True
    except (tarfile.ReadError, OSError):
        return False

async def create_snapshot_via_direct_api(snapshot_name: str) -> Path:
    """Create snapshot using direct HTTP API calls"""
    try:
        logger.info(f"Creating snapshot via direct API for '{COLLECTION_NAME}'")
        
        # Create snapshot using direct HTTP API
        create_url = f"{QDRANT_URL}/collections/{COLLECTION_NAME}/snapshots"
        headers = {"Content-Type": "application/json"}
        if QDRANT_API_KEY:
            headers["api-key"] = QDRANT_API_KEY
        
        logger.info(f"POST to: {create_url}")
        response = requests.post(create_url, headers=headers, timeout=30)
        
        # Log the full response for debugging
        logger.info(f"Response status: {response.status_code}")
        logger.info(f"Response headers: {dict(response.headers)}")
        
        response.raise_for_status()
        
        snapshot_info = response.json()
        logger.info(f"Snapshot creation response: {snapshot_info}")
        
        # Extract snapshot name from response
        actual_snapshot_name = snapshot_info['result']['name']
        snapshot_size = snapshot_info['result'].get('size', 0)
        logger.info(f"Snapshot created: {actual_snapshot_name}, size: {snapshot_size} bytes")
        
        # Wait longer for large snapshots
        wait_time = max(15, min(snapshot_size // 1000000, 60))  # 1 second per MB, max 60s
        logger.info(f"Waiting {wait_time}s for snapshot to be ready...")
        await asyncio.sleep(wait_time)
        
        # Download the snapshot
        return await download_snapshot_via_api(actual_snapshot_name)
        
    except Exception as e:
        logger.error(f"Direct API snapshot creation failed: {e}")
        # Log more details about the error
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Error response: {e.response.text}")
        raise

async def create_manual_snapshot(snapshot_name: str) -> Path:
    """Create a manual snapshot that Qdrant can actually import"""
    try:
        logger.info(f"Creating manual snapshot for collection '{COLLECTION_NAME}'")
        
        # Get collection info to understand the structure
        try:
            collection_info = qdrant_client.get_collection(COLLECTION_NAME)
            logger.info(f"Collection info: {collection_info}")
        except Exception as e:
            logger.warning(f"Could not get collection info: {e}")
        
        # Get all points with pagination
        all_points = []
        next_offset = None
        batch_count = 0
        
        while True:
            try:
                points, next_offset = qdrant_client.scroll(
                    collection_name=COLLECTION_NAME,
                    limit=1000,
                    offset=next_offset,
                    with_payload=True,
                    with_vectors=True
                )
                
                if points:
                    all_points.extend(points)
                    batch_count += 1
                    logger.info(f"Retrieved batch {batch_count}: {len(points)} points")
                
                if not next_offset or not points:
                    break
                    
            except Exception as e:
                logger.error(f"Error retrieving points: {e}")
                break
        
        logger.info(f"Total points retrieved: {len(all_points)}")
        
        # Create a snapshot file in a format that Qdrant can import
        # For manual snapshots, we'll create a JSON file that can be processed
        snapshot_file = SNAPSHOT_DIR / f"{snapshot_name}.json"
        
        snapshot_data = {
            "collection_name": COLLECTION_NAME,
            "vectors_config": {
                "size": VECTOR_SIZE,
                "distance": "Cosine"
            },
            "points": [
                {
                    "id": point.id,
                    "vector": point.vector,
                    "payload": point.payload
                }
                for point in all_points
            ],
            "total_points": len(all_points),
            "created_at": datetime.now().isoformat()
        }
        
        async with aiofiles.open(snapshot_file, 'w') as f:
            await f.write(json.dumps(snapshot_data, indent=2, default=str))
        
        logger.info(f"Created manual snapshot: {snapshot_file}")
        return snapshot_file
        
    except Exception as e:
        logger.error(f"Error creating manual snapshot: {e}")
        raise HTTPException(status_code=500, detail=f"Manual snapshot creation failed: {str(e)}")

async def create_memory_snapshot(snapshot_name: str) -> Path:
    """Create a snapshot for in-memory Qdrant"""
    try:
        # Get all points from the collection
        points, offset = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=10000,  # Reasonable limit
            with_payload=True,
            with_vectors=True
        )
        
        # Create snapshot data
        snapshot_data = {
            "collection": COLLECTION_NAME,
            "vectors_count": len(points),
            "points_count": len(points),
            "points": [
                {
                    "id": point.id,
                    "vector": point.vector,
                    "payload": point.payload
                }
                for point in points
            ],
            "created_at": str(datetime.now()),
        }
        
        # Save snapshot
        snapshot_file = SNAPSHOT_DIR / f"{snapshot_name}.snapshot"
        async with aiofiles.open(snapshot_file, 'w') as f:
            await f.write(json.dumps(snapshot_data, indent=2, default=str))
        
        logger.info(f"Created memory snapshot: {snapshot_file}")
        return snapshot_file
        
    except Exception as e:
        logger.error(f"Error creating memory snapshot: {e}")
        raise HTTPException(status_code=500, detail=f"Memory snapshot creation failed: {str(e)}")

async def create_cloud_snapshot(snapshot_name: str) -> Path:
    """Create and download snapshot from Qdrant Cloud using the correct API approach"""
    try:
        logger.info(f"Creating cloud snapshot for collection '{COLLECTION_NAME}'")
        
        # Step 1: Create snapshot using Qdrant client (as per documentation)
        snapshot_info = qdrant_client.create_snapshot(
            collection_name=COLLECTION_NAME,
            snapshot_name=snapshot_name
        )
        
        logger.info(f"Snapshot creation response: {snapshot_info}")
        
        # Extract the actual snapshot name from the response
        actual_snapshot_name = snapshot_info.name
        logger.info(f"Snapshot created with name: {actual_snapshot_name}")
        
        # Step 2: Wait a moment for the snapshot to be fully created
        await asyncio.sleep(3)
        
        # Step 3: Download the snapshot using direct HTTP API (as per documentation)
        snapshot_url = f"{QDRANT_URL}/collections/{COLLECTION_NAME}/snapshots/{actual_snapshot_name}"
        logger.info(f"Downloading snapshot from: {snapshot_url}")
        
        # Create headers with API key
        headers = {}
        if QDRANT_API_KEY:
            headers["api-key"] = QDRANT_API_KEY
        
        # Download the snapshot
        response = requests.get(snapshot_url, headers=headers, stream=True)
        response.raise_for_status()
        
        # Create the snapshot file with the exact name from Qdrant
        snapshot_file = SNAPSHOT_DIR / actual_snapshot_name
        
        # Download with progress
        total_size = int(response.headers.get('content-length', 0))
        logger.info(f"Downloading {total_size} bytes to {snapshot_file}")
        
        with open(snapshot_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        # Verify download
        if snapshot_file.exists() and snapshot_file.stat().st_size > 0:
            file_size = snapshot_file.stat().st_size
            logger.info(f"Successfully downloaded snapshot: {snapshot_file} ({file_size} bytes)")
            
            # The downloaded file should be a valid Qdrant snapshot
            # Don't modify it - Qdrant expects this exact format
            return snapshot_file
        else:
            raise Exception("Downloaded snapshot file is empty or doesn't exist")
            
    except Exception as e:
        logger.error(f"Error creating cloud snapshot: {e}")
        # Fallback to manual approach if cloud API fails
        logger.warning("Falling back to manual snapshot creation")
        return await create_manual_snapshot(snapshot_name)

def _is_gzip_bytes(prefix: bytes) -> bool:
    return len(prefix) >= 2 and prefix[0] == 0x1F and prefix[1] == 0x8B

def verify_public_download_sync(url: str, timeout: int = 30) -> bool:
    """
    Synchronously download `url` (no auth) and verify:
      - HTTP 200
      - starts with gzip magic
      - opens as a tar.gz and contains exactly `default.snapshot`
    This simulates what Gaia's `curl` + `tar -xzOf` will see.
    """
    try:
        resp = requests.get(url, timeout=timeout, allow_redirects=True)
        if resp.status_code != 200:
            logger.warning(f"Public GET {url} returned status {resp.status_code}")
            return False

        # Quick size check
        if len(resp.content) < 1024:
            logger.warning(f"Public GET {url} returned very small file ({len(resp.content)} bytes)")
            return False

        # Check gzip magic bytes
        if not _is_gzip_bytes(resp.content[:2]):
            logger.warning("Downloaded content is not gzip (magic mismatch)")
            return False

        # Try to open as tar.gz and validate contents
        bio = io.BytesIO(resp.content)
        try:
            with tarfile.open(fileobj=bio, mode="r:gz") as tar:
                names = tar.getnames()
                # Gaia expects one top-level member named 'default.snapshot'
                if names != ["default.snapshot"]:
                    logger.warning(f"Downloaded tar does not contain default.snapshot: {names}")
                    return False
        except Exception as e:
            logger.warning(f"Failed to read downloaded file as tar.gz: {e}")
            return False

        return True
    except Exception as e:
        logger.warning(f"verify_public_download_sync error: {e}")
        return False

async def compress_snapshot(snapshot_file: Path) -> Path:
    """
    Create a GaiaNet-compatible tar.gz containing exactly one entry named 'default.snapshot'.
    Returns the path to SNAPSHOT_DIR/default.snapshot.tar.gz
    """
    try:
        if not snapshot_file.exists() or snapshot_file.stat().st_size == 0:
            raise Exception("Snapshot file is empty or doesn't exist")

        # Use a deterministic output name so downstream URLs are predictable
        compressed_file = SNAPSHOT_DIR / "default.snapshot.tar.gz"
        temp_snapshot = SNAPSHOT_DIR / "default.snapshot"

        # Make sure snapshots dir exists
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

        # Copy the raw snapshot to default.snapshot (preserve metadata)
        shutil.copy2(snapshot_file, temp_snapshot)

        # Create the tar.gz containing exactly default.snapshot
        # Use GNU_FORMAT to maximize compatibility
        with tarfile.open(compressed_file, "w:gz", format=tarfile.GNU_FORMAT) as tar:
            tar.add(temp_snapshot, arcname="default.snapshot")

        # Verification: open it back and check it contains exactly default.snapshot and is non-empty
        with tarfile.open(compressed_file, "r:gz") as tar:
            names = tar.getnames()
            if names != ["default.snapshot"]:
                raise Exception(f"Archive contents invalid: {names}")
            f = tar.extractfile("default.snapshot")
            if f is None or f.read(1) == b"":
                raise Exception("default.snapshot inside tar.gz is empty")

        logger.info(f"Successfully created GaiaNet-compatible tar.gz: {compressed_file}")
        return compressed_file

    except Exception as e:
        logger.error(f"Error creating CLI-compatible archive: {e}")
        raise

async def upload_to_huggingface(compressed_snapshot: Path, snapshot_name: str) -> str:
    """
    Uploads compressed_snapshot to the fixed dataset `thenocode/gaia-console`.
    Creates a timestamped folder inside `snapshots/` to avoid collisions.
    Returns the public URL.
    """
    try:
        if not HF_TOKEN:
            return f"File: {compressed_snapshot} (upload manually - no HF_TOKEN)"

        api = HfApi(token=HF_TOKEN)

        # Always use this dataset
        repo_id = "thenocode/gaia-console"
        snapshot_filename = compressed_snapshot.name

        # Create timestamped folder inside snapshots/
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
        path_in_repo = f"snapshots/{ts}/{snapshot_filename}"

        logger.info(f"Uploading {compressed_snapshot} to {repo_id}:{path_in_repo}")

        api.upload_file(
            path_or_fileobj=str(compressed_snapshot),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
        )

        # Public URL Gaia can curl
        public_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{path_in_repo}"

        logger.info(f"✅ Uploaded snapshot. Public URL: {public_url}")
        return public_url

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return f"File: {compressed_snapshot} (upload failed: {str(e)})"
          
async def verify_tar_file(tar_file: Path) -> bool:
    """Verify the tar.gz file is valid using system tar command"""
    try:
        cmd = ["tar", "tzvf", str(tar_file)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            logger.info(f"Tar file contents: {result.stdout.strip()}")
            return True
        else:
            logger.warning(f"Tar verification failed: {result.stderr}")
            return False
    except Exception as e:
        logger.warning(f"Tar verification error: {e}")
        return False

async def compress_with_python_fallback(snapshot_file: Path) -> Path:
    """Fallback compression using Python tarfile with specific format"""
    try:
        logger.warning("Using Python tarfile fallback compression")
        
        compressed_file = SNAPSHOT_DIR / f"{snapshot_file.name}.tar.gz"
        
        # Use specific format options to match GNU tar
        with tarfile.open(compressed_file, "w:gz", format=tarfile.GNU_FORMAT) as tar:
            # Add file with preserve permissions
            tar.add(snapshot_file, arcname=snapshot_file.name, recursive=False)
        
        logger.info(f"Python fallback compression: {compressed_file}")
        return compressed_file
        
    except Exception as e:
        logger.error(f"Python fallback compression also failed: {e}")
        # If all else fails, return the original uncompressed file
        return snapshot_file

async def upload_without_snapshots_folder(compressed_snapshot: Path, snapshot_name: str) -> str:
    """Fallback upload without the snapshots folder"""
    try:
        api = HfApi(token=HF_TOKEN)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_filename = f"snapshot_{timestamp}.tar.gz"
        
        # Try uploading to root instead of snapshots folder
        api.upload_file(
            path_or_fileobj=str(compressed_snapshot),
            path_in_repo=snapshot_filename,
            repo_id=HF_DATASET_NAME,
            repo_type="dataset",
        )
        
        return f"https://huggingface.co/datasets/{HF_DATASET_NAME}/resolve/main/{snapshot_filename}"
        
    except Exception as e:
        raise Exception(f"Root upload failed: {str(e)}")

async def upload_with_force(compressed_snapshot: Path, snapshot_name: str) -> str:
    """Alternative upload method for existing repositories"""
    try:
        api = HfApi(token=HF_TOKEN)
        
        # Try to upload to the original repository with different approach
        api.upload_file(
            path_or_fileobj=str(compressed_snapshot),
            path_in_repo=f"snapshots/{snapshot_name}.tar.gz",
            repo_id=HF_DATASET_NAME,
            repo_type="dataset",
            # Add additional parameters that might help
        )
        
        return f"https://huggingface.co/datasets/{HF_DATASET_NAME}/raw/main/snapshots/{snapshot_name}.tar.gz"
        
    except Exception as e:
        # If that fails, try the folder approach
        return await upload_as_folder(compressed_snapshot, snapshot_name)

async def upload_as_folder(compressed_snapshot: Path, snapshot_name: str) -> str:
    """Upload using the folder method which might have different permissions"""
    try:
        api = HfApi(token=HF_TOKEN)
        
        # Create temporary directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a folder for the upload
            upload_folder = temp_path / "snapshot_upload"
            upload_folder.mkdir()
            
            # Copy the snapshot
            shutil.copy2(compressed_snapshot, upload_folder / "snapshot.tar.gz")
            
            # Upload the entire folder
            api.upload_folder(
                folder_path=str(upload_folder),
                repo_id=HF_DATASET_NAME,
                repo_type="dataset",
                path_in_repo=f"snapshots/{snapshot_name}",
            )
        
        return f"https://huggingface.co/datasets/{HF_DATASET_NAME}/tree/main/snapshots/{snapshot_name}"
        
    except Exception as e:
        raise Exception(f"Folder upload failed: {str(e)}")

async def upload_to_huggingface_alternative(compressed_snapshot: Path, snapshot_name: str) -> str:
    """Alternative upload method using different API approach"""
    try:
        # Use a different method - upload folder instead of single file
        api = HfApi(token=HF_TOKEN)
        
        # Create a temporary directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # Create snapshot directory
            snapshot_dir = temp_path / "snapshot"
            snapshot_dir.mkdir()
            
            # Copy our compressed snapshot
            shutil.copy2(compressed_snapshot, snapshot_dir / "snapshot.tar.gz")
            
            # Create a README for the dataset
            readme_content = f"""
            # Snapshot Dataset: {snapshot_name}
            
            This is an automatically generated snapshot from the Document Snapshot Generator.
            
            Created: {datetime.now().isoformat()}
            """
            
            with open(snapshot_dir / "README.md", "w") as f:
                f.write(readme_content)
            
            # Upload the entire folder
            api.upload_folder(
                folder_path=str(snapshot_dir),
                repo_id="thenocode/gaia-console",
                repo_type="dataset",
            )
            
            logger.info(f"Successfully uploaded via alternative method")
            return f"https://huggingface.co/datasets/{snapshot_name}"
            
    except Exception as e:
        logger.error(f"Alternative upload method also failed: {e}")
        # As a last resort, provide instructions for manual upload
        return f"https://huggingface.co/datasets/{snapshot_name} (manual upload required - file: {compressed_snapshot})"

async def cleanup_all_files(session_dir: Path, compressed_snapshot: Path = None, snapshot_file: Path = None):
    """Clean up all temporary files after successful upload"""
    try:
        # Clean up session directory (uploaded files)
        if session_dir and session_dir.exists():
            shutil.rmtree(session_dir, ignore_errors=True)
            logger.info(f"Cleaned up session directory: {session_dir}")
        
        # Clean up compressed snapshot
        if compressed_snapshot and compressed_snapshot.exists():
            try:
                compressed_snapshot.unlink()
                logger.info(f"Cleaned up compressed snapshot: {compressed_snapshot}")
            except:
                pass
        
        # Clean up uncompressed snapshot
        if snapshot_file and snapshot_file.exists():
            try:
                snapshot_file.unlink()
                logger.info(f"Cleaned up snapshot file: {snapshot_file}")
            except:
                pass
        
        # Clean up any embedding files that might have been created
        try:
            embedding_files = list(EMBEDDING_DIR.glob("*.json"))
            for embedding_file in embedding_files:
                if embedding_file.exists():
                    embedding_file.unlink()
                    logger.info(f"Cleaned up embedding file: {embedding_file}")
        except:
            pass
        
        logger.info("All temporary files cleaned up successfully")
        
    except Exception as e:
        logger.error(f"Error during file cleanup: {e}")
        # Don't raise exception during cleanup - it's not critical

async def cleanup_qdrant_collection():
    """Clean up the Qdrant collection after successful processing"""
    try:
        if qdrant_client and COLLECTION_NAME != "default":
            # Only delete if it's not the default collection and we're not using cloud Qdrant
            if not QDRANT_API_KEY:  # Only cleanup if not using cloud (for cost reasons)
                try:
                    qdrant_client.delete_collection(COLLECTION_NAME)
                    logger.info(f"Deleted Qdrant collection '{COLLECTION_NAME}'")
                except Exception as e:
                    logger.warning(f"Could not delete collection '{COLLECTION_NAME}': {e}")
            else:
                logger.info(f"Keeping collection '{COLLECTION_NAME}' on cloud Qdrant")
    except Exception as e:
        logger.warning(f"Error during Qdrant collection cleanup: {e}")

@app.get("/check-deployment/{droplet_id}")
async def check_deployment(droplet_id: int):
    """Manually check deployment status and return Gaia URL if available"""
    if droplet_id not in DEPLOYMENTS:
        raise HTTPException(status_code=404, detail="Droplet not found")
    
    deployment = DEPLOYMENTS[droplet_id]
    ip = deployment.get("ip")
    
    if not ip or ip == "pending":
        return {"status": "pending", "message": "Droplet IP not available yet"}
    
    try:
        gaia_url = fetch_gaia_url(ip)
        if gaia_url:
            deployment["gaia_url"] = gaia_url
            return {
                "status": "ready", 
                "gaia_url": gaia_url,
                "message": "Gaia node is ready"
            }
        else:
            return {
                "status": "installing", 
                "message": "Gaia node is still installing"
            }
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Error checking deployment: {str(e)}"
        }

@app.get("/process-stream")
async def process_stream(request: Request):
    """Server-sent events endpoint for progress updates"""
    session_id = request.query_params.get("session_id")
    
    async def event_generator():
        if not session_id:
            yield {"event": "error", "data": "No session ID provided"}
            return
            
        # Send initial connection message
        yield {
            "event": "message",
            "data": json.dumps({
                "percent": 0,
                "message": "Starting processing...",
                "step": "Initializing"
            })
        }
        
        # Wait for progress updates
        last_percent = 0
        while True:
            if session_id in progress_events:
                event = progress_events[session_id]
                if event.get("percent", 0) > last_percent:
                    yield {
                        "event": "message",
                        "data": json.dumps(event)
                    }
                    last_percent = event.get("percent", 0)
                    
                    if last_percent >= 100:
                        break
            
            await asyncio.sleep(0.5)
            
            # Check if client is still connected
            if await request.is_disconnected():
                break
    
    return EventSourceResponse(event_generator())

# -------------------------
# DigitalOcean Integration & Streaming
# -------------------------

def stream_installation_logs(ip: str, droplet_id: int):
    """Stream real-time installation logs from the droplet"""
    try:
        if SSH_PASSPHRASE:
            key = paramiko.RSAKey.from_private_key_file(SSH_PRIVATE_KEY, password=SSH_PASSPHRASE)
        else:
            key = paramiko.RSAKey.from_private_key_file(SSH_PRIVATE_KEY)

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(ip, username=SSH_USER, pkey=key, timeout=15)

        # Stream the detailed installation log
        stdin, stdout, stderr = ssh.exec_command("tail -f /var/log/gaianet-detailed-install.log")
        
        # Read logs in real-time
        while True:
            line = stdout.readline()
            if line:
                push_log(droplet_id, f"INSTALL_LOG: {line.strip()}")
            else:
                break
                
        ssh.close()
        
    except Exception as e:
        push_log(droplet_id, f"Log streaming error: {e}")

def get_installation_progress(ip: str) -> str:
    """Get the current installation progress"""
    try:
        if SSH_PASSPHRASE:
            key = paramiko.RSAKey.from_private_key_file(SSH_PRIVATE_KEY, password=SSH_PASSPHRASE)
        else:
            key = paramiko.RSAKey.from_private_key_file(SSH_PRIVATE_KEY)

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(ip, username=SSH_USER, pkey=key, timeout=10)

        # Check if installation is complete
        stdin, stdout, stderr = ssh.exec_command("grep -q 'INSTALLATION_COMPLETE' /var/log/gaianet-detailed-install.log && echo 'COMPLETE' || echo 'IN_PROGRESS'")
        status = stdout.read().decode().strip()
        
        # Get the last few lines of the log
        stdin, stdout, stderr = ssh.exec_command("tail -10 /var/log/gaianet-detailed-install.log")
        recent_logs = stdout.read().decode().strip()
        
        ssh.close()
        
        return f"{status}: {recent_logs}"
        
    except Exception as e:
        return f"ERROR: {e}"

def push_log(droplet_id: int, message: str):
    q = LOG_STREAMS.get(droplet_id)
    payload = json.dumps({
        "ts": int(time.time()),
        "message": message
    })
    if q:
        try:
            q.put_nowait(payload)
        except Exception:
            pass
    logger.info(f"[{droplet_id}] {message}")

def append_log(droplet_id: int, message: str):
    logger.info(f"[{droplet_id}] {message}")
    LOGS.setdefault(droplet_id, []).append(message)


def create_droplet(snapshot_url: str, user_id: str):
    if not DO_TOKEN or not SSH_KEY_ID:
        raise HTTPException(status_code=500, detail="DigitalOcean credentials missing")

    # Sanitize user_id for hostname (remove underscores and other invalid chars)
    sanitized_user_id = re.sub(r'[^a-zA-Z0-9-]', '', user_id.replace('_', '-'))

    cloud_init = f"""#cloud-config
write_files:
  - path: /root/install-gaianet.sh
    content: |
      #!/bin/bash
      set -e
      set -x  # Enable debug mode to show all commands

      # Create detailed log file
      exec > >(tee -a /var/log/gaianet-detailed-install.log) 2>&1

      echo "=== Gaia Installation Started at $(date) ==="
      echo "Snapshot URL: {snapshot_url}"

      # Function to log steps with timestamps
      log_step() {{
          echo "[$(date '+%Y-%m-%d %H:%M:%S')] STEP: $1"
      }}

      # Update package list
      log_step "Updating package list..."
      apt-get update

      # Install required packages
      log_step "Installing required packages..."
      apt-get install -y curl wget tar gzip

      # Install Gaia
      log_step "Downloading Gaia installer..."
      curl -sSfL 'https://github.com/GaiaNet-AI/gaianet-node/releases/latest/download/install.sh' -o /tmp/install.sh
      chmod +x /tmp/install.sh

      log_step "Running Gaia installer..."
      /tmp/install.sh

      # Wait for installation to complete
      log_step "Waiting for installation to complete..."
      sleep 10

      # Add gaianet to PATH
      echo "Adding gaianet to PATH..."
      echo 'export PATH="$PATH:/gaianet/bin"' >> /.bashrc
      source /.bashrc

      # Verify installation
      log_step "Verifying installation..."
      which gaianet && echo "gaianet found at: $(which gaianet)" || echo "gaianet not found in PATH"
      ls -la /gaianet/ && echo "GaiaNet directory contents listed"
      
      # Configure GaiaNet
      log_step "Configuring Gaia snapshot..."
      gaianet config --snapshot {snapshot_url}
    
      log_step "Reducing the chat context size..."
      gaianet config --chat-ctx-size 4096

      log_step "Configuring embedding URL..."
      gaianet config --embedding-url https://huggingface.co/gaianet/gte-Qwen2-1.5B-instruct-GGUF/resolve/main/gte-Qwen2-1.5B-instruct-f16.gguf

      log_step "Configuring embedding context size..."
      gaianet config --embedding-ctx-size 8192

      # Initialize Gaia
      log_step "Initializing GaiaNet..."
      gaianet init

      # Start Gaia and capture output
      log_step "Starting Gaia..."
      echo "=== GAIA_NODE_START_OUTPUT ==="
      gaianet start
      echo "=== GAIA_NODE_START_COMPLETED ==="

      log_step "Checking Gaia status..."
      gaianet status || echo "Status check failed"

      log_step "Checking running processes..."
      ps aux | grep -i gaia || echo "No Gaia processes found"

      echo "=== Gaia Installation Completed at $(date) ==="
      echo "INSTALLATION_COMPLETE"
    permissions: '0755'

  - path: /root/stream-logs.sh
    content: |
      #!/bin/bash
      # Stream installation logs in real-time
      tail -f /var/log/gaianet-detailed-install.log
    permissions: '0755'

runcmd:
  - /root/install-gaianet.sh
  - nohup /root/stream-logs.sh > /var/log/streaming.log 2>&1 &
"""

    resp = requests.post(
        "https://api.digitalocean.com/v2/droplets",
        headers={"Authorization": f"Bearer {DO_TOKEN}"},
        json={
            "name": f"gaia-node-{sanitized_user_id}-{uuid.uuid4().hex[:6]}",
            "region": DO_REGION,
            "size": DO_SIZE,
            "image": "ubuntu-22-04-x64",
            "ssh_keys": [SSH_KEY_ID],
            "user_data": cloud_init,
        }
    )

    if resp.status_code != 202:
        logger.error(resp.text)
        raise HTTPException(status_code=500, detail="Failed to create droplet")

    droplet = resp.json()["droplet"]
    push_log(droplet["id"], "Droplet created; waiting for it to become active…")
    return droplet


def get_droplet_info(droplet_id: int):
    resp = requests.get(
        f"https://api.digitalocean.com/v2/droplets/{droplet_id}",
        headers={"Authorization": f"Bearer {DO_TOKEN}"}
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch droplet status")

    droplet = resp.json()["droplet"]
    ip = None
    for net in droplet["networks"]["v4"]:
        if net["type"] == "public":
            ip = net["ip_address"]

    return droplet["status"], ip, droplet["created_at"]

def get_detailed_droplet_status(ip: str) -> dict:
    """Get detailed status information from the droplet for debugging"""
    try:
        if SSH_PASSPHRASE:
            key = paramiko.RSAKey.from_private_key_file(SSH_PRIVATE_KEY, password=SSH_PASSPHRASE)
        else:
            key = paramiko.RSAKey.from_private_key_file(SSH_PRIVATE_KEY)
    except Exception as e:
        return {"error": f"SSH key error: {e}"}

    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(ip, username=SSH_USER, pkey=key, timeout=20)

        status_info = {}

        # Check system info
        stdin, stdout, stderr = ssh.exec_command("uname -a")
        status_info["system_info"] = stdout.read().decode().strip()

        # Check if gaianet is installed
        stdin, stdout, stderr = ssh.exec_command("which gaianet || echo 'NOT_FOUND'")
        status_info["gaianet_installed"] = stdout.read().decode().strip() != "NOT_FOUND"

        # Check service status
        stdin, stdout, stderr = ssh.exec_command("systemctl is-active gaianet 2>/dev/null || echo 'inactive'")
        status_info["service_status"] = stdout.read().decode().strip()

        # Get recent logs
        stdin, stdout, stderr = ssh.exec_command("journalctl -u gaianet --no-pager -n 10 2>/dev/null || echo 'NO_LOGS'")
        status_info["recent_logs"] = stdout.read().decode().strip()

        # Check cloud-init status
        stdin, stdout, stderr = ssh.exec_command("cloud-init status 2>/dev/null || echo 'NO_CLOUD_INIT'")
        status_info["cloud_init_status"] = stdout.read().decode().strip()

        # Check cloud-init logs
        stdin, stdout, stderr = ssh.exec_command("tail -n 10 /var/log/cloud-init.log 2>/dev/null || echo 'NO_CLOUD_INIT_LOGS'")
        status_info["cloud_init_logs"] = stdout.read().decode().strip()

        # Check installation logs
        stdin, stdout, stderr = ssh.exec_command("cat /var/log/gaianet-install.log 2>/dev/null || echo 'NO_INSTALL_LOGS'")
        status_info["installation_logs"] = stdout.read().decode().strip()

        # Check if installation script exists
        stdin, stdout, stderr = ssh.exec_command("ls -la /root/install-gaianet.sh 2>/dev/null || echo 'NO_INSTALL_SCRIPT'")
        status_info["install_script"] = stdout.read().decode().strip()

        # Check for any running processes
        stdin, stdout, stderr = ssh.exec_command("ps aux | grep gaianet | grep -v grep || echo 'NO_PROCESSES'")
        status_info["running_processes"] = stdout.read().decode().strip()

        # Check if gaianet directory exists
        stdin, stdout, stderr = ssh.exec_command("ls -la /gaianet/ 2>/dev/null || echo 'GAIANET_DIR_NOT_FOUND'")
        status_info["gaianet_directory"] = stdout.read().decode().strip()

        # Check PATH
        stdin, stdout, stderr = ssh.exec_command("echo $PATH")
        status_info["path"] = stdout.read().decode().strip()

        # Check if gaianet command is available
        stdin, stdout, stderr = ssh.exec_command("which gaianet 2>/dev/null || echo 'NOT_IN_PATH'")
        status_info["gaianet_in_path"] = stdout.read().decode().strip()

        ssh.close()
        return status_info

    except Exception as e:
        return {"error": f"Connection failed: {e}"}

def fetch_gaia_url(ip: str) -> str:
    try:
        if SSH_PASSPHRASE:
            key = paramiko.RSAKey.from_private_key_file(SSH_PRIVATE_KEY, password=SSH_PASSPHRASE)
        else:
            key = paramiko.RSAKey.from_private_key_file(SSH_PRIVATE_KEY)

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(ip, username=SSH_USER, pkey=key, timeout=15)

        # Check multiple sources for the Gaia URL
        sources = [
            # Check gaianet status output
            "gaianet status 2>/dev/null | grep -i 'https://'",
            # Check journalctl for gaianet service
            "journalctl -u gaianet --no-pager | grep -i 'https://' | tail -n 5",
            # Check the detailed installation log
            "grep -i 'https://.*gaia.domains' /var/log/gaianet-detailed-install.log",
            # Check if gaianet is running and get info
            "ps aux | grep gaianet | grep -v grep",
            # Check service status
            "systemctl status gaianet 2>/dev/null | grep -A 10 -B 10 'https://'"
        ]

        for i, cmd in enumerate(sources):
            stdin, stdout, stderr = ssh.exec_command(cmd)
            output = stdout.read().decode().strip()
            
            if output:
                logger.info(f"Source {i+1} output: {output[:200]}...")
                
                # Look for Gaia URL patterns
                patterns = [
                    r"https://[0-9a-fA-F]+\.gaia\.domains",
                    r"https://\w+\.gaia\.domains",
                    r"https://[^ ]+\.gaia\.domains",
                    r"gaia\.domains",
                    r"https://[0-9a-fA-F]{40}\.gaia\.domains"  # Specific pattern for your URL
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, output)
                    if matches:
                        gaia_url = matches[0]
                        if not gaia_url.startswith('https://'):
                            gaia_url = f"https://{gaia_url}"
                        logger.info(f"Found Gaia URL: {gaia_url}")
                        ssh.close()
                        return gaia_url

        ssh.close()
        raise RuntimeError("Gaia URL not found in any logs or outputs")
        
    except Exception as e:
        raise RuntimeError(f"SSH error: {e}")

def destroy_droplet(droplet_id: int):
    resp = requests.delete(
        f"https://api.digitalocean.com/v2/droplets/{droplet_id}",
        headers={"Authorization": f"Bearer {DO_TOKEN}"}
    )
    if resp.status_code not in (202, 204):
        raise HTTPException(status_code=500, detail="Failed to destroy droplet")
    return {"status": "deleted"}

async def poll_until_ready(droplet_id: int):
    push_log(droplet_id, "Starting enhanced readiness checks with real-time logging...")
    gaia_url = None
    ip = None
    log_stream_task = None

    # Wait for droplet to become active
    for attempt in range(20):
        status, ip, created_at = get_droplet_info(droplet_id)
        DEPLOYMENTS[droplet_id] = {"status": status, "ip": ip, "gaia_url": gaia_url, "created_at": created_at}
        
        if status == "active" and ip:
            push_log(droplet_id, f"✅ Droplet active with IP: {ip}")
            break
            
        push_log(droplet_id, f"⏳ Waiting for droplet to become active... Status: {status}, IP: {ip or 'pending'} (attempt {attempt+1}/20)")
        await asyncio.sleep(10)  # Reduced to 10s

    if not ip or status != "active":
        push_log(droplet_id, "❌ Droplet failed to become active in time")
        return

    # Start log streaming in background
    try:
        # Run log streaming in a separate thread
        import threading
        log_thread = threading.Thread(target=stream_installation_logs, args=(ip, droplet_id), daemon=True)
        log_thread.start()
        push_log(droplet_id, "📋 Started real-time log streaming...")
    except Exception as e:
        push_log(droplet_id, f"⚠️ Could not start log streaming: {e}")

    # Now monitor installation progress
    for attempt in range(60):  # 60 attempts * 10s = 10 minutes max
        try:
            # Check installation progress
            progress = get_installation_progress(ip)
            push_log(droplet_id, f"📊 Installation status: {progress}")
            
            if "COMPLETE" in progress:
                push_log(droplet_id, "✅ Installation completed, looking for Gaia URL...")
                # Try to get Gaia URL
                try:
                    gaia_url = fetch_gaia_url(ip)
                    if gaia_url:
                        DEPLOYMENTS[droplet_id]["gaia_url"] = gaia_url
                        push_log(droplet_id, f"🎉 Gaia node is ready: {gaia_url}")
                        return
                except Exception as e:
                    push_log(droplet_id, f"⚠️ Installation complete but URL not found: {e}")
                    # Even if URL not found, installation is complete
                    DEPLOYMENTS[droplet_id]["status"] = "installation_complete"
                    push_log(droplet_id, "✅ Installation completed successfully (URL detection failed)")
                    return
                    
        except Exception as e:
            push_log(droplet_id, f"⚠️ Progress check error: {e}")

        # Check if droplet is still active
        if attempt % 5 == 0:  # Every 50 seconds
            status, current_ip, created_at = get_droplet_info(droplet_id)
            if current_ip != ip or status != "active":
                push_log(droplet_id, f"⚠️ Droplet status changed: {status}, IP: {current_ip}")
                ip = current_ip
                DEPLOYMENTS[droplet_id]["ip"] = ip
                DEPLOYMENTS[droplet_id]["status"] = status
        
        await asyncio.sleep(10)  # Check every 10 seconds

    push_log(droplet_id, "⏰ Timed out waiting for installation. Check logs above for details.")

@app.get("/view-logs/{droplet_id}")
async def view_logs(droplet_id: int):
    """View current installation logs"""
    if droplet_id not in DEPLOYMENTS:
        raise HTTPException(status_code=404, detail="Droplet not found")
    
    deployment = DEPLOYMENTS[droplet_id]
    ip = deployment.get("ip")
    
    if not ip or ip == "pending":
        return {"status": "pending", "message": "Droplet IP not available yet"}
    
    try:
        if SSH_PASSPHRASE:
            key = paramiko.RSAKey.from_private_key_file(SSH_PRIVATE_KEY, password=SSH_PASSPHRASE)
        else:
            key = paramiko.RSAKey.from_private_key_file(SSH_PRIVATE_KEY)

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(ip, username=SSH_USER, pkey=key, timeout=15)

        # Get the full installation log
        stdin, stdout, stderr = ssh.exec_command("cat /var/log/gaianet-detailed-install.log")
        full_log = stdout.read().decode().strip()
        error_log = stderr.read().decode().strip()
        
        ssh.close()

        return {
            "status": "success",
            "log": full_log,
            "error": error_log
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to retrieve logs: {e}"
        }

@app.get("/events/{droplet_id}")
async def sse_events(droplet_id: int, request: Request):
    # Create a queue if not exists
    if droplet_id not in LOG_STREAMS:
        LOG_STREAMS[droplet_id] = asyncio.Queue()
    q = LOG_STREAMS[droplet_id]

    async def event_generator():
        # If we already have a snapshot of state, emit it
        initial = DEPLOYMENTS.get(droplet_id)
        if initial:
            yield json.dumps({"ts": int(time.time()), "snapshot": initial})
        try:
            while True:
                # Client disconnected?
                if await request.is_disconnected():
                    break
                msg = await q.get()
                yield msg
        except asyncio.CancelledError:
            pass

    return EventSourceResponse(event_generator())


@app.websocket("/ws/{droplet_id}")
async def websocket_logs(websocket: WebSocket, droplet_id: int):
    await websocket.accept()
    if droplet_id not in LOG_STREAMS:
        LOG_STREAMS[droplet_id] = asyncio.Queue()
    q = LOG_STREAMS[droplet_id]

    # send initial state
    initial = DEPLOYMENTS.get(droplet_id)
    if initial:
        await websocket.send_text(json.dumps({"ts": int(time.time()), "snapshot": initial}))

    try:
        while True:
            msg = await q.get()
            await websocket.send_text(msg)
    except WebSocketDisconnect:
        pass

@app.post("/deploy")
def deploy_node(snapshot_url: str, user_id: str, background_tasks: BackgroundTasks):
    droplet = create_droplet(snapshot_url, user_id)
    droplet_id = droplet["id"]
    DEPLOYMENTS[droplet_id] = {"status": droplet["status"], "ip": None, "gaia_url": None, "created_at": droplet["created_at"]}
    # create a log stream queue
    LOG_STREAMS[droplet_id] = asyncio.Queue()
    push_log(droplet_id, "Deployment requested; provisioning VM and bootstrapping Gaia…")
    background_tasks.add_task(poll_until_ready, droplet_id)
    return {"droplet_id": droplet_id, "status": "deploying"}


@app.get("/status/{droplet_id}")
def node_status(droplet_id: int):
    if droplet_id not in DEPLOYMENTS:
        raise HTTPException(status_code=404, detail="Droplet not found")
    return DEPLOYMENTS[droplet_id]


@app.get("/droplet-status/{droplet_id}")
def get_droplet_detailed_status(droplet_id: int):
    """Get detailed status information from the droplet"""
    if droplet_id not in DEPLOYMENTS:
        raise HTTPException(status_code=404, detail="Droplet not found")

    deployment = DEPLOYMENTS[droplet_id]
    ip = deployment.get("ip")

    if not ip or ip == "pending":
        return {"status": "pending", "message": "Droplet IP not available yet"}

    try:
        detailed_status = get_detailed_droplet_status(ip)
        return {
            "droplet_id": droplet_id,
            "ip": ip,
            "deployment_status": deployment,
            "detailed_status": detailed_status
        }
    except Exception as e:
        return {
            "droplet_id": droplet_id,
            "ip": ip,
            "error": f"Failed to get detailed status: {e}"
        }


@app.delete("/destroy/{droplet_id}")
def delete_node(droplet_id: int):
    res = destroy_droplet(droplet_id)
    DEPLOYMENTS.pop(droplet_id, None)
    # signal end of stream
    q = LOG_STREAMS.pop(droplet_id, None)
    if q:
        try:
            q.put_nowait(json.dumps({"ts": int(time.time()), "message": "Deployment destroyed."}))
        except Exception:
            pass
    return res


@app.get("/logs/{droplet_id}")
async def stream_logs(request: Request, droplet_id: int):
    if droplet_id not in LOGS:
        raise HTTPException(status_code=404, detail="No logs for this droplet")

    async def event_generator():
        last_index = 0
        while True:
            if await request.is_disconnected():
                break

            logs = LOGS.get(droplet_id, [])
            if last_index < len(logs):
                for msg in logs[last_index:]:
                    yield {"event": "message", "data": msg}
                last_index = len(logs)

            await asyncio.sleep(2)

    return EventSourceResponse(event_generator())

 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)