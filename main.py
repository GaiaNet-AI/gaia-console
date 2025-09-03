import os
import uuid
import shutil
import asyncio
import aiofiles
import tarfile
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import subprocess
import logging
from typing import List
import json
from datetime import datetime
from huggingface_hub import HfApi
import time
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from sse_starlette.sse import EventSourceResponse

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Create directories
for directory in [UPLOAD_DIR, EMBEDDING_DIR, SNAPSHOT_DIR, WASM_DIR, MODEL_DIR]:
    directory.mkdir(exist_ok=True)

# Global Qdrant client
qdrant_client = None
wasmedge_available = False
progress_events = {}

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

# @app.get("/", response_class=HTMLResponse)
# async def read_root():
#     """Return the main HTML interface"""
#     html_content = """
#     <!DOCTYPE html>
#     <html lang="en">
#     <head>
#         <meta charset="UTF-8">
#         <meta name="viewport" content="width=device-width, initial-scale=1.0">
#         <title>Gaia Node Knowledge Base Snapshot Generator</title>
#         <style>
#             :root {
#                 --gaia-primary: #2563eb;
#                 --gaia-secondary: #1e40af;
#                 --gaia-accent: #3b82f6;
#                 --gaia-dark: #1e293b;
#                 --gaia-light: #f8fafc;
#                 --gaia-gradient: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
#             }
            
#             body {
#                 font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
#                 margin: 0;
#                 padding: 0;
#                 background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
#                 min-height: 100vh;
#             }
            
#             .container {
#                 max-width: 900px;
#                 margin: 2rem auto;
#                 background: white;
#                 padding: 2rem;
#                 border-radius: 16px;
#                 box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
#                 border: 1px solid #e2e8f0;
#             }
            
#             .header {
#                 text-align: center;
#                 margin-bottom: 2rem;
#             }
            
#             .logo {
#                 display: flex;
#                 align-items: center;
#                 justify-content: center;
#                 gap: 12px;
#                 margin-bottom: 1rem;
#             }
            
#             .logo-icon {
#                 width: 48px;
#                 height: 48px;
#                 background: var(--gaia-gradient);
#                 border-radius: 12px;
#                 display: flex;
#                 align-items: center;
#                 justify-content: center;
#                 color: white;
#                 font-weight: bold;
#                 font-size: 20px;
#             }
            
#             .logo-text {
#                 font-size: 28px;
#                 font-weight: 800;
#                 background: var(--gaia-gradient);
#                 -webkit-background-clip: text;
#                 -webkit-text-fill-color: transparent;
#                 background-clip: text;
#             }
            
#             h1 {
#                 color: var(--gaia-dark);
#                 font-weight: 700;
#                 font-size: 2rem;
#                 margin: 0.5rem 0;
#             }
            
#             .subtitle {
#                 color: #64748b;
#                 font-size: 1.1rem;
#                 margin-bottom: 2rem;
#             }
            
#             .drop-zone {
#                 border: 3px dashed #cbd5e1;
#                 border-radius: 12px;
#                 padding: 3rem 2rem;
#                 text-align: center;
#                 margin: 2rem 0;
#                 cursor: pointer;
#                 transition: all 0.3s ease;
#                 background: #f8fafc;
#             }
            
#             .drop-zone:hover, .drop-zone.dragover {
#                 border-color: var(--gaia-primary);
#                 background: #f1f5f9;
#                 transform: translateY(-2px);
#             }
            
#             .drop-zone p {
#                 font-size: 1.2rem;
#                 color: #64748b;
#                 margin: 0;
#             }
            
#             .drop-zone .icon {
#                 font-size: 3rem;
#                 color: var(--gaia-primary);
#                 margin-bottom: 1rem;
#             }
            
#             .file-list {
#                 margin: 2rem 0;
#                 border-radius: 12px;
#                 overflow: hidden;
#             }
            
#             .file-item {
#                 display: flex;
#                 justify-content: space-between;
#                 align-items: center;
#                 padding: 1rem 1.5rem;
#                 background: #f8fafc;
#                 border-bottom: 1px solid #e2e8f0;
#                 transition: background 0.2s ease;
#             }
            
#             .file-item:hover {
#                 background: #f1f5f9;
#             }
            
#             .file-item:last-child {
#                 border-bottom: none;
#             }
            
#             .file-info {
#                 display: flex;
#                 align-items: center;
#                 gap: 12px;
#             }
            
#             .file-icon {
#                 color: var(--gaia-primary);
#                 font-weight: bold;
#             }
            
#             .file-name {
#                 font-weight: 500;
#                 color: var(--gaia-dark);
#             }
            
#             .file-size {
#                 color: #64748b;
#                 font-size: 0.9rem;
#             }
            
#             .remove-btn {
#                 background: #ef4444;
#                 color: white;
#                 border: none;
#                 padding: 0.5rem 1rem;
#                 border-radius: 6px;
#                 cursor: pointer;
#                 font-size: 0.9rem;
#                 transition: background 0.2s ease;
#             }
            
#             .remove-btn:hover {
#                 background: #dc2626;
#             }
            
#             .process-btn {
#                 background: var(--gaia-gradient);
#                 color: white;
#                 border: none;
#                 padding: 1rem 2rem;
#                 border-radius: 10px;
#                 cursor: pointer;
#                 font-size: 1.1rem;
#                 font-weight: 600;
#                 display: block;
#                 margin: 2rem auto;
#                 transition: all 0.3s ease;
#                 width: auto;
#             }
            
#             .process-btn:hover:not(:disabled) {
#                 transform: translateY(-2px);
#                 box-shadow: 0 8px 20px rgba(37, 99, 235, 0.3);
#             }
            
#             .process-btn:disabled {
#                 background: #cbd5e1;
#                 cursor: not-allowed;
#                 transform: none;
#             }
            
#             .progress {
#                 margin: 2rem 0;
#                 background: #f8fafc;
#                 padding: 1.5rem;
#                 border-radius: 12px;
#             }
            
#             .progress-bar {
#                 height: 20px;
#                 background: #e2e8f0;
#                 border-radius: 10px;
#                 overflow: hidden;
#                 margin: 1rem 0;
#             }
            
#             .progress-fill {
#                 height: 100%;
#                 background: var(--gaia-gradient);
#                 width: 0%;
#                 transition: width 0.5s ease;
#                 border-radius: 10px;
#             }
            
#             .progress-text {
#                 font-weight: 600;
#                 color: var(--gaia-dark);
#                 margin-bottom: 0.5rem;
#             }
            
#             .step-details {
#                 margin-top: 1rem;
#                 font-size: 0.9rem;
#                 color: #64748b;
#                 max-height: 200px;
#                 overflow-y: auto;
#                 background: white;
#                 padding: 1rem;
#                 border-radius: 8px;
#                 border: 1px solid #e2e8f0;
#             }
            
#             .step-details div {
#                 padding: 0.3rem 0;
#                 border-bottom: 1px solid #f1f5f9;
#             }
            
#             .step-details div:last-child {
#                 border-bottom: none;
#             }
            
#             .result {
#                 margin-top: 2rem;
#                 padding: 2rem;
#                 border-radius: 12px;
#                 background: #f0f9ff;
#                 border: 1px solid #bae6fd;
#                 display: none;
#             }
            
#             .result h3 {
#                 color: var(--gaia-primary);
#                 margin: 0 0 1rem 0;
#             }
            
#             .snapshot-url {
#                 color: var(--gaia-primary);
#                 text-decoration: none;
#                 font-weight: 500;
#                 word-break: break-all;
#             }
            
#             .snapshot-url:hover {
#                 text-decoration: underline;
#             }
            
#             .error {
#                 color: #dc2626;
#                 background: #fef2f2;
#                 padding: 1rem;
#                 border-radius: 8px;
#                 border: 1px solid #fecaca;
#                 margin-top: 1rem;
#                 display: none;
#             }
            
#             .warning {
#                 color: #92400e;
#                 background: #fffbeb;
#                 border: 1px solid #fed7aa;
#                 padding: 1rem;
#                 border-radius: 8px;
#                 margin-bottom: 1.5rem;
#                 display: none;
#             }
            
#             .footer {
#                 text-align: center;
#                 margin-top: 3rem;
#                 color: #64748b;
#                 font-size: 0.9rem;
#             }
            
#             .footer a {
#                 color: var(--gaia-primary);
#                 text-decoration: none;
#             }
            
#             .footer a:hover {
#                 text-decoration: underline;
#             }
            
#             @keyframes pulse {
#                 0% { transform: scale(1); }
#                 50% { transform: scale(1.05); }
#                 100% { transform: scale(1); }
#             }
            
#             .processing {
#                 animation: pulse 2s infinite;
#             }
#             .logo-img {
#                 height: 60px;
#                 width: auto;
#                 object-fit: contain;
#                 margin-bottom: 0.5rem;
#             }
            
#             /* Responsive logo */
#             @media (max-width: 640px) {
#                 .logo-img {
#                     height: 48px;
#                 }
#             }
#         </style>
#         <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
#         <script src="https://cdn.tailwindcss.com"></script>
#         <script>
#             tailwind.config = {
#                 theme: {
#                     extend: {
#                         colors: {
#                             primary: {
#                                 50: '#eff6ff',
#                                 100: '#dbeafe',
#                                 500: '#3b82f6',
#                                 600: '#2563eb',
#                                 700: '#1d4ed8',
#                                 900: '#1e3a8a',
#                             }
#                         }
#                     }
#                 }
#             }
#         </script>
#     </head>
#     <body>
#         <div class="container">
#             <div class="header">
#                 <div style="text-align: center;">
#                     <img src="/static/logo.png" alt="Gaia Node Logo" class="logo-img" loading="lazy">
#                 </div>
#                 <h1>Snapshot Generator</h1>
#                 <p class="subtitle">Create knowledge snapshots for your Gaia Node deployment</p>
#             </div>
            
#             <div id="wasmWarning" class="warning" style="display: none;">
#                 <strong>⚠️ Performance Note:</strong> WasmEdge is not available. Using fallback embedding generation.
#             </div>
            
#             <p>Drag and drop your knowledge files here (TXT, MD, PDF, CSV) to create a snapshot for your Gaia Node. Max 10MB per file.</p>
            
#             <div class="drop-zone" id="dropZone">
#                 <div class="icon">📁</div>
#                 <p>Drag files here or click to browse</p>
#                 <input type="file" id="fileInput" multiple style="display: none;">
#             </div>
            
#             <div class="file-list" id="fileList"></div>
            
#             <button class="process-btn" id="processBtn" disabled>Generate Snapshot</button>
            
#             <div class="progress" id="progressSection" style="display: none;">
#                 <h3>Processing Progress</h3>
#                 <div class="progress-bar">
#                     <div class="progress-fill" id="progressFill"></div>
#                 </div>
#                 <p id="progressText">Starting snapshot generation...</p>
#                 <div class="step-details" id="stepDetails"></div>
#             </div>
            
#             <div class="result" id="resultSection">
#                 <h3>✅ Snapshot Created Successfully!</h3>
#                 <p>Your Gaia Node snapshot is ready for deployment:</p>
#                 <p><a id="snapshotUrl" target="_blank" class="snapshot-url"></a></p>
#                 <p><small>Use this URL to load the snapshot into your Gaia Node configuration.</small></p>
#             </div>
            
#             <div class="error" id="errorSection"></div>
            
#             <div class="footer">
#                 <p>Powered by <a href="https://gaianet.ai" target="_blank">Gaia Network</a> - Open Source AI Infrastructure</p>
#                 <p>Version 1.0.0 | <a href="https://github.com/gaia-network" target="_blank">GitHub</a> | <a href="https://docs.gaianet.ai" target="_blank">Documentation</a></p>
#             </div>
#         </div>


#         <script>
#             const dropZone = document.getElementById('dropZone');
#             const fileInput = document.getElementById('fileInput');
#             const fileList = document.getElementById('fileList');
#             const processBtn = document.getElementById('processBtn');
#             const progressSection = document.getElementById('progressSection');
#             const progressFill = document.getElementById('progressFill');
#             const progressText = document.getElementById('progressText');
#             const resultSection = document.getElementById('resultSection');
#             const snapshotUrl = document.getElementById('snapshotUrl');
#             const errorSection = document.getElementById('errorSection');
#             const wasmWarning = document.getElementById('wasmWarning');
#             const stepDetails = document.getElementById('stepDetails');
            
#             let files = [];
            
#             // Check if WasmEdge is available
#             fetch('/check-wasm')
#                 .then(response => response.json())
#                 .then(data => {
#                     if (!data.available) {
#                         wasmWarning.style.display = 'block';
#                     }
#                 });
            
#             // Drag and drop handlers
#             dropZone.addEventListener('click', () => fileInput.click());
            
#             dropZone.addEventListener('dragover', (e) => {
#                 e.preventDefault();
#                 dropZone.classList.add('dragover');
#             });
            
#             dropZone.addEventListener('dragleave', () => {
#                 dropZone.classList.remove('dragover');
#             });
            
#             dropZone.addEventListener('drop', (e) => {
#                 e.preventDefault();
#                 dropZone.classList.remove('dragover');
#                 handleFiles(e.dataTransfer.files);
#             });
            
#             fileInput.addEventListener('change', () => {
#                 handleFiles(fileInput.files);
#             });
            
#             function handleFiles(fileList) {
#                 for (let i = 0; i < fileList.length; i++) {
#                     const file = fileList[i];
                    
#                     // Check file size (10MB limit)
#                     if (file.size > 10 * 1024 * 1024) {
#                         showError(`File ${file.name} exceeds 10MB limit`);
#                         continue;
#                     }
                    
#                     // Check file type
#                     const ext = file.name.split('.').pop().toLowerCase();
#                     if (!['txt', 'md', 'pdf', 'csv'].includes(ext)) {
#                         showError(`File type ${ext} not supported. Please use TXT, MD, PDF, or CSV files.`);
#                         continue;
#                     }
                    
#                     // Add to files list
#                     if (!files.some(f => f.name === file.name && f.size === file.size)) {
#                         files.push(file);
#                         addFileToList(file);
#                     }
#                 }
                
#                 updateProcessButton();
#             }
            
#             function addFileToList(file) {
#                 const fileItem = document.createElement('div');
#                 fileItem.className = 'file-item';
#                 fileItem.innerHTML = `
#                     <div class="file-info">
#                         <span class="file-icon">📄</span>
#                         <span class="file-name">${file.name}</span>
#                         <span class="file-size">(${formatFileSize(file.size)})</span>
#                     </div>
#                     <button class="remove-btn" onclick="removeFile('${file.name}', ${file.size})">Remove</button>
#                 `;
#                 fileList.appendChild(fileItem);
#             }
            
#             function formatFileSize(bytes) {
#                 if (bytes < 1024) return bytes + ' B';
#                 else if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
#                 else return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
#             }
            
#             function removeFile(name, size) {
#                 files = files.filter(f => !(f.name === name && f.size === size));
                
#                 // Rebuild file list
#                 fileList.innerHTML = '';
#                 files.forEach(addFileToList);
                
#                 updateProcessButton();
#             }
            
#             function updateProcessButton() {
#                 processBtn.disabled = files.length === 0;
#                 processBtn.textContent = files.length > 0 ? 
#                     `Generate Snapshot (${files.length} file${files.length > 1 ? 's' : ''})` : 
#                     'Generate Snapshot';
#             }
            
#             function showError(message) {
#                 errorSection.textContent = message;
#                 errorSection.style.display = 'block';
#                 setTimeout(() => {
#                     errorSection.style.display = 'none';
#                 }, 5000);
#             }
            
#             // Process files
#             processBtn.addEventListener('click', async () => {
#                 progressSection.style.display = 'block';
#                 resultSection.style.display = 'none';
#                 errorSection.style.display = 'none';
#                 processBtn.disabled = true;
#                 processBtn.classList.add('processing');
                
#                 const session_id = 'session_' + Date.now();
                
#                 const formData = new FormData();
#                 files.forEach(file => formData.append('files', file));
#                 formData.append('session_id', session_id);
                
#                 try {
#                     const eventSource = new EventSource(`/process-stream?session_id=${session_id}`);
#                     eventSource.onmessage = function(event) {
#                         const data = JSON.parse(event.data);
#                         updateProgress(data.percent, data.message, data.step);
                        
#                         if (data.percent === 100) {
#                             eventSource.close();
#                             snapshotUrl.href = data.snapshot_url;
#                             snapshotUrl.textContent = data.snapshot_url;
#                             resultSection.style.display = 'block';
#                             processBtn.disabled = false;
#                             processBtn.classList.remove('processing');
#                         }
#                     };
                    
#                     eventSource.onerror = function(error) {
#                         console.error('EventSource failed:', error);
#                         eventSource.close();
#                         showError('Connection error during processing');
#                         processBtn.disabled = false;
#                         processBtn.classList.remove('processing');
#                     };
                    
#                     const response = await fetch('/process', {
#                         method: 'POST',
#                         body: formData
#                     });
                    
#                     if (!response.ok) {
#                         const error = await response.text();
#                         throw new Error(error);
#                     }
                    
#                 } catch (error) {
#                     showError('Error processing files: ' + error.message);
#                     processBtn.disabled = false;
#                     processBtn.classList.remove('processing');
#                 }
#             });

#             function updateProgress(percent, message, step) {
#                 progressFill.style.width = percent + '%';
#                 progressText.textContent = message;
                
#                 if (step) {
#                     stepDetails.innerHTML += `<div>${new Date().toLocaleTimeString()}: ${step}</div>`;
#                     stepDetails.scrollTop = stepDetails.scrollHeight;
#                 }
#             }
            
#             window.removeFile = removeFile;
#         </script>
#     </body>
#     </html>
#     """
#     return HTMLResponse(content=html_content)

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
                                // Update the config command with the actual URL
                                updateConfigCommand(data.snapshot_url);
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
                    gaianet config \\<br>
                    &nbsp;&nbsp;--snapshot <span class="text-yellow-300">${snapshotUrl}</span> \\<br>
                    &nbsp;&nbsp;--embedding-url https://huggingface.co/gaianet/gte-Qwen2-1.5B-instruct-GGUF/resolve/main/gte-Qwen2-1.5B-instruct-f16.gguf \\<br>
                    &nbsp;&nbsp;--embedding-ctx-size 8192
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
                const configCommand = `gaianet config \\
                --snapshot ${snapshotUrl} \\
                --embedding-url https://huggingface.co/gaianet/gte-Qwen2-1.5B-instruct-GGUF/resolve/main/gte-Qwen2-1.5B-instruct-f16.gguf \\
                --embedding-ctx-size 8192`;
                
                navigator.clipboard.writeText(configCommand).then(() => {
                    showToast('Config command copied to clipboard!');
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

            window.removeFile = removeFile;
            window.copySnapshotUrl = copySnapshotUrl;
            window.copyConfigCommand = copyConfigCommand;
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

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
        # Create a unique collection name based on files and timestamp
        global COLLECTION_NAME
        if files:
            first_filename = files[0].filename
            base_name = Path(first_filename).stem.lower().replace(' ', '_').replace('.', '_')[:20]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            COLLECTION_NAME = f"{base_name}_{timestamp}"
        else:
            COLLECTION_NAME = f"documents_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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
            "message": f"Processed {successful_files} files with {total_embeddings} embeddings"
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
    """Create a Qdrant snapshot"""
    try:
        # For cloud Qdrant, we need to handle snapshots differently
        # Cloud Qdrant may not allow direct filesystem access
        
        # Create a manual snapshot by exporting all data
        return await create_manual_snapshot(snapshot_name)
        
    except Exception as e:
        logger.error(f"Error creating Qdrant snapshot: {e}")
        raise HTTPException(status_code=500, detail=f"Snapshot creation failed: {str(e)}")

async def create_manual_snapshot(snapshot_name: str) -> Path:
    """Create a manual snapshot by exporting all data from Qdrant"""
    try:
        # Try to get points from Qdrant
        points = []
        try:
            points, offset = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                limit=10000,
                with_payload=True,
                with_vectors=True
            )
            logger.info(f"Retrieved {len(points)} points from Qdrant")
        except Exception as e:
            logger.warning(f"Could not retrieve points from Qdrant: {e}")
            # If we can't get points, create an empty snapshot with metadata
            points = []
        
        # Get collection info or use defaults
        try:
            collection_info = qdrant_client.get_collection(COLLECTION_NAME)
            vectors_count = collection_info.vectors_count
            points_count = collection_info.points_count
        except:
            vectors_count = len(points)
            points_count = len(points)
        
        # Create snapshot data
        snapshot_data = {
            "collection": COLLECTION_NAME,
            "vectors_count": vectors_count,
            "points_count": points_count,
            "points": [
                {
                    "id": point.id,
                    "vector": point.vector,
                    "payload": point.payload
                }
                for point in points
            ],
            "created_at": str(datetime.now()),
            "vector_size": VECTOR_SIZE,
            "distance_metric": "Cosine",
            "note": "Manual snapshot - some data may be in separate embedding files"
        }
        
        # Save snapshot
        snapshot_file = SNAPSHOT_DIR / f"{snapshot_name}.snapshot"
        async with aiofiles.open(snapshot_file, 'w') as f:
            await f.write(json.dumps(snapshot_data, indent=2, default=str))
        
        logger.info(f"Created manual snapshot: {snapshot_file} with {len(points)} points")
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

async def compress_snapshot(snapshot_file: Path) -> Path:
    """Compress the snapshot file"""
    try:
        compressed_file = SNAPSHOT_DIR / f"{snapshot_file.stem}.tar.gz"
        
        with tarfile.open(compressed_file, "w:gz") as tar:
            tar.add(snapshot_file, arcname=snapshot_file.name)
        
        logger.info(f"Compressed snapshot: {compressed_file}")
        return compressed_file
        
    except Exception as e:
        logger.error(f"Error compressing snapshot: {e}")
        raise HTTPException(status_code=500, detail=f"Snapshot compression failed: {str(e)}")

async def upload_to_huggingface(compressed_snapshot: Path, snapshot_name: str) -> str:
    """Upload snapshot to existing Hugging Face dataset in /snapshots folder"""
    try:
        if not HF_TOKEN:
            return f"File: {compressed_snapshot} (manual upload required - no HF_TOKEN)"
        
        if not HF_DATASET_NAME:
            return f"File: {compressed_snapshot} (manual upload required - no HF_DATASET_NAME)"
        
        api = HfApi(token=HF_TOKEN)
        
        # Verify the target dataset exists and we have access
        try:
            repo_info = api.repo_info(repo_id=HF_DATASET_NAME, repo_type="dataset")
            logger.info(f"Target dataset exists: {HF_DATASET_NAME}")
        except Exception as e:
            logger.error(f"Target dataset not found or inaccessible: {HF_DATASET_NAME} - {e}")
            return f"https://huggingface.co/datasets/{HF_DATASET_NAME} (dataset not found)"
        
        # Create a unique filename for the snapshot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_filename = f"snapshot_{timestamp}.tar.gz"
        
        # Upload directly to the snapshots folder in the existing dataset
        api.upload_file(
            path_or_fileobj=str(compressed_snapshot),
            path_in_repo=f"snapshots/{snapshot_filename}",
            repo_id=HF_DATASET_NAME,
            repo_type="dataset",
        )
        
        # Return the direct download URL
        return f"https://huggingface.co/datasets/{HF_DATASET_NAME}/resolve/main/snapshots/{snapshot_filename}"
        
    except Exception as e:
        logger.error(f"Upload to existing dataset failed: {e}")
        
        # Fallback: try without the snapshots folder
        try:
            return await upload_without_snapshots_folder(compressed_snapshot, snapshot_name)
        except Exception as fallback_error:
            logger.error(f"Fallback upload also failed: {fallback_error}")
            return f"https://huggingface.co/datasets/{HF_DATASET_NAME} (upload failed: {str(e)})"

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
        
        return f"https://huggingface.co/datasets/{HF_DATASET_NAME}/resolve/main/snapshots/{snapshot_name}.tar.gz"
        
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

async def cleanup_qdrant_collection():
    """Clean up the Qdrant collection after processing"""
    try:
        if not QDRANT_API_KEY:  # Only cleanup if not using cloud (for cost reasons)
            qdrant_client.delete_collection(COLLECTION_NAME)
            logger.info(f"Deleted collection '{COLLECTION_NAME}'")
        else:
            logger.info(f"Keeping collection '{COLLECTION_NAME}' on cloud Qdrant")
    except Exception as e:
        logger.warning(f"Could not delete collection '{COLLECTION_NAME}': {e}")

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
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)