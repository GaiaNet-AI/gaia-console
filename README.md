# Gaia Console - Knowledge Base Snapshot Generator

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A web-based platform for converting raw knowledge base files into deployable vector database snapshots for the Gaia Network. Upload your documents, get AI-powered embeddings, and deploy instantly to your Gaia Node.

<img width="1550" height="1574" alt="image" src="https://github.com/user-attachments/assets/be49c308-37e0-4f59-bb32-395c5f1c8845" />


## Features

- **Multi-Format Support**: Process TXT, Markdown, PDF, and CSV files
- **AI-Powered Embeddings**: Uses gte-Qwen2-1.5B model with 1536-dimensional vectors
- **WasmEdge Acceleration**: High-performance processing when available
- **Drag & Drop Interface**: Intuitive web-based file upload
- **Real-Time Progress**: Live updates during processing
- **Auto-Deploy Ready**: Generates Hugging Face URLs for instant Gaia Node deployment
- **Cloud & Local**: Works with Qdrant Cloud or local instances

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Qdrant instance (local or cloud)
- Hugging Face account (for uploads)
- Optional: WasmEdge for performance optimization

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/gaia-console.git
   cd gaia-console
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Configure your environment**
   ```env
   # Qdrant Configuration
   QDRANT_URL=http://localhost:6333
   QDRANT_API_KEY=your_qdrant_api_key

   # Hugging Face Configuration
   HF_TOKEN=your_hugging_face_token
   HF_DATASET_NAME=your_username/your_dataset_name

   # Optional Settings
   MAX_FILE_SIZE=10485760  # 10MB
   ```

5. **Start the server**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

6. **Open your browser** and navigate to `http://localhost:8000`

## Usage

### Basic Workflow

1. **Upload Files**: Drag and drop your knowledge base files (TXT, MD, PDF, CSV)
2. **Process**: Click "Generate Snapshot" and watch real-time progress
3. **Deploy**: Copy the generated Hugging Face URL and configuration commands
4. **Use with Gaia Node**:
   ```bash
   gaianet config \
     --snapshot YOUR_SNAPSHOT_URL \
     --embedding-url https://huggingface.co/gaianet/gte-Qwen2-1.5B-instruct-GGUF/resolve/main/gte-Qwen2-1.5B-instruct-f16.gguf \
     --embedding-ctx-size 8192
   
   gaianet init
   gaianet start
   ```

### Supported File Formats

| Format | Description | Processing Method |
|--------|-------------|-------------------|
| **TXT** | Plain text files | Paragraph-based chunking |
| **MD** | Markdown documents | Header-based sectioning |
| **PDF** | PDF documents | Converted to Markdown via markitdown |
| **CSV** | Tabular data | Row-based processing |

### File Size Limits
- Maximum file size: 10MB per file
- No limit on number of files per session
- Batch processing supported

## Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `QDRANT_URL` | Qdrant instance URL | Yes | `http://localhost:6333` |
| `QDRANT_API_KEY` | Qdrant API key (for cloud) | No | - |
| `HF_TOKEN` | Hugging Face write token | Yes | - |
| `HF_DATASET_NAME` | Target HF dataset | Yes | - |
| `MAX_FILE_SIZE` | Max file size in bytes | No | `10485760` |

### Setting up Qdrant

#### Local Qdrant (Docker)
```bash
docker run -p 6333:6333 qdrant/qdrant
```

#### Qdrant Cloud
1. Sign up at [Qdrant Cloud](https://cloud.qdrant.io/)
2. Create a cluster
3. Get your API key and URL
4. Update your `.env` file

### Setting up Hugging Face

1. **Create a Hugging Face account** at [huggingface.co](https://huggingface.co)
2. **Generate an access token** with write permissions
3. **Create a dataset** or use existing one for snapshots
4. **Update your `.env`** with token and dataset name

## Optional: WasmEdge Setup

For optimal performance, install WasmEdge:

```bash
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install_v2.sh | bash
source ~/.bashrc
```

The application will work without WasmEdge but with reduced performance.

## API Reference

### Endpoints

- `GET /` - Web interface
- `POST /process` - Process uploaded files
- `GET /process-stream` - Server-sent events for progress updates
- `GET /check-wasm` - Check WasmEdge availability

### Process API

```bash
curl -X POST "http://localhost:8000/process" \
  -F "files=@document1.txt" \
  -F "files=@document2.pdf" \
  -F "session_id=my_session_123"
```

Response:
```json
{
  "status": "success",
  "snapshot_url": "https://huggingface.co/datasets/your_dataset/resolve/main/snapshots/snapshot_20241201_143022.tar.gz",
  "message": "Processed 2 files with 150 embeddings"
}
```

## Architecture

### System Components

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
│   Web Browser   │◄──►│  FastAPI     │◄──►│   Qdrant    │
│   (Frontend)    │    │  (Backend)   │    │ (VectorDB)  │
└─────────────────┘    └──────────────┘    └─────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  WasmEdge +      │
                    │  gte-Qwen2-1.5B  │
                    │  (AI Processing) │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Hugging Face    │
                    │  (Storage)       │
                    └──────────────────┘
```

### Processing Pipeline

1. **File Upload** → Temporary storage with validation
2. **Format Detection** → Route to appropriate processor
3. **Content Extraction** → Text extraction and cleaning
4. **Embedding Generation** → AI model creates vectors
5. **Vector Storage** → Qdrant collection creation
6. **Snapshot Creation** → Database export and compression
7. **Upload & Deploy** → Hugging Face hosting

## Troubleshooting

### Common Issues

**Problem**: "Qdrant connection failed"
```bash
# Solution: Check if Qdrant is running
docker ps | grep qdrant
# Or test connection manually
curl http://localhost:6333/collections
```

**Problem**: "WasmEdge not available" warning
```bash
# Solution: Install WasmEdge (optional but recommended)
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install_v2.sh | bash
```

**Problem**: "Hugging Face upload failed"
```bash
# Solution: Check token permissions
huggingface-cli whoami
# Ensure dataset exists and you have write access
```

**Problem**: PDF processing fails
```bash
# Solution: Install additional markitdown dependencies
pip install markitdown[all]
```

### Debug Mode

Enable debug logging:
```bash
export PYTHONPATH=.
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
import main
"
```

### Performance Tips

1. **Use WasmEdge** for 3-5x faster embedding generation
2. **Batch related files** for better context understanding
3. **Clean documents** before upload (remove headers/footers)
4. **Use local Qdrant** for faster processing (if possible)

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass: `pytest`
6. Commit your changes: `git commit -m 'Add amazing feature'`
7. Push to the branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

### Code Style

This project uses:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting

```bash
# Format code
black .
isort .
flake8 .
```

## Security

- File size limits prevent DoS attacks
- Input validation on all file types
- Temporary file cleanup after processing
- No sensitive data stored permanently
- Secure token handling for external services

## Support

- **Issues**: [GitHub Issues](https://github.com/GaiaNet-AI/gaia-console/issues)
- **Discussions**: [GitHub Discussions](https://github.com/GaiaNet-AI/gaia-console/discussions)
- **Gaia Network**: [Official Website](https://gaianet.ai)

---

Built with ❤️ for the Gaia Network ecosystem
