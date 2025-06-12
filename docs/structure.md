# Pakati Project Structure

This document outlines the folder and file organization for the Pakati project.

## Root Directory Structure

```
pakati/
├── .github/                  # GitHub workflow configurations
├── app/                      # Frontend web application
├── docs/                     # Documentation files
├── pakati/                   # Main Python package
├── scripts/                  # Utility scripts
├── tests/                    # Test suite
├── .env.example              # Example environment variables
├── .gitignore                # Git ignore patterns
├── LICENSE                   # Project license
├── README.md                 # Project overview
├── pyproject.toml            # Python package configuration
└── requirements.txt          # Python dependencies
```

## Python Package Structure

```
pakati/
├── __init__.py               # Package initialization
├── canvas.py                 # Canvas and region management
├── models/                   # AI model interfaces
│   ├── __init__.py           # Models package initialization
│   ├── base.py               # Base model interface
│   ├── stable_diffusion.py   # Stable Diffusion implementation
│   ├── dalle.py              # DALL-E API implementation
│   ├── claude.py             # Claude API implementation
│   └── controlnet.py         # ControlNet extensions
├── processing/               # Image processing utilities
│   ├── __init__.py           # Processing package initialization
│   ├── diffusion.py          # Core diffusion algorithms
│   ├── masking.py            # Mask creation and manipulation
│   ├── inpainting.py         # Inpainting utilities
│   └── compositing.py        # Image composition tools
├── storage/                  # Project persistence
│   ├── __init__.py           # Storage package initialization
│   ├── project.py            # Project file format
│   ├── history.py            # Edit history tracking
│   └── export.py             # Export utilities
├── server/                   # API server implementation
│   ├── __init__.py           # Server package initialization
│   ├── api.py                # API routes
│   ├── websocket.py          # WebSocket server for real-time updates
│   └── auth.py               # Authentication handlers
└── utils/                    # Shared utility functions
    ├── __init__.py           # Utils package initialization
    ├── config.py             # Configuration management
    ├── logging.py            # Logging utilities
    └── geometry.py           # Geometry calculations
```

## Web Application Structure

```
app/
├── public/                   # Static public assets
│   ├── index.html            # HTML entry point
│   ├── favicon.ico           # Site favicon
│   └── assets/               # Other static assets
├── components/               # React components
│   ├── Canvas/               # Canvas-related components
│   │   ├── Canvas.tsx        # Main canvas component
│   │   ├── Region.tsx        # Region component
│   │   └── RegionControls.tsx # UI for region manipulation
│   ├── Sidebar/              # Sidebar components
│   │   ├── ModelSelector.tsx # AI model selection
│   │   ├── PromptEditor.tsx  # Prompt editing interface
│   │   └── LayerManager.tsx  # Layer/region management
│   ├── Toolbar/              # Tool selection components
│   └── Common/               # Shared UI components
├── hooks/                    # React hooks
│   ├── useCanvas.ts          # Canvas state management
│   ├── useRegions.ts         # Region manipulation
│   └── useGeneration.ts      # Image generation state
├── services/                 # Frontend services
│   ├── api.ts                # API client
│   ├── websocket.ts          # WebSocket client
│   └── storage.ts            # Local storage utilities
├── state/                    # Global state management
│   ├── store.ts              # Redux/Zustand store
│   ├── canvasSlice.ts        # Canvas state slice
│   └── generationSlice.ts    # Generation state slice
├── styles/                   # Global styles
├── utils/                    # Frontend utilities
├── App.tsx                   # Main application component
├── index.tsx                 # Application entry point
├── package.json              # Frontend dependencies
└── tsconfig.json             # TypeScript configuration
```

## Documentation Structure

```
docs/
├── api/                      # API documentation
├── examples/                 # Example projects and use cases
├── guides/                   # User guides
│   ├── getting-started.md    # Getting started guide
│   ├── regions.md            # Working with regions
│   └── models.md             # Using different AI models
├── architecture.md           # System architecture documentation
├── contributing.md           # Contribution guidelines
└── structure.md              # This file - project structure
```

## Tests Structure

```
tests/
├── unit/                     # Unit tests
│   ├── test_canvas.py        # Canvas functionality tests
│   ├── test_masking.py       # Masking tests
│   └── test_models.py        # Model integration tests
├── integration/              # Integration tests
│   ├── test_api.py           # API endpoint tests
│   └── test_generation.py    # End-to-end generation tests
└── fixtures/                 # Test fixtures
    ├── images/               # Sample images for testing
    └── prompts/              # Sample prompts for testing
```

## Configuration Files

```
.env.example
# API keys for various services
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
HUGGINGFACE_API_KEY=your_huggingface_api_key

# Server configuration
PORT=8000
HOST=localhost
DEBUG=False

# Storage configuration
STORAGE_PATH=./storage
```

```
pyproject.toml
[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "pakati"
version = "0.1.0"
description = "A tool for regional control in AI image generation"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]

[project.dependencies]
torch = ">=1.10.0"
transformers = ">=4.15.0"
diffusers = ">=0.10.0"
pillow = ">=9.0.0"
fastapi = ">=0.70.0"
uvicorn = ">=0.15.0"
python-dotenv = ">=0.19.0"
numpy = ">=1.20.0"
requests = ">=2.25.0"

[project.optional-dependencies]
dev = [
    "pytest>=6.0.0",
    "black>=21.5b2",
    "isort>=5.8.0",
    "mypy>=0.812",
]

[project.urls]
"Homepage" = "https://github.com/yourusername/pakati"
"Bug Tracker" = "https://github.com/yourusername/pakati/issues"
