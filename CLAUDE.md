# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
- **Start server**: `./run.sh` or `cd backend && uv run uvicorn app:app --reload --port 8000`
- **Install dependencies**: `uv sync`
- **Access points**:
  - Web Interface: http://localhost:8000
  - API Documentation: http://localhost:8000/docs

### Environment Setup
- Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`
- Python 3.13+ required with uv package manager

## Architecture Overview

This is a RAG (Retrieval-Augmented Generation) chatbot system with a modular backend architecture:

### Core Components
- **RAGSystem** (`rag_system.py:10`): Main orchestrator that coordinates all components
- **VectorStore** (`vector_store.py:34`): ChromaDB-based vector storage with dual collections:
  - `course_catalog`: Course metadata for semantic course name matching
  - `course_content`: Chunked course content for content search
- **DocumentProcessor** (`document_processor.py:6`): Processes course documents with structured format parsing
- **AIGenerator** (`ai_generator.py:4`): Claude API integration with tool-calling support
- **ToolManager** (`search_tools.py:116`): Manages search tools available to the AI

### Key Patterns
- **Tool-based Search**: AI uses `CourseSearchTool` for semantic course content searches rather than manual vector queries
- **Session Management**: Conversation history tracked per session with configurable history limit
- **Dual Vector Collections**: Separate storage for course metadata vs content enables smart course name resolution
- **Structured Document Format**: Course files follow specific format with metadata headers and lesson markers

### Data Models
- **Course** (`models.py:10`): Course with title (unique ID), optional instructor/link, and lessons list
- **Lesson** (`models.py:4`): Individual lesson with number, title, optional link
- **CourseChunk** (`models.py:17`): Text chunks with course/lesson context for vector storage

### FastAPI Backend Structure
- **Main app** (`app.py:16`): FastAPI server with CORS, static file serving, and API endpoints
- **Endpoints**:
  - `POST /api/query`: Process user queries with RAG system
  - `GET /api/courses`: Get course analytics and statistics
- **Frontend**: Static HTML/CSS/JS served from `/frontend/`

### Configuration
All settings centralized in `config.py:9` with environment variable loading. Key settings include chunk size (800), overlap (100), max results (5), and conversation history limit (2).

### Course Document Format
Expected format for course files in `/docs/`:
```
Course Title: [title]
Course Link: [url]  
Course Instructor: [instructor]

Lesson 1: [lesson title]
Lesson Link: [lesson url]
[lesson content]

Lesson 2: [lesson title]
[lesson content]
```