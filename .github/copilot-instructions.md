# Copilot Instructions for eve-pgvector-langgraph

## Project Overview
This project is a demo that vectorizes content from Eve Online MediaWiki to PgVector database. This vector database is used by LangGraph bot for Resource Augmented Generation (RAG).
Most of this project is ran from containers, with exception of script that handles MediaWiki crawling into PgVector.

## Environment Setup
- **Activate Python venv:**
  ```bash
  source .venv/bin/activate
  ```
- All Python commands should run inside the activated virtual environment.

## Directory Structure
- `iac/`: Reserved for infrastructure-as-code (currently empty)
- `.venv/`: Python virtual environment (do not commit)
- `readme.md`: Minimal setup instructions

## Coding Conventions
- Follow standard Python best practices unless project-specific patterns emerge.
- No custom linting, formatting, or test commands are documented yet.

## AI Agent Guidance
- If adding new code, prefer Python and place source files in the project root or a new `src/` directory.
- If starting infrastructure work, use the `iac/` directory and document any conventions in this file.
- Update this file as new workflows, conventions, or architectural decisions are introduced.

## Example Workflow
1. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```
2. Add Python source files to the root or `src/`.
3. Document new conventions here as the project evolves.

---
**Note:** This file should be updated as the codebase grows and new patterns or workflows are established.
