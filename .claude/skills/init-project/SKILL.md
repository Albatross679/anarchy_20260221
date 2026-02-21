---
name: init-project
description: |
  Set up standard project structure with directories, git, .gitignore, README.md, CLAUDE.md, and Python environment.
  Use when: (1) Starting a new project and need to create the initial directory hierarchy, (2) Initializing a project with standard boilerplate files, (3) Setting up a new Python project environment with venv and dependencies.
---

# Project Initialization

Use this skill when starting a new project to set up the standard structure and files.

## Project Hierarchy

Create the following directory structure:

```
project/
├── .claude/        # Claude Code settings and memory
├── .git/           # Git version control
├── .gitignore
├── README.md
├── CLAUDE.md
├── requirements.txt  # (optional) pinned dependencies
├── pyproject.toml    # (optional) alternative to requirements.txt
├── setup.sh          # (optional) environment setup script
├── data/           # datasets and data files
├── doc/            # documentation (markdown, PDFs, LaTeX)
├── media/          # images, videos, and GIFs
├── model/          # saved model weights
├── output/         # outputs and errors from running
├── script/         # standalone scripts (shell, examples)
├── src/            # core source code
│   └── utils/      # generic reusable helper modules
└── tests/          # unit and integration tests
```

## Initialization Steps

1. **Create directories** (only those needed for the project type):
   ```bash
   mkdir -p data doc media model output script src/utils tests
   ```

2. **Initialize git**:
   ```bash
   git init
   ```

3. **Create .gitignore** with common exclusions:
   - Python: `__pycache__/`, `*.pyc`, `.venv/`, `*.egg-info/`
   - Environment: `.env`, `.env.local`
   - IDE: `.vscode/`, `.idea/`
   - OS: `.DS_Store`, `Thumbs.db`
   - Project: `output/`, `model/`, `.claude/`

4. **Create README.md** with:
   - Project title and description
   - Setup instructions
   - Usage examples
   - License (if applicable)

5. **Create CLAUDE.md** with:
   - Project overview
   - File roles and architecture
   - Commands for setup and running
   - Do Not Modify section (list driver files)
   - Files You Can Modify section

6. **Set up Python environment** (if Python project):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install <dependencies>
   pip freeze > requirements.txt
   ```

## CLAUDE.md Template

```markdown
# CLAUDE.md

## Project Overview

[Brief description of what this project does]

## Do Not Modify

- [List driver/framework files here]

## Files You Can Modify

- [List implementation files here]

## Commands

### Setup
\`\`\`bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
\`\`\`

### Running
\`\`\`bash
[Add run commands]
\`\`\`

## Architecture

[Describe file roles and structure]
```

## Important Notes

- Only create directories that are needed for the specific project
- Do NOT modify CLAUDE.md after initial creation unless explicitly asked
- Keep existing files in root directory; don't move them unless user specifies
- Ask user about project type and requirements before initializing
