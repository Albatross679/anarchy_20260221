---
name: markdown-for-project
description: Project documentation workflow for logs, experiments, issues, and knowledge in the doc/ folder. Use when (1) completing any code change or task (add log), (2) running simulations, tests, or investigations (add experiment), (3) encountering errors, bugs, or unexpected behavior (add issue), (4) capturing domain knowledge or reference material (add knowledge), (5) creating or editing any .md file in doc/. Each entry is its own file with Obsidian-compatible frontmatter.
---

# Markdown for Project

Project documentation lives in `doc/`. Each entry is a **separate file** in its own subdirectory:

| Subdirectory | When to create | Naming pattern |
|---|---|---|
| `doc/logs/` | After ANY code change or task | `<topic>.md` |
| `doc/experiments/` | When running simulations/tests | `<topic>.md` |
| `doc/issues/` | When encountering bugs/errors | `<topic>.md` |
| `doc/knowledge/` | When capturing domain knowledge, concepts, or reference material | `<topic>.md` |

## Workflow

1. Do the work
2. After completing, create a new file in the appropriate subdirectory
3. If experiments were run, create a file in `doc/experiments/`
4. If issues were found, create a file in `doc/issues/`
5. If domain knowledge or reference material was gathered, create a file in `doc/knowledge/`

## Frontmatter (Obsidian)

Every `.md` file in `doc/` must have this frontmatter:

```yaml
---
id: <uuid-v4>
name: <note-name>
description: <brief description>
type: <experiment|issue|knowledge|log|plan|notes|reference>
created: <YYYY-MM-DDTHH:MM:SS>
updated: <YYYY-MM-DDTHH:MM:SS>
tags: [tag1, tag2]
aliases: [alternate-name]
---
```

Generate UUID: `python -c "import uuid; print(uuid.uuid4())"`
Get datetime: `date +%Y-%m-%dT%H:%M:%S`

## File Naming

- All files: `<topic>.md` (lowercase, hyphenated, no spaces)
- Examples: `reward-shaping.md`, `ppo-training-run.md`, `cpg-frequency-bug.md`

## Entry Templates

See [references/templates.md](references/templates.md) for the exact format of log, experiment, and issue entries.

## Linking

Use basic wikilinks only: `[[note-name]]`

Avoid: `[[note|display]]`, `[[note#heading]]`, `![[embed]]`
