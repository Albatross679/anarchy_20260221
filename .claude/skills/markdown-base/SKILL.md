---
name: markdown-base
description: Universal frontmatter standards for all Markdown files. Use when creating or editing any .md file to ensure required properties (date_created, date_modified, tags) are present. Applies to all Markdown files unless explicitly excluded.
---

# Markdown Base

Universal frontmatter requirements for all Markdown files.

## Required Frontmatter

Every Markdown file must include:

```yaml
---
date_created: YYYY-MM-DD
date_modified: YYYY-MM-DD
tags: []
---
```

### Property Rules

- **date_created**: Set once when file is created, never modify
- **date_modified**: Update to current date on every edit
- **tags**: YAML list format, can be empty `[]` or populated `[tag1, tag2]`

## When Editing Existing Files

1. If frontmatter exists but missing required properties, add them
2. Always update `date_modified` to today's date
3. Never overwrite existing `date_created`

## Exceptions

Do NOT apply this skill to files listed in [references/exceptions.md](references/exceptions.md).

Common exclusions:
- README.md files
- CLAUDE.md files
- Skill documentation (SKILL.md)
- Configuration files
