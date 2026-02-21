---
name: fileclass-management
description: |
  Manage Obsidian notes using Metadata Menu fileClasses.
  Use when creating, modifying, or organizing notes with structured properties.
---

# FileClass Management Skill

## Core Principle

**ALWAYS define `fileClass` explicitly in frontmatter** when creating or modifying notes.

## FileClass Definitions

All fileClass definitions are stored in: `fileClasses/`

Before creating or modifying a note:
1. Read the appropriate fileClass definition file (e.g., `fileClasses/Contact.md`)
2. Use the properties and valid values defined there
3. Always include `fileClass: ClassName` as the first frontmatter property

## Priority Rules

When multiple fileClasses could apply:
1. **Explicit `fileClass` in frontmatter** (highest - always use this)
2. Folder-based query
3. Tag-based mapping (lowest)

Always use explicit frontmatter declaration to avoid ambiguity.

## Creating New FileClasses

When user needs a new note type:
1. Create definition file in `fileClasses/NewType.md`
2. Define properties with types (Input, Select, Multi, Date, etc.)
3. Update Metadata Menu config if folder-based auto-assignment needed
