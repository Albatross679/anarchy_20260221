# Entry Templates

Each entry is its own file. Include frontmatter at the top of every file.

## Log Entry (`doc/logs/<topic>.md`)

```markdown
---
id: <uuid-v4>
name: <topic>
description: <brief description>
type: log
created: <YYYY-MM-DDTHH:MM:SS>
updated: <YYYY-MM-DDTHH:MM:SS>
tags: [log]
aliases: []
---

# Log: [Title]

**Date:** YYYY-MM-DD

**Task:** [Brief description of what was requested]

## Changes Made
- [File changed]: [What was modified]
- [File changed]: [What was modified]

## Summary
[Brief summary of what was accomplished]
```

**Rules:**
- One file per log entry
- List every file modified with a concise description
- Summary should be 1-3 sentences
- Optional sections: `## Verification`, `## Key Findings`, `## Issues Discovered`

## Experiment Entry (`doc/experiments/<topic>.md`)

```markdown
---
id: <uuid-v4>
name: <topic>
description: <brief description>
type: experiment
created: <YYYY-MM-DDTHH:MM:SS>
updated: <YYYY-MM-DDTHH:MM:SS>
tags: [experiment]
aliases: []
---

# Experiment: [Title]

**Date:** YYYY-MM-DD

**Objective:** [What you're trying to learn/test]

## Setup
[Configuration, parameters, environment]

## Results
[Observations, data, outputs]

## Conclusions
[What was learned, next steps]
```

**Rules:**
- One file per experiment
- Include tables for quantitative results
- List generated artifacts (files, plots, data)
- Optional sections: `## Background`, `## Key Findings`, `## Recommendations`, `## Usage`

## Issue Entry (`doc/issues/<topic>.md`)

```markdown
---
id: <uuid-v4>
name: <topic>
description: <brief description>
type: issue
created: <YYYY-MM-DDTHH:MM:SS>
updated: <YYYY-MM-DDTHH:MM:SS>
tags: [issue]
aliases: []
---

# Issue: [Title]

**Date:** YYYY-MM-DD

**Status:** Open | In Progress | Resolved

**Component:** [File or system affected]

## Description
[What the issue is]

## Steps to Reproduce
[How to trigger the issue]

## Workaround / Resolution
[How to fix or avoid it]
```

**Rules:**
- One file per issue
- Include error messages and stack traces in code blocks
- Reference related files with paths
- Update status when resolved

## Knowledge Entry (`doc/knowledge/<topic>.md`)

```markdown
---
id: <uuid-v4>
name: <topic>
description: <brief description>
type: knowledge
created: <YYYY-MM-DDTHH:MM:SS>
updated: <YYYY-MM-DDTHH:MM:SS>
tags: [knowledge]
aliases: []
---

# [Title]

## Overview
[What this concept/topic is about]

## Details
[In-depth explanation, formulas, diagrams, references]

## Related
[Links to related notes, papers, or code]
```

**Rules:**
- One file per concept or topic
- Use for domain knowledge, reference material, or design rationale
- Keep content evergreen — update rather than duplicate
- Optional sections: `## Examples`, `## References`, `## Open Questions`
