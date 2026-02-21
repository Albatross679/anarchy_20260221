# Survey Note Template

Use this template for landscape overviews that map an entire field or category. Survey notes help readers understand the breadth of a domain, identify major approaches, and see where the field is heading.

**Naming convention:** `Knowledge/[Field or Category] Landscape.md`

---

## Note Body Structure

```markdown
# [Field/Category] Landscape

## Overview

[What this field is about, why it matters, and the scope of this survey.
Define clear boundaries: what's included and what's not.]

## Historical Timeline

[Key milestones in chronological order. Keep brief — 1 line per milestone.]

- **[YYYY]** — [Milestone description]
- **[YYYY]** — [Milestone description]
- **[YYYY]** — [Milestone description]
- **[YYYY]** — [Milestone description]

## Taxonomy

[ASCII tree or structured breakdown of the field into sub-categories.
Show how the major approaches relate to each other.]

```
[Field Name]
├── [Category A]
│   ├── [Subcategory A1]
│   └── [Subcategory A2]
├── [Category B]
│   ├── [Subcategory B1]
│   └── [Subcategory B2]
└── [Category C]
    └── [Subcategory C1]
```

## Major Approaches

### [Approach/School 1]

- **Key idea**: [1-2 sentence summary]
- **Representative work**: [Paper, tool, or project name]
- **Strengths**: [What it does well]
- **Limitations**: [Where it falls short]
- **Status**: Active / Mature / Declining

### [Approach/School 2]

- **Key idea**: ...
- **Representative work**: ...
- **Strengths**: ...
- **Limitations**: ...
- **Status**: ...

### [Approach/School 3]

[Continue with as many approaches as needed]

## Comparison Matrix

[Compare all major approaches side-by-side.
Only compare methodologically parallel things.]

| Approach | Year | Key Innovation | Current Status | Best For |
|----------|------|---------------|----------------|----------|
| [Approach 1] | YYYY | ... | Active/Mature/Declining | ... |
| [Approach 2] | YYYY | ... | Active/Mature/Declining | ... |
| [Approach 3] | YYYY | ... | Active/Mature/Declining | ... |

## Current Trends

[What's hot, what's declining, emerging directions.
Use relative timeframes ("as of writing") rather than absolute claims about "the latest".
Prefer patterns over perishable specifics.]

- **Rising**: [Trend 1], [Trend 2]
- **Stable**: [Trend 3]
- **Declining**: [Trend 4]

## Open Problems

[Unsolved challenges in the field. These help readers understand where research opportunities exist.]

- **Problem 1**: [Description and why it's hard]
- **Problem 2**: [Description and why it's hard]
- **Problem 3**: [Description and why it's hard]

## Related Concepts

[Aim for 5+ wikilinks. Link every concept mentioned in the body.]

- [[Parent Field]] — Broader field this belongs to
- [[Key Approach 1]] — Major approach covered in this survey
- [[Key Approach 2]] — Another major approach
- [[Related Field]] — Adjacent field worth exploring
- [[Foundation Concept]] — Underlying concept the field builds on

## References

1. [Survey Paper](URL) — Comprehensive survey of the field
2. [Foundational Paper](URL) — Seminal work that defined the field
3. [Recent Review](URL) — Recent state-of-the-art review
```

---

## FileClass Reference

This template uses the `Survey` fileClass defined in `fileClasses/Survey.md`.

### Property Definitions

| Property | Type | Description |
|----------|------|-------------|
| domain | Select | ML/AI, NLP, Deep Learning, Statistics, Mathematics, Computer Science, General |
| scope | Input | Brief description of what the survey covers |
| num_concepts | Number | How many concepts/approaches are covered |
| related_concepts | Multi | Wikilinks to related notes |
| date_created | Date | Creation date |
| date_modified | Date | Last modified date |
| tags | Multi | Obsidian tags |

### Example Frontmatter

```yaml
---
fileClass: Survey
domain: Deep Learning
scope: "Optimization algorithms for training neural networks"
num_concepts: 12
related_concepts:
  - "[[Gradient Descent]]"
  - "[[Adam]]"
  - "[[SGD]]"
  - "[[Learning Rate Scheduling]]"
  - "[[Neural Network Training]]"
date_created: 2026-02-06
date_modified: 2026-02-06
tags:
  - survey
  - deep-learning
  - optimization
---
```
