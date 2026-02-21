---
name: knowledge-base
description: |
  Research and document technical knowledge with structured semantic metadata.
  Use when the user asks to research, explain, document, or compare technical concepts,
  create a knowledge note in Obsidian, write a tutorial or how-to guide,
  do a deep dive analysis, create a field survey/landscape overview,
  or review/improve an existing knowledge note.
---

# Knowledge Base Skill

Research and document technical knowledge with structured semantic metadata in Obsidian.

**Note types:**
- **Introduction** — Concept primer for someone encountering the topic for the first time
- **Deep Dive** — Advanced internals, mechanisms, edge cases, failure modes
- **Comparison** — Side-by-side analysis of 2+ concepts to support a decision
- **Tutorial** — Step-by-step practical guide for accomplishing a task
- **Survey** — Landscape overview mapping an entire field or category

## Workflow

### 1. Determine Note Type

Use this decision tree:

| Signal | Note Type | FileClass |
|--------|-----------|-----------|
| Single concept, beginner-friendly | **Introduction** | Concept |
| Single concept, advanced internals | **Deep Dive** | Concept |
| 2+ concepts side-by-side | **Comparison** | Comparison |
| Step-by-step practical guide | **Tutorial** | Tutorial |
| Mapping an entire field/category | **Survey** | Survey |

If the user doesn't specify, default to **Introduction**. Ask only if genuinely ambiguous.

### 2. Research

Use `WebSearch` to gather comprehensive information:

```
WebSearch: "[concept] definition explanation"
WebSearch: "[concept] examples use cases"
WebSearch: "[concept] advantages disadvantages limitations"
WebSearch: "[concept] vs [related concept]" (for counterparts)
WebSearch: "[concept] history origin year introduced paper"
WebSearch: "[concept] governing principles design criteria" (what laws/principles govern this concept?)
```

**IMPORTANT**: Always research:
- When the concept was first introduced (year, original paper/author)
- Sophistication level (basic building block vs cutting-edge)
- Governing principles or laws that constrain the concept (e.g., Goodhart's Law for benchmarks)

### 3. Extract Semantic Relationships

From research, identify:
- **Synonyms**: Alternative names (e.g., "cost function" = "loss function")
- **Antonyms**: True opposites (e.g., "overfitting" ↔ "underfitting")
- **Hypernyms**: Broader categories (IS-A relationship, e.g., "cross-entropy" IS-A "loss function")
- **Hyponyms**: More specific variants (e.g., "binary cross-entropy" is a type of "cross-entropy")
- **Counterparts**: Related concepts often compared (e.g., "precision" & "recall")

### 4. Create the Note

Read the appropriate template from `references/` and create the note:

| Note Type | Template | Location |
|-----------|----------|----------|
| Introduction | `references/concept-template.md` | `Knowledge/[Concept Name].md` |
| Deep Dive | `references/concept-template.md` | `Knowledge/[Concept Name].md` |
| Comparison | `references/comparison-template.md` | `Knowledge/[A] vs [B].md` |
| Tutorial | `references/tutorial-template.md` | `Knowledge/How to [Action].md` |
| Survey | `references/survey-template.md` | `Knowledge/[Field] Landscape.md` |

## Frontmatter Templates

### Concept (Introduction + Deep Dive)

```yaml
---
fileClass: Concept
note_type: Introduction  # or Deep Dive
domain: [ML/AI | NLP | Deep Learning | Statistics | Mathematics | Computer Science | General]
year_introduced: YYYY
sophistication: [Foundational | Intermediate | Advanced | State-of-the-Art]
antonyms:
  - opposite_concept
synonyms:
  - alternative_name
hypernyms:
  - broader_category
hyponyms:
  - specific_variant
counterparts:
  - comparison_concept
related_concepts:
  - "[[Related Note]]"
date_created: YYYY-MM-DD
date_modified: YYYY-MM-DD
tags:
  - concept
  - [domain-tag]
---
```

### Comparison

```yaml
---
fileClass: Comparison
concepts_compared:
  - concept_a
  - concept_b
domain: [ML/AI | NLP | Deep Learning | Statistics | Mathematics | Computer Science | General]
verdict: "Brief recommendation or conclusion"
related_concepts:
  - "[[Related Note]]"
date_created: YYYY-MM-DD
date_modified: YYYY-MM-DD
tags:
  - comparison
  - [domain-tag]
---
```

### Tutorial

```yaml
---
fileClass: Tutorial
domain: [ML/AI | NLP | Deep Learning | Statistics | Mathematics | Computer Science | General]
difficulty: [Beginner | Intermediate | Advanced]
prerequisites:
  - "[[Prerequisite Concept]]"
estimated_time: "30 minutes"
related_concepts:
  - "[[Related Note]]"
date_created: YYYY-MM-DD
date_modified: YYYY-MM-DD
tags:
  - tutorial
  - [domain-tag]
---
```

### Survey

```yaml
---
fileClass: Survey
domain: [ML/AI | NLP | Deep Learning | Statistics | Mathematics | Computer Science | General]
scope: "Brief description of what this survey covers"
num_concepts: 0
related_concepts:
  - "[[Related Note]]"
date_created: YYYY-MM-DD
date_modified: YYYY-MM-DD
tags:
  - survey
  - [domain-tag]
---
```

## Content Quality Guidelines

### Definition Quality

- Identify the **fundamental tension** the concept addresses (e.g., a benchmark's tension is between "measuring real capability" and "being practical to compute")
- Surface **proxy relationships** — what does this concept approximate, and what gets lost? (e.g., "a benchmark score is a proxy for real-world capability")
- Keep to 2-3 sentences. Move elaboration to other sections.

### Governing Principles

After Key Characteristics, include a **Governing Principles** section when a known law or principle constrains the concept:

```markdown
> [!abstract] Goodhart's Law
> **"When a measure becomes a target, it ceases to be a good measure."**
> — Charles Goodhart (1975)
>
> [Explanation of how this principle specifically governs the concept being documented]
```

Organize the **Known Limitations** section under the governing principles — limitations should flow from the principles, not be a flat list.

### Design Criteria (Conditional)

Include a "Design Criteria" section only for tools, methods, and frameworks — things that can be designed well or poorly. Ask: "What makes a good X?" Skip for pure theoretical concepts.

### Content Freshness

- Prefer **patterns** over perishable specifics (say "transformer-based models" not "GPT-4")
- Never hardcode model version numbers or leaderboard rankings
- Use relative timeframes ("as of writing" or "since 2020") instead of absolute claims about "the latest"
- Facts that will age should be in Examples, not in Definition

### Comparison Tables

- Only compare **methodologically parallel** things in the same table
- Bad: comparing a metric with a benchmark suite in the same row
- Good: comparing two metrics, or two benchmark suites, against the same aspects
- Every column header should be the same "kind of thing"

### Section Order (Concept Notes)

The preferred section order for Concept notes (Introduction + Deep Dive) is:

1. **Definition**
2. **Concept Family Tree**
3. **Comparison & Trade-offs**
4. **Historical Context & Sophistication**
5. Key Characteristics
6. Governing Principles (conditional)
7. Design Criteria (conditional)
8. Known Limitations
9. Examples
10. Internals / Mechanism (Deep Dive only)
11. Mathematical Formulation (conditional)
12. Use Cases
13. Related Concepts
14. References

### Concept Family Tree

- Place the concept at the correct **taxonomic level** — don't make it appear as a leaf if it's actually a branch
- Never make the concept appear as its own child or parent (no self-referential loops)
- Include at least 2 levels above and 1 level below the concept

### Related Concepts

- Aim for **5+ wikilinks** in the Related Concepts section
- **Link every concept mentioned in the body** that has or could have its own note
- Use wikilink format: `[[Concept Name]]`

### Conditional Sections

- **Mathematical Formulation**: Only include for concepts with core formulas. Skip for high-level concepts, tools, or frameworks.
- **Design Criteria**: Only for tools/methods/frameworks (see above)
- **Examples**: Use the most natural format — narrative, calculation, before/after, code snippet. Don't force the rigid Setup/Calculation/Result structure.

## Reference Templates

Read the appropriate template when creating a note:

- **Introduction / Deep Dive**: Read `references/concept-template.md`
- **Comparison**: Read `references/comparison-template.md`
- **Tutorial**: Read `references/tutorial-template.md`
- **Survey**: Read `references/survey-template.md`

Each template is self-contained with full body structure, section guidance, and FileClass reference.

## Sophistication Levels Guide

| Level | Description | Example |
|-------|-------------|---------|
| **Foundational** | Core concepts everyone in the field should know. Basic math, widely taught. | Linear regression, gradient descent |
| **Intermediate** | Requires solid fundamentals. Standard in practice but needs training to use well. | CNNs, LSTMs, cross-entropy loss |
| **Advanced** | Cutting-edge techniques requiring deep expertise, significant compute, or specialized knowledge. | Transformers (2017), diffusion models |
| **State-of-the-Art** | Bleeding-edge research, not yet widely adopted. Requires reading recent papers. | Latest architecture innovations |

## Notes

- Always use `WebSearch` for current, accurate information
- **ALWAYS include the year the concept was introduced** — research the original paper/author
- **ALWAYS explain the sophistication level** — Is it a basic building block or cutting-edge?
- Include numerical examples whenever possible for ML/Statistics concepts
- Use LaTeX for mathematical formulations
- Create wikilinks to existing notes in the vault when known
- Keep the definition section concise; use other sections for details
