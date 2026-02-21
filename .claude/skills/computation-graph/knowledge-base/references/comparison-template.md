# Comparison Note Template

Use this template for side-by-side analysis of 2+ concepts. Comparison notes help readers make decisions between related approaches, tools, or techniques.

**Naming convention:** `Knowledge/[Concept A] vs [Concept B].md` (use "vs" for 2, list all for 3+)

---

## Note Body Structure

```markdown
# [Concept A] vs [Concept B] (vs [Concept C])

## Overview

[1-2 sentences: Why compare these concepts? What decision does this help with?]

## At a Glance

[Only compare methodologically parallel things. Every column header should be the same "kind of thing".]

| Aspect | [Concept A] | [Concept B] | [Concept C] |
|--------|-------------|-------------|-------------|
| Year introduced | ... | ... | ... |
| Core idea | ... | ... | ... |
| Best for | ... | ... | ... |
| Key limitation | ... | ... | ... |

## [Concept A]

### How It Works
[Brief explanation of the core mechanism]

### Strengths
- Strength 1
- Strength 2

### Weaknesses
- Weakness 1
- Weakness 2

## [Concept B]

### How It Works
[Brief explanation of the core mechanism]

### Strengths
- Strength 1
- Strength 2

### Weaknesses
- Weakness 1
- Weakness 2

## Head-to-Head Analysis

### When to Choose [A] over [B]
[Specific scenarios, use cases, or constraints that favor A]

### When to Choose [B] over [A]
[Specific scenarios, use cases, or constraints that favor B]

### Common Misconceptions
[Address frequent misunderstandings about the differences]

## Decision Framework

[Flowchart, decision tree, or structured criteria for choosing between the concepts.
Can be a simple list of "If X, use Y" rules or a more detailed analysis.]

> [!tip] Quick Decision Guide
> - **Choose [A] when**: [condition]
> - **Choose [B] when**: [condition]
> - **Consider [C] when**: [condition]

## Related Concepts

[Aim for 5+ wikilinks. Link every concept mentioned in the body.]

- [[Concept A]] — First concept compared
- [[Concept B]] — Second concept compared
- [[Parent Category]] — Broader category both belong to
- [[Related Tool]] — Relevant tool or framework
- [[Alternative Approach]] — Another option not covered here

## References

1. [Source Title 1](URL) — Brief description
2. [Source Title 2](URL) — Brief description
3. [Source Title 3](URL) — Brief description
```

---

## FileClass Reference

This template uses the `Comparison` fileClass defined in `fileClasses/Comparison.md`.

### Property Definitions

| Property | Type | Description |
|----------|------|-------------|
| concepts_compared | Multi | List of concepts being compared |
| domain | Select | ML/AI, NLP, Deep Learning, Statistics, Mathematics, Computer Science, General |
| verdict | Input | Brief recommendation or conclusion |
| related_concepts | Multi | Wikilinks to related notes |
| date_created | Date | Creation date |
| date_modified | Date | Last modified date |
| tags | Multi | Obsidian tags |

### Example Frontmatter

```yaml
---
fileClass: Comparison
concepts_compared:
  - SGD
  - Adam
  - AdaGrad
domain: Deep Learning
verdict: "Adam for most cases; SGD with momentum for final fine-tuning"
related_concepts:
  - "[[Gradient Descent]]"
  - "[[Learning Rate]]"
  - "[[Neural Network Training]]"
  - "[[Loss Function]]"
  - "[[Optimization]]"
date_created: 2026-02-06
date_modified: 2026-02-06
tags:
  - comparison
  - deep-learning
  - optimization
---
```
