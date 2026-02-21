# Tutorial Note Template

Use this template for step-by-step practical guides. Tutorials help readers accomplish a specific task with clear instructions and expected outcomes.

**Naming convention:** `Knowledge/How to [Action] with [Tool or Concept].md`

---

## Note Body Structure

```markdown
# How to [Action] with [Tool/Concept]

## Prerequisites

> [!info] Before You Start
> - Prerequisite 1 (link to [[concept note]] if available)
> - Prerequisite 2
> - Required tools/software: ...

## Overview

[What you'll learn, the expected outcome, and estimated time to complete.]

> [!tip] What You'll Build/Achieve
> [Brief description of the end result]

## Steps

### Step 1: [Action]

[Explanation of what this step does and why]

[Code, commands, or instructions:]
```
[code or command]
```

[Expected output or result of this step]

### Step 2: [Action]

[Explanation + code/commands]

### Step 3: [Action]

[Explanation + code/commands]

[Continue with as many steps as needed]

## Common Pitfalls

> [!warning] Watch Out For
> - **Pitfall 1**: What goes wrong → How to fix it
> - **Pitfall 2**: What goes wrong → How to fix it
> - **Pitfall 3**: What goes wrong → How to fix it

## Variations

[Alternative approaches, configurations, or modifications:]

### Variation A: [Description]
[How to adapt the tutorial for a different scenario]

### Variation B: [Description]
[Another adaptation]

## Next Steps

[What to learn or do after completing this tutorial:]

- [[Advanced Topic]] — Take this further
- [[Related Tutorial]] — Similar guide for related task
- [[Underlying Concept]] — Understand the theory behind what you just did

## Related Concepts

[Aim for 5+ wikilinks. Link every concept mentioned in the body.]

- [[Core Concept]] — The main concept this tutorial covers
- [[Tool/Framework]] — Tool used in this tutorial
- [[Alternative Approach]] — Different way to achieve the same goal
- [[Prerequisite]] — Background knowledge needed
- [[Related Concept]] — Related topic worth exploring

## References

1. [Official Documentation](URL) — Primary reference
2. [Tutorial Source](URL) — Inspiration or source material
3. [Additional Resource](URL) — Further reading
```

---

## FileClass Reference

This template uses the `Tutorial` fileClass defined in `fileClasses/Tutorial.md`.

### Property Definitions

| Property | Type | Description |
|----------|------|-------------|
| domain | Select | ML/AI, NLP, Deep Learning, Statistics, Mathematics, Computer Science, General |
| difficulty | Select | Beginner, Intermediate, Advanced |
| prerequisites | Multi | Wikilinks to prerequisite concepts |
| estimated_time | Input | Expected completion time (e.g., "30 minutes") |
| related_concepts | Multi | Wikilinks to related notes |
| date_created | Date | Creation date |
| date_modified | Date | Last modified date |
| tags | Multi | Obsidian tags |

### Example Frontmatter

```yaml
---
fileClass: Tutorial
domain: ML/AI
difficulty: Intermediate
prerequisites:
  - "[[Python]]"
  - "[[Neural Networks]]"
  - "[[Loss Function]]"
estimated_time: "45 minutes"
related_concepts:
  - "[[PyTorch]]"
  - "[[Training Loop]]"
  - "[[Gradient Descent]]"
  - "[[Backpropagation]]"
  - "[[Model Evaluation]]"
date_created: 2026-02-06
date_modified: 2026-02-06
tags:
  - tutorial
  - ml
  - pytorch
---
```
