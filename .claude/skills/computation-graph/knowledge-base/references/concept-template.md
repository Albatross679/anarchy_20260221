# Concept Note Template (Introduction + Deep Dive)

Use this template for both **Introduction** and **Deep Dive** notes. The difference:

- **Introduction**: All sections at standard length, breadth-focused. Suitable for someone encountering the concept for the first time.
- **Deep Dive**: Shorter Definition (assumes reader knows basics). Adds extended "Internals / Mechanism" section. Deeper mathematical treatment, edge cases, failure modes.

---

## Note Body Structure

```markdown
# [Full Concept Name] ([Abbreviation])

## Definition

[2-3 sentences. Identify the FUNDAMENTAL TENSION the concept addresses.
Surface PROXY RELATIONSHIPS — what does this concept approximate, and what gets lost?
For Deep Dive: keep to 1-2 sentences since readers already know the basics.]

## Concept Family Tree

[ASCII tree showing where this concept fits in the broader taxonomy.
Place the concept at the correct taxonomic level.
Include at least 2 levels above and 1 level below.
Never make the concept its own child or parent.]

```
[Root Category]
├── [Branch 1]
│   └── [Sub-branch]
└── [Branch 2]
    ├── [Distant Cousins]
    └── [Closer Branch]
        ├── [Cousins]
        └── [Parent Category]
            ├── [Sibling 1]
            ├── [Sibling 2]
            └── **[THIS CONCEPT]** ← YOU ARE HERE
                ├── [Child 1]
                ├── [Child 2]
                └── [Child 3]
```

## Comparison & Trade-offs

[Only compare methodologically parallel things in the same table.
Every column header should be the same "kind of thing".]

| Aspect | [This Concept] | [Counterpart 1] | [Counterpart 2] |
|--------|----------------|-----------------|-----------------|
| Purpose | ... | ... | ... |
| When to use | ... | ... | ... |
| Strengths | ... | ... | ... |
| Limitations | ... | ... | ... |

## Historical Context & Sophistication

> [!info] Origin
> **Introduced:** [YYYY] by [Author(s)/Organization]
> **Original Work:** "[Paper/Book Title]"
> **Sophistication Level:** [Foundational | Intermediate | Advanced | State-of-the-Art]

[1-2 paragraphs explaining:]
- When and why this concept emerged
- The problem it was designed to solve
- How sophisticated it is: basic building block or cutting-edge technique?
- How it has evolved since introduction (key milestones)

## Key Characteristics

- **Characteristic 1**: Description
- **Characteristic 2**: Description
- **Characteristic 3**: Description

## Governing Principles

[CONDITIONAL: Include when a known law, principle, or constraint governs the concept.
Skip if no recognized governing principle applies.]

> [!abstract] [Principle Name]
> **"[Statement of the principle]"**
> — [Attribution] ([Year])
>
> [Explanation of how this principle specifically governs the concept being documented.
> This is where you connect the abstract principle to the concrete concept.]

[If limitations flow from the governing principle, organize them here rather than as a flat list.]

## Design Criteria

[CONDITIONAL: Include ONLY for tools, methods, and frameworks — things that can be designed well or poorly.
Skip for pure theoretical concepts.
Ask: "What makes a good [concept]?"]

A good [concept] should:
- **Criterion 1**: Explanation
- **Criterion 2**: Explanation
- **Criterion 3**: Explanation

## Known Limitations

[Organize under governing principles when applicable. Otherwise use a straightforward list.]

- **Limitation 1**: Description and implications
- **Limitation 2**: Description and implications

## Examples

[Use the most natural format for the concept. Options include:]
- Narrative explanation with concrete scenario
- Numerical calculation with step-by-step walkthrough
- Before/after demonstration
- Code snippet with output
- Visual/diagram description

[Don't force the rigid Setup/Calculation/Result structure. Pick what makes the concept clearest.]

### Example 1: [Descriptive Title]

[Detailed example]

### Example 2: [Descriptive Title]

[Another example showing a different use case or perspective]

## Internals / Mechanism

[DEEP DIVE ONLY: Extended section covering internal workings, implementation details,
edge cases, failure modes, and subtleties not covered in an Introduction note.
For Introduction notes, skip this section entirely.]

## Mathematical Formulation

[CONDITIONAL: Include only for concepts with core formulas.
Skip for high-level concepts, tools, or frameworks.]

$$
[LaTeX formula]
$$

Where:
- $variable_1$ = description
- $variable_2$ = description

## Use Cases

> [!example] Use Cases
> - **Use Case 1**: Description of when/why to use this concept
> - **Use Case 2**: Another application scenario
> - **Use Case 3**: Domain-specific application

## Related Concepts

[Aim for 5+ wikilinks. Link every concept mentioned in the body that has or could have its own note.]

- [[Hypernym Concept]] — Broader category this belongs to
- [[Hyponym Concept]] — More specific variant
- [[Counterpart Concept]] — Often compared with this concept
- [[Related Concept 1]] — Related topic worth exploring
- [[Related Concept 2]] — Another related topic

## References

1. [Source Title 1](URL) — Brief description
2. [Source Title 2](URL) — Brief description
3. [Source Title 3](URL) — Brief description
```

---

## FileClass Reference

This template uses the `Concept` fileClass defined in `fileClasses/Concept.md`.

### Property Definitions

| Property | Type | Description |
|----------|------|-------------|
| note_type | Select | Introduction or Deep Dive |
| domain | Select | ML/AI, NLP, Deep Learning, Statistics, Mathematics, Computer Science, General |
| year_introduced | Number | Year the concept was first introduced |
| sophistication | Select | Foundational, Intermediate, Advanced, State-of-the-Art |
| antonyms | Multi | True opposite concepts |
| synonyms | Multi | Alternative names |
| hypernyms | Multi | Broader category concepts (IS-A) |
| hyponyms | Multi | More specific concepts |
| counterparts | Multi | Related comparison concepts |
| related_concepts | Multi | Wikilinks to related notes |
| date_created | Date | Creation date |
| date_modified | Date | Last modified date |
| tags | Multi | Obsidian tags |

### Example Frontmatter

```yaml
---
fileClass: Concept
note_type: Introduction
domain: Deep Learning
year_introduced: 1948
sophistication: Foundational
antonyms: []
synonyms:
  - log loss
  - logistic loss
hypernyms:
  - loss function
  - divergence measure
hyponyms:
  - binary cross-entropy
  - categorical cross-entropy
counterparts:
  - mean squared error
  - hinge loss
related_concepts:
  - "[[Softmax]]"
  - "[[Logistic Regression]]"
  - "[[Neural Network Training]]"
  - "[[KL Divergence]]"
  - "[[Information Theory]]"
date_created: 2026-02-04
date_modified: 2026-02-04
tags:
  - concept
  - deep-learning
  - loss-function
---
```
