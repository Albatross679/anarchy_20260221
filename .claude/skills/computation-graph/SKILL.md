---
name: computation-graph
description: |
  Generate or modify interactive HTML computation graphs with consistent visual style, draggable nodes, SVG edges, and KaTeX formula rendering.
  Use when: (1) Creating a computation graph or architecture diagram, (2) Visualizing a neural network, transformer, or model architecture, (3) Generating an interactive HTML graph with nodes and edges, (4) Modifying an existing computation graph, (5) Capturing a high-resolution snapshot of a computation graph.
---

# Computation Graph

Generate or modify interactive HTML computation graphs with a consistent visual style and structure.

## Trigger

Use when the user asks to:
- Create a computation graph or architecture diagram
- Visualize a neural network, transformer, or model architecture
- Generate an interactive HTML graph with nodes and edges
- Modify an existing computation graph
- Capture a high-resolution snapshot of a computation graph

## Input

The user provides:
- A description of the model/computation to visualize, OR
- An existing HTML graph file to modify, OR
- A request to capture a snapshot of a graph

## Output Format

### Interactive HTML Graph

A self-contained HTML file with:
- **Nodes**: Draggable, resizable boxes with headers, bodies, and optional formulas
- **Edges**: SVG paths connecting nodes with arrowheads
- **Sidebar**: Legend, controls, and description panel (320px, right side)
- **Interactivity**: Pan, zoom, drag nodes, multi-select, save positions

### Snapshot

A high-resolution PNG captured with Playwright (default 4x scale factor).

## Graph Structure

### Node Data Schema

```javascript
const nodesData = [
  {
    "id": "unique_id",           // Unique identifier
    "type": "NodeType",          // Display type (e.g., "Self-Attention")
    "formula": "LaTeX formula",  // KaTeX-rendered formula (optional)
    "color": "#E91E63",          // Background color (hex)
    "x": 100,                    // X position in pixels
    "y": 200,                    // Y position in pixels
    "width": null,               // Custom width (null for auto)
    "height": null,              // Custom height (null for auto)
    "sideLabel": "LAYER 1",      // Vertical side label (optional)
    "sections": [                // Additional info sections (optional)
      { "label": "Input", "content": "(B, T, d)" },
      { "label": "Output", "content": "(B, T, d)" }
    ]
  }
];
```

### Edge Data Schema

```javascript
const edgesData = [
  {
    "source": "source_node_id",
    "target": "target_node_id",
    "style": "vertical_down"     // Edge routing style
  }
];
```

### Edge Styles

| Style | Description |
|-------|-------------|
| `vertical_down` | Straight down from source to target |
| `vertical_up` | Straight up from source to target |
| `horizontal_right` | Straight right from source to target |
| `horizontal_left` | Straight left from source to target |
| `skip_right` | Curved path bypassing nodes on the right |
| `skip_left` | Curved path bypassing nodes on the left |

## Color Palette

Use consistent colors for component types:

| Component | Color | Hex |
|-----------|-------|-----|
| Input/Output | Blue | `#2196F3` |
| Embedding | Green | `#4CAF50` |
| Self-Attention | Pink | `#E91E63` |
| Layer Normalization | Cyan | `#00BCD4` |
| Feed-Forward Network | Orange | `#FF9800` |
| Linear Projection | Purple | `#9C27B0` |
| Residual/Add | Blue-Grey | `#607D8B` |
| Softmax | Teal | `#009688` |

## Visual Style

1. **Nodes**:
   - Border radius: 12px
   - Box shadow: `0 4px 12px rgba(0,0,0,0.3)`
   - Font: System UI stack (Apple, Segoe UI, Roboto)
   - Header: 14px bold, centered
   - Body: 12px, left-aligned with sections

2. **Formulas**:
   - Rendered with KaTeX
   - Background: `rgba(255,255,255,0.1)`
   - Padding: 4px 8px
   - Border radius: 6px

3. **Edges**:
   - Stroke width: 3px
   - Color: `#666` (normal), `#999` (skip connections)
   - Arrow markers at target end

4. **Layout**:
   - White background (`#ffffff`)
   - Sidebar: 320px fixed right, `#f5f5f5` background
   - Container: Full width minus sidebar

## Snapshot Preferences

When capturing snapshots:
- **Scale factor**: 4x (highest resolution)
- **Padding**: 50px around all nodes
- **Sidebar**: Hidden during capture
- **Format**: PNG
- **Viewport**: Auto-sized to fit all nodes

Use the Playwright script at `reference/capture_graph_snapshot.py`:

```bash
python .claude/skills/computation-graph/reference/capture_graph_snapshot.py <input.html> <output.png> --scale 4
```

## Workflow

1. **Analyze requirements**: Understand the model/computation structure
2. **Define nodes**: Create node data with positions, colors, formulas
3. **Define edges**: Connect nodes with appropriate edge styles
4. **Generate HTML**: Use the template structure with KaTeX, SVG edges, and interactivity
5. **Save to media/**: Output HTML file to `media/` directory
6. **Capture snapshot** (if requested): Run Playwright script with 4x scale

## Rules

1. Use KaTeX CDN for LaTeX formula rendering
2. All nodes must be draggable and resizable
3. Include a legend in the sidebar matching node colors
4. Support save functionality (File System Access API with fallback)
5. Position nodes to minimize edge crossings
6. Use side labels for grouped components (e.g., "LAYER 1", "LAYER 2")
7. Include shape dimensions in node sections when relevant

## Example

**Input:**
> Create a computation graph for a 2-layer transformer encoder

**Output:**
An interactive HTML file with:
- Input node
- Embedding node
- 2x Transformer layers (each with Attention, Add, LayerNorm, FFN, Add, LayerNorm)
- Output projection node
- All edges connecting the data flow
- Sidebar with legend and controls

## Reference Files

- [Template HTML structure](reference/template.html) - Base HTML template
- [Playwright snapshot script](reference/capture_graph_snapshot.py) - High-res screenshot capture
