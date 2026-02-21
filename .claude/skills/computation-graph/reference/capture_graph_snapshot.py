#!/usr/bin/env python3
"""
Playwright script to capture high-resolution snapshots of computation graphs.

Usage:
    python script/capture_graph_snapshot.py [HTML_FILE] [OUTPUT_PNG]

Example:
    python script/capture_graph_snapshot.py media/3layer_layernorm_graph_1.html media/graph_snapshot.png

Requirements:
    pip install playwright
    playwright install chromium
"""

import argparse
import asyncio
from pathlib import Path


async def capture_graph_snapshot(
    html_path: str,
    output_path: str,
    scale_factor: int = 4,
    padding: int = 50,
):
    """
    Capture a high-resolution snapshot of a computation graph HTML file.

    Args:
        html_path: Path to the HTML file
        output_path: Path for the output PNG file
        scale_factor: Device scale factor for high resolution (default: 4 for 4x resolution)
        padding: Padding around the nodes in pixels (default: 50)
    """
    from playwright.async_api import async_playwright

    html_path = Path(html_path).resolve()
    output_path = Path(output_path).resolve()

    if not html_path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        # Launch browser with high DPI settings
        browser = await p.chromium.launch()

        # Create context with high device scale factor for crisp rendering
        context = await browser.new_context(
            device_scale_factor=scale_factor,
            viewport={"width": 1920, "height": 1080},
        )

        page = await context.new_page()

        # Load the HTML file
        await page.goto(f"file://{html_path}")

        # Wait for the page to fully render (including KaTeX)
        await page.wait_for_load_state("networkidle")
        await page.wait_for_timeout(1000)  # Extra time for KaTeX rendering

        # Hide the sidebar and expand container to full width
        await page.evaluate("""
            () => {
                // Hide sidebar
                const sidebar = document.getElementById('sidebar');
                if (sidebar) {
                    sidebar.style.display = 'none';
                }

                // Expand container to full width
                const container = document.getElementById('container');
                if (container) {
                    container.style.width = '100%';
                }
            }
        """)

        # Calculate bounding box of all nodes and fit them in view
        bounds = await page.evaluate("""
            () => {
                const graph = document.getElementById('graph');
                const nodes = document.querySelectorAll('.node');

                if (!nodes.length) return null;

                // Get current transform values
                const style = graph.style.transform;
                const translateMatch = style.match(/translate\\(([\\d.-]+)px,\\s*([\\d.-]+)px\\)/);
                const scaleMatch = style.match(/scale\\(([\\d.-]+)\\)/);

                const currentPanX = translateMatch ? parseFloat(translateMatch[1]) : 0;
                const currentPanY = translateMatch ? parseFloat(translateMatch[2]) : 0;
                const currentScale = scaleMatch ? parseFloat(scaleMatch[1]) : 1;

                // Calculate bounding box of all nodes in graph coordinates
                let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;

                nodes.forEach(node => {
                    // Get node position from style (these are in graph coordinates)
                    const left = parseFloat(node.style.left) || 0;
                    const top = parseFloat(node.style.top) || 0;
                    const width = node.offsetWidth;
                    const height = node.offsetHeight;

                    minX = Math.min(minX, left);
                    minY = Math.min(minY, top);
                    maxX = Math.max(maxX, left + width);
                    maxY = Math.max(maxY, top + height);
                });

                return {
                    minX, minY, maxX, maxY,
                    width: maxX - minX,
                    height: maxY - minY,
                    currentPanX, currentPanY, currentScale
                };
            }
        """)

        if not bounds:
            raise ValueError("No nodes found in the graph")

        print(f"Node bounds: {bounds['width']:.0f}x{bounds['height']:.0f} pixels")

        # Calculate viewport size needed to fit all nodes with padding
        content_width = bounds['width'] + 2 * padding
        content_height = bounds['height'] + 2 * padding

        # Set viewport to fit content at scale 1
        viewport_width = int(content_width)
        viewport_height = int(content_height)

        # Ensure minimum viewport size
        viewport_width = max(viewport_width, 800)
        viewport_height = max(viewport_height, 600)

        print(f"Viewport size: {viewport_width}x{viewport_height}")
        print(f"Output resolution: {viewport_width * scale_factor}x{viewport_height * scale_factor}")

        await page.set_viewport_size({
            "width": viewport_width,
            "height": viewport_height
        })

        # Reset transform to show all nodes with padding
        pan_x = padding - bounds['minX']
        pan_y = padding - bounds['minY']

        await page.evaluate(f"""
            () => {{
                const graph = document.getElementById('graph');
                graph.style.transform = 'translate({pan_x}px, {pan_y}px) scale(1)';
                graph.style.transformOrigin = '0 0';
            }}
        """)

        # Wait for transform to apply
        await page.wait_for_timeout(500)

        # Take the screenshot
        await page.screenshot(
            path=str(output_path),
            full_page=False,
            type="png",
        )

        await browser.close()

    print(f"Snapshot saved to: {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Capture high-resolution snapshot of computation graph HTML"
    )
    parser.add_argument(
        "html_file",
        nargs="?",
        default="media/3layer_layernorm_graph_1.html",
        help="Path to HTML file (default: media/3layer_layernorm_graph_1.html)"
    )
    parser.add_argument(
        "output_file",
        nargs="?",
        default=None,
        help="Output PNG path (default: same name with _snapshot.png suffix)"
    )
    parser.add_argument(
        "--scale", "-s",
        type=int,
        default=4,
        choices=[1, 2, 3, 4],
        help="Device scale factor for resolution (default: 4 for 4x)"
    )
    parser.add_argument(
        "--padding", "-p",
        type=int,
        default=50,
        help="Padding around nodes in pixels (default: 50)"
    )

    args = parser.parse_args()

    # Default output path
    if args.output_file is None:
        html_path = Path(args.html_file)
        args.output_file = str(html_path.parent / f"{html_path.stem}_snapshot.png")

    asyncio.run(capture_graph_snapshot(
        html_path=args.html_file,
        output_path=args.output_file,
        scale_factor=args.scale,
        padding=args.padding,
    ))


if __name__ == "__main__":
    main()
