#!/bin/bash
# export_svg.sh — Export all DOT diagrams to SVG
#
# Usage:
#   chmod +x export_svg.sh
#   ./export_svg.sh
#
# Requires: graphviz (brew install graphviz)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SVG_DIR="${SCRIPT_DIR}/svg"

mkdir -p "$SVG_DIR"

echo "Exporting DOT diagrams to SVG..."
echo "================================"

count=0
for dotfile in "$SCRIPT_DIR"/*.dot; do
    if [ -f "$dotfile" ]; then
        basename=$(basename "$dotfile" .dot)
        svgfile="${SVG_DIR}/${basename}.svg"
        echo "  ${basename}.dot → svg/${basename}.svg"
        dot -Tsvg "$dotfile" -o "$svgfile"
        count=$((count + 1))
    fi
done

echo "================================"
echo "Exported ${count} diagrams to ${SVG_DIR}/"
echo ""
echo "Tip: SVGs can be opened in any browser or embedded in Markdown:"
echo "  ![Figure](diagrams/svg/fig_2_1_core_loop.svg)"
