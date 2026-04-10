#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# launch.sh — Start the Weather & Sunrise/Sunset Agent web UI
#
# Usage:
#   chmod +x launch.sh
#   ./launch.sh
#
# Or with a custom port:
#   ./launch.sh --port 9000
# ─────────────────────────────────────────────────────────────────────────

# Resolve the directory this script lives in
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Weather & Sunrise/Sunset Agent — ADK Web UI"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Agents dir : $SCRIPT_DIR"
echo "  URL        : http://localhost:8000"
echo ""

# Change to the agents directory so relative imports work correctly
cd "$SCRIPT_DIR"

# Run ADK web — pass '.' as AGENTS_DIR (current directory contains the agents)
adk web . "$@"
