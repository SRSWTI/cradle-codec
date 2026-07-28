#!/usr/bin/env bash
# Script to update gstreamer subprojects

echo "Updating meson subprojects..."
uv run meson subprojects update
echo "Subprojects updated!"
