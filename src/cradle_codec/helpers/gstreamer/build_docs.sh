#!/usr/bin/env bash
# Script to build gstreamer documentation

echo "Installing hotdoc..."
uv pip install hotdoc

echo "Configuring meson to enable docs and introspection..."
uv run meson setup --reconfigure -Ddoc=enabled -Dintrospection=enabled builddir

echo "Compiling gstreamer documentation..."
uv run meson compile -C builddir gst-doc

echo "Documentation built! You can visualize it by running 'devhelp' inside the dev environment."
