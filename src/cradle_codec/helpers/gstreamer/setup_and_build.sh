#!/usr/bin/env bash
# Script to setup and build gstreamer



# lol
# Edited setup_and_build.sh to explicitly disable documentation building (-Ddoc=disabled -Dgtk_doc=disabled).
# Discovered that even with documentation disabled, the meson.build script in subprojects/gstreamer/docs/ still unconditionally attempts to read gst_plugins_cache.json, which was entirely missing from your repository.
# Created an empty JSON file ({}) at subprojects/gstreamer/docs/plugins/gst_plugins_cache.json to satisfy the build script's requirements.
# Started ./setup_and_build.sh in the background.


mkdir -p subprojects/gstreamer/docs/plugins
echo '{}' > subprojects/gstreamer/docs/plugins/gst_plugins_cache.json

rm -rf builddir
uv run meson setup builddir -Dgstreamer:doc=disabled
uv run meson compile -C builddir

echo "Build complete! You can enter the dev environment using:"
echo "uv run meson devenv -C builddir"
