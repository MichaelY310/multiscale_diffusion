#!/bin/bash
export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118

# Try to get tectonic (single-file LaTeX compiler)
pip install tectonic 2>/dev/null || true

# Check if tectonic works
which tectonic 2>/dev/null && echo "tectonic found" || echo "no tectonic"

# Alternative: use python typst to compile a .typ file
python3 -c "import typst; print('typst python OK')" 2>/dev/null
