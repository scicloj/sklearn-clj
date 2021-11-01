#!/usr/bin/env bash
set -euo pipefail

docker run -it --rm -v $HOME/.m2:/home/user/.m2 -v "$(pwd):/app" -p  12345:12345 -w /app sklearn-clj
