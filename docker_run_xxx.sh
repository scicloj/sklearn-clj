#!/usr/bin/env bash
set -euo pipefail

docker run -it --rm -v $HOME/.m2:/home/user/.m2 -v "$(pwd):/app"  -w /app sklearn-clj "$@"
