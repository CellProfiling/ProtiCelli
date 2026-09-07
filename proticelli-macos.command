#!/bin/sh
set -eu

PROTICELLI_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec /bin/sh "$PROTICELLI_ROOT/proticelli-local.sh" "$@"
