set -eu
probe_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "${PYTHON:-python3}" -B "$probe_dir/build.py" "$@"
