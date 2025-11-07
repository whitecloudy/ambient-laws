#!/usr/bin/env bash
# Clears all temp directories in the system temp dir that start with 'ambient-rf'
set -euo pipefail
IFS=$'\n\t'

TMPDIR="${TMPDIR:-/tmp}"
PREFIX="ambient-rf"

usage() {
    cat <<EOF
Usage: $(basename "$0") [-n] [-f] [-v] [-h]
    -n   Dry run (list only)
    -f   Force, no confirmation
    -v   Verbose
    -h   Help
EOF
}

DRY_RUN=0
FORCE=0
VERBOSE=0

while getopts "nfvh" opt; do
    case "$opt" in
        n) DRY_RUN=1 ;;
        f) FORCE=1 ;;
        v) VERBOSE=1 ;;
        h) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
done

# Find matching directories (only top-level entries in TMPDIR)
mapfile -d '' dirs < <(find "$TMPDIR" -maxdepth 1 -type d -name "${PREFIX}*" -print0)

if [ "${#dirs[@]}" -eq 0 ]; then
    [ "$VERBOSE" -eq 1 ] && echo "No directories matching '${PREFIX}*' in $TMPDIR"
    exit 0
fi

echo "Found ${#dirs[@]} directory(ies) in $TMPDIR matching '${PREFIX}*':"
for d in "${dirs[@]}"; do
    echo "  $d"
done

if [ "$DRY_RUN" -eq 1 ]; then
    echo "Dry run enabled; no changes made."
    exit 0
fi

if [ "$FORCE" -ne 1 ]; then
    read -r -p "Delete these directories? [y/N] " ans
    case "$ans" in
        [yY]|[yY][eE][sS]) : ;;
        *) echo "Aborted."; exit 1 ;;
    esac
fi

# Remove directories
for d in "${dirs[@]}"; do
    # safety: ensure basename starts with prefix
    base="$(basename "$d")"
    if [[ "$base" != $PREFIX* ]]; then
        echo "Skipping unexpected entry: $d" >&2
        continue
    fi
    if [ "$VERBOSE" -eq 1 ]; then
        echo "Removing: $d"
    fi
    rm -rf -- "$d"
done

echo "Done."