#!/usr/bin/env bash
# Copy source from csMachine to radish using cp.
# Usage: bash scripts/sync_radish.sh [destination/COSMIC]
set -euo pipefail

source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
target_dir="${1:-/radish/phamd/duypd-proj/COSMIC}"
target_dir="$(realpath -m -- "$target_dir")"

fail() { echo "$*" >&2; exit 1; }
[[ "$target_dir" == */COSMIC ]] || fail "Destination must be named COSMIC: $target_dir"
[[ "$target_dir" != "$source_dir" && "$target_dir" != "$source_dir/"* && "$source_dir" != "$target_dir/"* ]] \
    || fail "Source and destination must be separate directories."
if [[ "$target_dir" == /radish/* ]]; then
    mountpoint -q /radish || fail "/radish is not mounted."
fi

required_files=(main.py config.yaml requirements.txt Dockerfile.gpu .devcontainer/docker-compose.yml scripts/lab.sh)
for file in "${required_files[@]}"; do
    [[ -f "$source_dir/$file" ]] || fail "Missing source file: $source_dir/$file"
done
mkdir -p -- "$target_dir"
[[ -w "$target_dir" ]] || fail "Destination is not writable: $target_dir"

# Enumerate regular source files without following dataset links.
# Runtime assets, private config and experiment results stay at the destination.
file_list="$(mktemp)"
trap 'rm -f -- "$file_list"' EXIT
cd -- "$source_dir"
find . \
    \( -type d \( -name .git -o -name .cache -o -name __pycache__ \
        -o -name '.venv*' -o -name venv -o -name env \
        -o -name .agents -o -name .codex -o -name .aws -o -name .ssh \
        -o -name .vscode -o -name .idea \
        -o -path ./outputs -o -path ./runs -o -path ./wandb \
        -o -path ./Dataset -o -path ./Datasets -o -path ./dataset -o -path ./datasets \
        -o -path ./Pitts30k -o -path ./data/OvSGTR/weights \
        -o -path ./comparison/external -o -path ./comparison/weights \
    \) -prune \) -o \
    \( -type f ! -name '*.py[cod]' ! -name '*.pyd' ! -name '*.so' ! -name '*.dll' \
        ! -name .env ! \( -name '.env.*' ! -name .env.example \) ! -name .cosmic-revision \
        ! -path './.devcontainer/devcontainer.env' -print0 \) > "$file_list"

# The small provenance manifest is source; downloaded weights are runtime assets.
if [[ -f data/OvSGTR/weights/assets_manifest.json ]]; then
    printf '%s\0' './data/OvSGTR/weights/assets_manifest.json' >> "$file_list"
fi
copied=0
while IFS= read -r -d '' file; do
    destination="$target_dir/${file#./}"
    resolved_destination="$(realpath -m -- "$destination")"
    [[ "$resolved_destination" == "$target_dir/"* ]] || fail "Destination link points outside COSMIC: $destination"
    [[ ! -L "$destination" ]] || fail "Destination file is a symlink: $destination"
    mkdir -p -- "$(dirname -- "$destination")"
    cp -f -- "$file" "$destination"
    copied=$((copied + 1))
done < "$file_list"

for file in "${required_files[@]}"; do
    [[ -f "$target_dir/$file" ]] || fail "Missing copied file: $target_dir/$file"
done
printf 'Copied %s source files from %s to %s\n' "$copied" "$source_dir" "$target_dir"
