#!/usr/bin/env bash
# Publish the csMachine working copy to shared storage; Git stays on csMachine.
set -euo pipefail

source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
target_dir="${1:-/radish/phamd/duypd-proj/COSMIC}"
target_dir="$(realpath -m -- "$target_dir")"

[[ -f "$source_dir/main.py" && -f "$source_dir/config.yaml" ]] || {
    echo "Source is not a COSMIC project: $source_dir" >&2; exit 1;
}
# --delete is restricted to a COSMIC directory under the mounted radish share.
case "$target_dir" in
    /radish/*/COSMIC) ;;
    *) echo "Target must be a COSMIC directory below /radish: $target_dir" >&2; exit 1 ;;
esac
[[ "$target_dir" != "$source_dir" && "$target_dir" != "$source_dir/"* ]] || {
    echo "Refusing to synchronize into the source tree." >&2; exit 1;
}
mountpoint -q /radish || { echo "/radish is not mounted." >&2; exit 1; }
command -v rsync >/dev/null || { echo "Install rsync on csMachine first." >&2; exit 1; }
datasets_dir="$(dirname -- "$target_dir")/SG-Retrieval/Datasets"
[[ -d "$datasets_dir" ]] || {
    echo "Missing sibling datasets directory: $datasets_dir" >&2; exit 1;
}
mkdir -p -- "$target_dir"

# A relative link also resolves inside the container's sibling bind mounts.
datasets_link="$target_dir/Datasets"
link_target='../SG-Retrieval/Datasets'
if [[ -L "$datasets_link" ]]; then
    [[ "$(realpath -e -- "$datasets_link")" == "$(realpath -e -- "$datasets_dir")" ]] || {
        echo "Datasets points elsewhere; refusing to replace it: $datasets_link" >&2; exit 1;
    }
    if [[ "$(readlink -- "$datasets_link")" != "$link_target" ]]; then
        ln -sfnT -- "$link_target" "$datasets_link"
    fi
elif [[ -e "$datasets_link" ]]; then
    echo "Datasets exists and is not a symlink: $datasets_link" >&2; exit 1
else
    ln -s -- "$link_target" "$datasets_link" || {
        echo "Cannot create the Datasets symlink. Check symlink support on the radish SMB mount." >&2
        exit 1
    }
fi
[[ -L "$datasets_link" && -d "$datasets_link" ]] || {
    echo "The Datasets symlink is not usable on this mount: $datasets_link" >&2; exit 1;
}

# Excluded runtime directories are protected from deletion on the receiver.
# No trailing slash on dataset exclusions: protect symlinks as well as directories.
# Keep the pinned asset manifest while preserving downloaded weights.
# SMB may reject preserved timestamps/permissions. Compare content checksums
# instead of modification times so equal-size code edits still synchronize.
rsync -rc --delete --human-readable --info=progress2 \
    --exclude='/.git/' --exclude='/.cache/' --exclude='/outputs/' \
    --exclude='/Dataset' --exclude='/Datasets' --exclude='/datasets' \
    --exclude='/Pitts30k' --exclude='/comparison/external/' --exclude='/comparison/weights/' \
    --exclude='/.devcontainer/.env' --exclude='/.devcontainer/devcontainer.env' \
    --exclude='/.env' --exclude='/.cosmic-revision' \
    --exclude='/.venv/' --exclude='/.vscode/' --exclude='/.idea/' \
    --exclude='/.agents/' --exclude='/.codex/' --exclude='/.aws/' --exclude='/.ssh/' \
    --exclude='__pycache__/' --exclude='*.py[cod]' \
    --exclude='*.pyd' --exclude='*.so' \
    --include='/data/OvSGTR/weights/' \
    --include='/data/OvSGTR/weights/assets_manifest.json' \
    --exclude='/data/OvSGTR/weights/*' \
    "$source_dir/" "$target_dir/"

if command -v git >/dev/null && git -C "$source_dir" rev-parse --verify HEAD >/dev/null 2>&1; then
    revision="$(git -C "$source_dir" rev-parse HEAD)"
    if [[ -n "$(git -C "$source_dir" status --porcelain)" ]]; then revision="$revision (working copy modified)"; fi
    printf '%s\n' "$revision" > "$target_dir/.cosmic-revision"
fi
printf '\nSource synchronized to %s\n' "$target_dir"
printf 'Datasets symlink: %s -> %s\n' "$datasets_link" "$link_target"
