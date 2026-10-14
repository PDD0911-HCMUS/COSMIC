#!/usr/bin/env bash
# Run build/export on csMachine; load/check/train/valid on ginger.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
env_file="${COSMIC_ENV_FILE:-$project_dir/.devcontainer/.env}"
action="${1:-help}"
if (($#)); then shift; fi

usage() {
    cat <<'EOF'
Usage: bash scripts/lab.sh ACTION [arguments]
  build                  Build the environment image on csMachine.
  export-image [file]     Save the image to a shared /radish archive.
  load-image [file]       Load that archive into Docker on ginger.
  config                 Show the resolved runtime Compose configuration.
  check                  Check GPU, mounted data, and local BERT/DINO assets.
  train [main.py args]    Run GSV training with periodic Pitts30k validation.
  train --detach [args]   Run in the background and print a container ID.
  valid [main.py args]    Evaluate best.pt on Pitts30k-val.
  up                     Start an idle container for a development terminal.
  shell                  Open a terminal in the idle development container.
  logs CONTAINER_ID      Follow a detached job's logs.

The default Docker context is "default" on the machine running this script.
Override COSMIC_DOCKER_CONTEXT explicitly if using a configured remote context.
All GPUs are exposed to the container. Select GPUs for a command with, e.g.:
  CUDA_VISIBLE_DEVICES=3 bash scripts/lab.sh train --detach
  CUDA_VISIBLE_DEVICES=3 bash scripts/lab.sh valid
Default image archive: outputs/docker/cosmic-training-image.tar
EOF
}

case "$action" in
    help|-h|--help) usage; exit 0 ;;
    build|export-image|load-image|config|check|train|valid|up|shell|logs) ;;
    *) usage >&2; exit 2 ;;
esac

command -v docker >/dev/null || { echo "Docker CLI is missing." >&2; exit 1; }
docker_cmd=(docker --context "${COSMIC_DOCKER_CONTEXT:-default}")
if [[ "$action" == logs ]]; then
    (($#)) || { echo "Pass the training container ID." >&2; exit 2; }
    exec "${docker_cmd[@]}" logs --follow "$@"
fi
[[ -f "$env_file" ]] || {
    echo "Missing $env_file. Copy .devcontainer/devcontainer.env.example to .devcontainer/.env and configure paths." >&2
    exit 1
}

runtime_compose() {
    "${docker_cmd[@]}" compose --env-file "$env_file" \
        -f "$project_dir/.devcontainer/docker-compose.yml" "$@"
}
build_compose() {
    "${docker_cmd[@]}" compose --env-file "$env_file" \
        -f "$project_dir/.devcontainer/docker-compose.build.yml" "$@"
}
# Docker does not automatically forward the host's CUDA process mask.
# Preserve an explicitly empty value as well: it intentionally hides all GPUs.
cuda_env_args=()
if [[ -v CUDA_VISIBLE_DEVICES ]]; then
    cuda_env_args=(-e "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
fi
default_archive="$project_dir/outputs/docker/cosmic-training-image.tar"

case "$action" in
    build)
        build_compose build "$@" cosmic
        ;;
    export-image)
        archive="${1:-$default_archive}"
        image="$(build_compose config --images)"
        mkdir -p -- "$(dirname -- "$archive")"
        "${docker_cmd[@]}" image save --output "$archive" "$image"
        printf 'Image saved: %s\n' "$archive"
        ;;
    load-image)
        archive="${1:-$default_archive}"
        [[ -f "$archive" ]] || { echo "Missing image archive: $archive" >&2; exit 1; }
        "${docker_cmd[@]}" image load --input "$archive"
        ;;
    config)
        runtime_compose config "$@"
        ;;
    check)
        runtime_compose run --rm --no-deps --pull never -T "${cuda_env_args[@]}" cosmic \
            python -B scripts/check_lab_runtime.py "$@"
        ;;
    train)
        args=(python -u -B main.py --config config.yaml --mode train --device cuda)
        if [[ "${1:-}" == --detach ]]; then
            shift
            container_id="$(runtime_compose run --no-deps --pull never -d -T "${cuda_env_args[@]}" cosmic "${args[@]}" "$@")"
            printf 'Training container: %s\nFollow logs: bash scripts/lab.sh logs %s\n' "$container_id" "$container_id"
        else
            runtime_compose run --rm --no-deps --pull never -T "${cuda_env_args[@]}" cosmic "${args[@]}" "$@"
        fi
        ;;
    valid)
        runtime_compose run --rm --no-deps --pull never -T "${cuda_env_args[@]}" cosmic \
            python -u -B main.py --config configs/pitts30k.yaml --mode evaluate \
            --device cuda --resume outputs/gsv/checkpoints/best.pt "$@"
        ;;
    up)
        runtime_compose up -d --no-build --pull never cosmic
        ;;
    shell)
        runtime_compose exec cosmic bash
        ;;
esac
