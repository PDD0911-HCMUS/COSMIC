#!/usr/bin/env bash
# csMachine copies source; ginger builds and runs COSMIC from the radish share.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
env_file="${COSMIC_ENV_FILE:-$project_dir/.devcontainer/.env}"
action="${1:-help}"
if (($#)); then shift; fi

usage() {
    cat <<'EOF'
On csMachine: bash scripts/sync_to_radish.sh
On ginger:    bash scripts/lab.sh ACTION [arguments]

  build                  Build COSMIC's own cosmic:gpu-dev image on ginger.
  up                     Start an idle development container.
  shell                  Open its terminal (/workspace/app).
  check                  Check data, pretrained encoders and CUDA.
  train [main.py args]    Train GSV with periodic Pitts30k validation.
  train --detach [args]   Train in the background; print the container ID.
  valid [main.py args]    Evaluate outputs/gsv/checkpoints/best.pt.
  logs CONTAINER_ID      Follow a COSMIC job's output.
  restart                Recreate the idle container after an image rebuild.
  config                 Show the resolved Compose configuration.

Select the GPU per command: CUDA_VISIBLE_DEVICES=3 bash scripts/lab.sh train
Source, model cache, logs and results stay on radish.
Build again after changing Dockerfile.gpu or requirements.
EOF
}

case "$action" in
    help|-h|--help) usage; exit 0 ;;
    build|up|shell|check|train|valid|logs|restart|config) ;;
    *) usage >&2; exit 2 ;;
esac
command -v docker >/dev/null || { echo "Docker CLI is missing." >&2; exit 1; }
docker_cmd=(docker --context default)
if [[ "$action" == logs ]]; then
    (($#)) || { echo "Pass the COSMIC container ID." >&2; exit 2; }
    exec "${docker_cmd[@]}" logs --follow "$@"
fi
[[ -z "${COSMIC_ENV_FILE:-}" || -f "$env_file" ]] || {
    echo "Missing requested environment file: $env_file" >&2; exit 1;
}
compose_args=(-f "$project_dir/.devcontainer/docker-compose.yml")
if [[ -f "$env_file" ]]; then
    compose_args=(--env-file "$env_file" "${compose_args[@]}")
fi
compose() { "${docker_cmd[@]}" compose "${compose_args[@]}" "$@"; }

case "$action" in
    train|valid|check|up|restart)
        "${docker_cmd[@]}" image inspect cosmic:gpu-dev >/dev/null 2>&1 || {
            echo "Build cosmic:gpu-dev on ginger first: bash scripts/lab.sh build" >&2
            exit 1
        }
        ;;
esac
cuda_env_args=()
if [[ -v CUDA_VISIBLE_DEVICES ]]; then
    cuda_env_args=(-e "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
fi

run_python() {
    local detached="$1"
    shift
    local command=(bash -e -o pipefail -c '
        task="$1"; shift
        mkdir -p outputs/logs
        log_file="$(mktemp "outputs/logs/${task}_$(TZ=JST-9 date +%Y%m%d_%H%M%S).XXXXXX.log")"
        printf "Log: %s\n" "$log_file"
        "$@" 2>&1 | tee "$log_file"
    ' cosmic "$action" "$@")
    if [[ "$detached" == 1 ]]; then
        local container_id
        container_id="$(compose run --no-deps --pull never -d -T "${cuda_env_args[@]}" cosmic "${command[@]}")"
        printf 'Training container: %s\nFollow logs: bash scripts/lab.sh logs %s\n' "$container_id" "$container_id"
    else
        compose run --rm --no-deps --pull never -T "${cuda_env_args[@]}" cosmic "${command[@]}"
    fi
}

case "$action" in
    build) compose build "$@" cosmic ;;
    up) compose up -d --no-build --pull never cosmic ;;
    shell) compose exec cosmic bash ;;
    restart) compose up -d --no-build --pull never --force-recreate cosmic ;;
    config) compose config "$@" ;;
    check) run_python 0 python -B scripts/check_lab_runtime.py "$@" ;;
    train)
        detached=0
        if [[ "${1:-}" == --detach ]]; then detached=1; shift; fi
        run_python "$detached" python -u -B main.py --config config.yaml --mode train --device cuda "$@"
        ;;
    valid)
        run_python 0 python -u -B main.py --config configs/pitts30k.yaml --mode evaluate \
            --device cuda --resume outputs/gsv/checkpoints/best.pt "$@"
        ;;
esac
