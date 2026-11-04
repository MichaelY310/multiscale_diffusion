#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
checkpoint_root="${repo_root}/checkpoints"

repo_root="$(realpath -m -- "${repo_root}")"
checkpoint_root="$(realpath -m -- "${checkpoint_root}")"

if [[ ! -d "${checkpoint_root}" ]]; then
    echo "Checkpoint root does not exist: ${checkpoint_root}" >&2
    exit 1
fi

deleted_count=0
deleted_bytes=0

while IFS= read -r -d '' latest_path; do
    latest_path="$(realpath -m -- "${latest_path}")"
    parent_dir="$(dirname -- "${latest_path}")"

    case "${latest_path}" in
        "${checkpoint_root}"/RDD-CSRA-*/latest.pt) ;;
        *)
            echo "Refusing path outside the intended CSRA scope: ${latest_path}" >&2
            exit 1
            ;;
    esac

    [[ -f "${parent_dir}/complete.json" ]] || continue
    [[ -f "${parent_dir}/best.pt" ]] || continue

    file_bytes="$(stat -c '%s' -- "${latest_path}")"
    echo "Deleting completed-run duplicate: ${latest_path} (${file_bytes} bytes)"
    rm -- "${latest_path}"
    deleted_count=$((deleted_count + 1))
    deleted_bytes=$((deleted_bytes + file_bytes))
done < <(
    find "${checkpoint_root}" \
        -mindepth 2 -maxdepth 2 \
        -type f -path "${checkpoint_root}/RDD-CSRA-*/latest.pt" \
        -size +1G -print0
)

echo "Deleted files: ${deleted_count}"
echo "Deleted bytes: ${deleted_bytes}"
df -h "${repo_root}"
