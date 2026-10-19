set -uo pipefail
cd /mnt/c/AI/staging/ninfer-all-pin || exit 2
printf '%s\n' running > /mnt/c/AI/tuning/ninfer-migration-20260927/ninfer-build.exit
if timeout --signal=INT --kill-after=30s 7200 docker build --progress=plain --file /mnt/c/AI/tuning/engine-eval-20260926/stage5-Dockerfile.parallel6 --tag ninfer-codex-20260927:sm86 . > /mnt/c/AI/tuning/ninfer-migration-20260927/ninfer-build.log 2>&1; then
    status=0
else
    status=$?
fi
printf '%s\n' "$status" > /mnt/c/AI/tuning/ninfer-migration-20260927/ninfer-build.exit
exit "$status"
