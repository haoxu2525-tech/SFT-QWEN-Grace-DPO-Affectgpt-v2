#!/bin/sh
# Runs on the operator's computer (Git Bash), not on the pod: read every recorded v2 result from
# the pod (v2_results_report.py, read-only there) into results/ of this repository, and commit
# and push to main when something changed. Only small text files; no weights, no datasets.
# Run once at every status check; there is no loop.
#   GRACE_POD_SSH="-p PORT root@HOST" GRACE_KNOWN_HOSTS=/path/to/known_hosts sh ops/v2_sync_results.sh
: "${GRACE_POD_SSH:?set GRACE_POD_SSH, e.g. \"-p 22184 root@HOST\"}"
: "${GRACE_KNOWN_HOSTS:?set GRACE_KNOWN_HOSTS to a known_hosts file holding the pod host key}"
cd "$(dirname "$0")/.." || exit 1
TAR=${TMPDIR:-/tmp}/grace-v2-results.$$.tar

[ "$(git rev-parse --abbrev-ref HEAD)" = main ] || { echo "not on main; nothing synced"; exit 1; }
# shellcheck disable=SC2086
ssh -o UserKnownHostsFile="$GRACE_KNOWN_HOSTS" -o StrictHostKeyChecking=yes -o ConnectTimeout=20 \
    -o BatchMode=yes $GRACE_POD_SSH /workspace/grace-env/bin/python - < ops/v2_results_report.py > "$TAR" \
  && tar -tf "$TAR" > /dev/null || { echo "$(date -u +%FT%TZ) could not read the pod"; rm -f "$TAR"; exit 1; }
mkdir -p results && tar -xf "$TAR" -C results; rm -f "$TAR"
git add results
# the README carries the time it was written: commit only when another file changed
if ! git diff --cached --quiet -- results ':(exclude)results/README.md' > /dev/null; then
  git commit -q -m "v2 results from the pod" \
      -m "Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>" -- results || exit 1
else
  git reset -q -- results; git checkout -q -- results 2> /dev/null
fi
[ -n "$(git log --oneline origin/main..main -- results)" ] || exit 0
git push -q origin main || { echo "$(date -u +%FT%TZ) commit made, push failed"; exit 1; }
echo "$(date -u +%FT%TZ) pushed v2 results"
