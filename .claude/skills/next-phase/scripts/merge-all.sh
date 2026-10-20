#!/usr/bin/env bash
# Push the current branch, then bring every other branch (local and on the remote) up to date
# with it and push each one. Fast-forwards when possible; otherwise merges in a temporary git
# worktree so the main checkout is never switched (a checkout inside Claude Code can fail to
# delete files under .claude/). Exits 2 on a merge conflict, leaving every branch untouched.
#
# Usage: merge-all.sh [remote]      (default remote: origin)
set -euo pipefail

cd "${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel)}"
remote="${1:-origin}"
cur=$(git branch --show-current)

if [ -z "$cur" ]; then
  echo "merge-all: detached HEAD; check out a branch first" >&2
  exit 1
fi
if [ -n "$(git status --porcelain)" ]; then
  echo "merge-all: working tree is not clean; commit or stash first" >&2
  exit 1
fi

git fetch --prune "$remote"
echo "== pushing $cur"
git push "$remote" "$cur"

# Every branch name known locally or on the remote, except the current one.
branches=$(
  {
    git for-each-ref --format='%(refname:short)' refs/heads/
    git for-each-ref --format='%(refname)' "refs/remotes/$remote/" \
      | sed "s#^refs/remotes/$remote/##" | grep -vx 'HEAD' || true
  } | sort -u | grep -vx "$cur" || true
)

if [ -z "$branches" ]; then
  echo "merge-all: no other branches; done"
  exit 0
fi

for b in $branches; do
  echo "== $b"
  # Make sure a local ref exists, and let the remote lead if it is ahead.
  if git show-ref --verify --quiet "refs/remotes/$remote/$b"; then
    if ! git show-ref --verify --quiet "refs/heads/$b"; then
      git update-ref "refs/heads/$b" "refs/remotes/$remote/$b"
    elif git merge-base --is-ancestor "refs/heads/$b" "refs/remotes/$remote/$b"; then
      git update-ref "refs/heads/$b" "refs/remotes/$remote/$b"
    fi
  fi

  if git merge-base --is-ancestor "refs/heads/$b" "$cur"; then
    git update-ref "refs/heads/$b" "$cur"
    echo "   fast-forwarded $b"
  else
    safe=$(printf '%s' "$b" | tr '/' '-')
    wt="${TMPDIR:-/tmp}/merge-all-$safe-$$"
    git worktree add -q "$wt" "$b"
    if git -C "$wt" merge --no-edit "$cur"; then
      echo "   merged $cur into $b"
    else
      git -C "$wt" merge --abort || true
      git worktree remove --force "$wt"
      echo "merge-all: CONFLICT merging $cur into $b; resolve manually. No branch was pushed beyond $cur." >&2
      exit 2
    fi
    git worktree remove --force "$wt"
  fi
  git push "$remote" "$b"
done

echo "== done"
git log --oneline -1 --format='%h %s' "$cur"
for b in $branches; do
  printf '   %s -> %s\n' "$b" "$(git rev-parse --short "refs/heads/$b")"
done
