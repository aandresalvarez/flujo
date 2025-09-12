#!/usr/bin/env bash
set -euo pipefail

# Clean invalid git head refs (e.g., files/dirs with spaces like "main 2").
# - Backs up any SHA it finds to tags under refs/tags/invalid/<sanitized-name>
# - Removes invalid refs from .git/refs/heads
# - Optionally runs git fetch --prune --tags and git gc when requested
#
# Usage:
#   scripts/git-clean-invalid-refs.sh [--dry-run] [--gc] [--fetch]
#
# Flags:
#   --dry-run   Show actions without changing anything
#   --gc        Run `git gc --prune=now` after cleanup
#   --fetch     Run `git fetch --prune --tags` after cleanup

dry_run=false
do_gc=false
do_fetch=false

for arg in "$@"; do
  case "$arg" in
    --dry-run) dry_run=true ;;
    --gc)      do_gc=true ;;
    --fetch)   do_fetch=true ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

# Ensure we're in a git repo
if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "Not inside a git repository." >&2
  exit 1
fi

git_dir=$(git rev-parse --git-dir)
heads_dir="$git_dir/refs/heads"

if [ ! -d "$heads_dir" ]; then
  echo "No heads directory at $heads_dir — nothing to do."
  exit 0
fi

sanitize() {
  # Turn a path or name into a safe tag suffix
  # Replace slashes with underscores and spaces with dashes
  printf '%s' "$1" | sed -e 's|/|_|g' -e 's| |-|g'
}

backup_sha_as_tag() {
  local name="$1" sha="$2"
  if git cat-file -e "$sha^{commit}" 2>/dev/null; then
    local tag="invalid/$(sanitize "$name")"
    if git rev-parse -q --verify "refs/tags/$tag" >/dev/null 2>&1; then
      echo "Tag $tag already exists; skipping backup" >&2
    else
      echo "+ tag $tag -> $sha"
      $dry_run || git tag "$tag" "$sha"
    fi
  fi
}

is_valid_ref_name() {
  local name="$1"
  git check-ref-format "refs/heads/$name" >/dev/null 2>&1
}

changed=false

while IFS= read -r -d '' entry; do
  name=$(basename "$entry")
  if is_valid_ref_name "$name"; then
    continue
  fi

  echo "Invalid ref detected: $entry"

  # If it's a file with a SHA, back it up
  if [ -f "$entry" ]; then
    sha=$(sed -n '1p' "$entry" | tr -d '\n' || true)
    if [[ "$sha" =~ ^[0-9a-f]{40}$ ]]; then
      backup_sha_as_tag "$name" "$sha"
    fi
    echo "- removing file: $entry"
    $dry_run || rm -f "$entry"
    changed=true
    continue
  fi

  # If it's a directory (e.g., created by Finder), try to back up any SHAs inside
  if [ -d "$entry" ]; then
    while IFS= read -r -d '' f; do
      rel=${f#"$heads_dir/"}
      sha=$(sed -n '1p' "$f" | tr -d '\n' || true)
      if [[ "$sha" =~ ^[0-9a-f]{40}$ ]]; then
        backup_sha_as_tag "$rel" "$sha"
      fi
    done < <(find "$entry" -type f -print0)
    echo "- removing directory: $entry"
    $dry_run || rm -rf "$entry"
    changed=true
    continue
  fi
done < <(find "$heads_dir" -maxdepth 1 -mindepth 1 -print0)

if $changed; then
  echo "Heads cleaned."
else
  echo "No invalid head refs found."
fi

if $do_fetch; then
  echo "+ git fetch --prune --tags"
  $dry_run || git fetch --prune --tags
fi

if $do_gc; then
  echo "+ git gc --prune=now"
  $dry_run || git gc --prune=now
fi

exit 0

