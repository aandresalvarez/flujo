#!/usr/bin/env bash
# Fetch all comments and reviews for a GitHub PR and emit a formatted Markdown report.
# Requirements:
#   - gh (GitHub CLI) authenticated (`gh auth login`)
#   - jq
# Usage:
#   fetch_pr_feedback.sh <repo> <pr_number> [out.md]
#     e.g., fetch_pr_feedback.sh aandresalvarez/flujo 554 pr_554_feedback.md
#   fetch_pr_feedback.sh https://github.com/OWNER/REPO/pull/123 [out.md]
#     e.g., fetch_pr_feedback.sh https://github.com/aandresalvarez/flujo/pull/554
# Output:
#   - Writes Markdown with sections:
#       * Issue Comments (Conversation)
#       * Review Comments (Inline)
#       * Reviews (Approve / Request Changes / Comment)

set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  fetch_pr_feedback.sh <repo> <pr_number> [out.md]
  fetch_pr_feedback.sh https://github.com/OWNER/REPO/pull/123 [out.md]

Examples:
  fetch_pr_feedback.sh aandresalvarez/flujo 554
  fetch_pr_feedback.sh https://github.com/aandresalvarez/flujo/pull/554 pr_554_feedback.md

Requires: gh (authenticated) and jq.
EOF
  exit 1
}

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI is required. Install from https://cli.github.com/ and run 'gh auth login'." >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required. Install via your package manager." >&2
  exit 1
fi

if [[ $# -lt 1 ]]; then
  usage
fi

REPO=""
PR=""
OUT=""

if [[ $1 == https://github.com/*/pull/* ]]; then
  if [[ $1 =~ github.com/([^/]+)/([^/]+)/pull/([0-9]+) ]]; then
    REPO="${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"
    PR="${BASH_REMATCH[3]}"
    OUT="${2:-pr_${PR}_feedback.md}"
  else
    echo "Could not parse PR URL: $1" >&2
    exit 1
  fi
else
  if [[ $# -lt 2 ]]; then
    usage
  fi
  REPO="$1"
  PR="$2"
  OUT="${3:-pr_${PR}_feedback.md}"
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

echo "Fetching feedback for $REPO #$PR ..."

gh api --paginate "/repos/${REPO}/issues/${PR}/comments" > "${tmpdir}/issue_comments.json"
gh api --paginate "/repos/${REPO}/pulls/${PR}/comments" > "${tmpdir}/review_comments.json"
gh api --paginate "/repos/${REPO}/pulls/${PR}/reviews"  > "${tmpdir}/reviews.json"

{
  echo "# PR Feedback for ${REPO} #${PR}"
  echo

  echo "## Issue Comments (Conversation)"
  if jq -e 'length>0' "${tmpdir}/issue_comments.json" >/dev/null; then
    jq -r '
      def fmt(t): (t | fromdateiso8601 | strftime("%Y-%m-%d %H:%M:%S UTC"));
      .[] |
        "- @\(.user.login) at \(fmt(.created_at))" +
        "\n  - " + ((.body // "" ) | gsub("\r";"") | gsub("\n";"\n  "))
    ' "${tmpdir}/issue_comments.json"
  else
    echo "- None"
  fi
  echo

  echo "## Review Comments (Inline)"
  if jq -e 'length>0' "${tmpdir}/review_comments.json" >/dev/null; then
    jq -r '
      def fmt(t): (t | fromdateiso8601 | strftime("%Y-%m-%d %H:%M:%S UTC"));
      .[] |
        "- @\(.user.login) at \(fmt(.created_at)) — `\(.path // "unknown"):\(.line // .original_line // "n/a")`" +
        "\n  - " + ((.body // "" ) | gsub("\r";"") | gsub("\n";"\n  "))
    ' "${tmpdir}/review_comments.json"
  else
    echo "- None"
  fi
  echo

  echo "## Reviews (Approve / Request Changes / Comment)"
  if jq -e 'length>0' "${tmpdir}/reviews.json" >/dev/null; then
    jq -r '
      def fmt(t): (t | fromdateiso8601 | strftime("%Y-%m-%d %H:%M:%S UTC"));
      .[] |
        "- @\(.user.login) at \(fmt(.submitted_at)) — **\(.state // "UNKNOWN")**" +
        "\n  - " + ((.body // "" ) | gsub("\r";"") | gsub("\n";"\n  "))
    ' "${tmpdir}/reviews.json"
  else
    echo "- None"
  fi
  echo
} > "${OUT}"

echo "Wrote ${OUT}"

