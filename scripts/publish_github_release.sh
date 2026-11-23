#!/usr/bin/env bash
# Publish packaged FrameFlow weights (`dist/`) to a GitHub Release.
# Works locally (with `gh` logged in) OR inside CNB / CI (with `GITHUB_TOKEN` env var).
set -euo pipefail

TAG="${1:?Usage: $0 <release-tag> [dist-dir=dist] [repo=CyberNova2333/frameflow]}"
DIST_DIR="${2:-dist}"
REPO="${3:-${GITHUB_REPOSITORY:-CyberNova2333/frameflow}}"
PRERELEASE="${PRERELEASE:-true}"

if [ ! -d "$DIST_DIR" ]; then
  echo "ERROR: dist directory not found: $DIST_DIR" >&2
  exit 1
fi

NOTES_FILE="$DIST_DIR/RELEASE_NOTES.md"
mapfile -t ASSETS < <(find "$DIST_DIR" -maxdepth 1 -type f ! -name 'RELEASE_NOTES.md' | sort)
if [ "${#ASSETS[@]}" -eq 0 ]; then
  echo "ERROR: no asset files found in $DIST_DIR" >&2
  exit 1
fi

# Path 1: Prefer GitHub CLI (`gh`) when installed
if command -v gh >/dev/null 2>&1; then
  if [ -z "${GH_TOKEN:-}" ] && [ -n "${GITHUB_TOKEN:-}" ]; then
    export GH_TOKEN="$GITHUB_TOKEN"
  fi
  if gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
    echo "[github-release] Updating existing release $TAG on $REPO..."
    [ -f "$NOTES_FILE" ] && gh release edit "$TAG" --repo "$REPO" --notes-file "$NOTES_FILE"
    gh release upload "$TAG" "${ASSETS[@]}" --repo "$REPO" --clobber
  else
    echo "[github-release] Creating release $TAG on $REPO..."
    PRE_FLAG=()
    [ "$PRERELEASE" = "true" ] && PRE_FLAG+=(--prerelease)
    NOTES_ARGS=()
    [ -f "$NOTES_FILE" ] && NOTES_ARGS+=(--notes-file "$NOTES_FILE") || NOTES_ARGS+=(--notes "FrameFlow trained weights $TAG")
    gh release create "$TAG" "${ASSETS[@]}" --repo "$REPO" --title "FrameFlow $TAG" "${PRE_FLAG[@]}" "${NOTES_ARGS[@]}"
  fi
  echo "[github-release] Done: https://github.com/$REPO/releases/tag/$TAG"
  exit 0
fi

# Path 2: Pure curl + python3 fallback (works inside minimal PyTorch Docker containers on CNB!)
TOKEN="${GITHUB_TOKEN:-${GH_TOKEN:-}}"
if [ -z "$TOKEN" ]; then
  echo "ERROR: Neither 'gh' CLI nor GITHUB_TOKEN is available." >&2
  exit 1
fi

python3 - "$REPO" "$TAG" "$DIST_DIR" "$PRERELEASE" <<'PY'
import json, os, sys, urllib.request, urllib.error, urllib.parse
from pathlib import Path

repo, tag, dist_dir, pre_str = sys.argv[1:5]
token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
api = f"https://api.github.com/repos/{repo}"
headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "frameflow-release-publisher",
}

def req(method, url, data=None, content_type="application/json"):
    h = dict(headers)
    if data is not None:
        h["Content-Type"] = content_type
    r = urllib.request.Request(url, data=data, headers=h, method=method)
    with urllib.request.urlopen(r) as resp:
        return json.loads(resp.read().decode("utf-8"))

notes_file = Path(dist_dir) / "RELEASE_NOTES.md"
body = notes_file.read_text(encoding="utf-8") if notes_file.exists() else f"FrameFlow trained weights `{tag}`"

try:
    rel = req("GET", f"{api}/releases/tags/{urllib.parse.quote(tag)}")
    rel = req("PATCH", f"{api}/releases/{rel['id']}", json.dumps({"name": f"FrameFlow {tag}", "body": body}).encode())
    print(f"[github-release] Updated release {tag} (id={rel['id']})")
except urllib.error.HTTPError as e:
    if e.code != 404:
        raise
    payload = {
        "tag_name": tag,
        "name": f"FrameFlow {tag}",
        "body": body,
        "prerelease": (pre_str.lower() == "true"),
    }
    rel = req("POST", f"{api}/releases", json.dumps(payload).encode())
    print(f"[github-release] Created release {tag} (id={rel['id']})")

existing = {a["name"]: a["id"] for a in rel.get("assets", [])}
upload_base = rel["upload_url"].split("{", 1)[0]

for path in sorted(Path(dist_dir).iterdir()):
    if not path.is_file() or path.name == "RELEASE_NOTES.md":
        continue
    if path.name in existing:
        r = urllib.request.Request(f"{api}/releases/assets/{existing[path.name]}", headers=headers, method="DELETE")
        urllib.request.urlopen(r).read()
    url = f"{upload_base}?name={urllib.parse.quote(path.name)}"
    data = path.read_bytes()
    req("POST", url, data=data, content_type="application/octet-stream")
    print(f"[github-release] Uploaded {path.name} ({len(data):,} bytes)")

print(f"[github-release] Done: {rel['html_url']}")
PY
