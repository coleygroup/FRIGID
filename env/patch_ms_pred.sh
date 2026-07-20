#!/usr/bin/env bash
# Apply FRIGID's local ms-pred patches on top of the pinned submodule commit.
#
# Usage:
#   bash env/patch_ms_pred.sh           # apply (idempotent if already applied)
#   bash env/patch_ms_pred.sh --check   # exit 0 if patch already applied / would apply cleanly
#   bash env/patch_ms_pred.sh --reverse # remove the instrument-profile patch
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MS_PRED="${ROOT}/ms-pred"
PATCH="${ROOT}/env/patches/ms-pred-e446eeb-instrument-profile.patch"
EXPECTED_SHA="e446eebb0f83e53ede016c62522ac2dd371801de"

MODE="apply"
if [[ "${1:-}" == "--check" ]]; then
  MODE="check"
elif [[ "${1:-}" == "--reverse" ]]; then
  MODE="reverse"
elif [[ -n "${1:-}" ]]; then
  echo "Unknown option: $1" >&2
  echo "Usage: $0 [--check|--reverse]" >&2
  exit 2
fi

if [[ ! -d "${MS_PRED}/.git" && ! -f "${MS_PRED}/.git" ]]; then
  echo "ms-pred submodule missing at ${MS_PRED}. Run: git submodule update --init ms-pred" >&2
  exit 1
fi

if [[ ! -f "${PATCH}" ]]; then
  echo "Patch file not found: ${PATCH}" >&2
  exit 1
fi

cd "${MS_PRED}"
HEAD_SHA="$(git rev-parse HEAD)"
if [[ "${HEAD_SHA}" != "${EXPECTED_SHA}" ]]; then
  echo "ms-pred HEAD is ${HEAD_SHA}, expected ${EXPECTED_SHA}." >&2
  echo "Checkout the pinned commit first, e.g.:" >&2
  echo "  git -C ms-pred checkout ${EXPECTED_SHA}" >&2
  exit 1
fi

case "${MODE}" in
  check)
    if git apply --reverse --check "${PATCH}" >/dev/null 2>&1; then
      echo "ms-pred already patched (instrument profile)."
      exit 0
    fi
    if git apply --check "${PATCH}" >/dev/null 2>&1; then
      echo "ms-pred patch can be applied cleanly."
      exit 0
    fi
    echo "ms-pred patch does not apply cleanly." >&2
    exit 1
    ;;
  reverse)
    if git apply --reverse --check --whitespace=nowarn "${PATCH}" >/dev/null 2>&1; then
      git apply --reverse --whitespace=nowarn "${PATCH}"
      echo "Reversed instrument-profile patch on ms-pred@${EXPECTED_SHA:0:7}."
    else
      echo "Patch does not appear to be applied; nothing to reverse."
    fi
    ;;
  apply)
    if git apply --reverse --check --whitespace=nowarn "${PATCH}" >/dev/null 2>&1; then
      echo "ms-pred already patched (instrument profile)."
      exit 0
    fi
    git apply --whitespace=nowarn "${PATCH}"
    echo "Applied instrument-profile patch on ms-pred@${EXPECTED_SHA:0:7}."
    echo "Use ICEBERG_INSTRUMENT_PROFILE=msg|canopus before ICEBERG inference."
    ;;
esac
