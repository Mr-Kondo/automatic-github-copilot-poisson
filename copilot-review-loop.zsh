#!/bin/zsh
setopt ERR_EXIT NO_UNSET PIPE_FAIL

# =========================================================
# Copilot implementation-review auto loop with auto-commit
# - implementer: Claude Sonnet
# - reviewer: GPT-5.4
# - repeats until PASS with zero issues
# - auto-commits only after PASS
#
# Design notes:
# - Prompts passed to copilot are intentionally short.
# - Large data (task details, review JSON, diff logs, check logs) are read via file paths.
# - *_CMD values are restricted to simple argv-style commands.
#   Shell composition (pipes, redirects, command substitution, env expansion) is intentionally rejected.
# =========================================================

: "${MAX_ITERS:=5}"
: "${LOG_DIR:=.copilot-loop-logs}"
: "${AUTO_COMMIT:=1}"
: "${INCLUDE_TASK_FILE_IN_COMMIT:=0}"
: "${TASK_FILE:=task.md}"
: "${IMPLEMENTER_AGENT:=implementer}"
: "${REVIEWER_AGENT:=reviewer}"
: "${COMMIT_MESSAGE:=}"
: "${UNTRACKED_DIFF_MAX_BYTES:=524288}"  # 512 KiB

: "${TEST_CMD:=}"
: "${LINT_CMD:=}"
: "${TYPECHECK_CMD:=}"
: "${EXTRA_IMPL_INSTR:=}"

usage() {
  cat <<'EOF'
Usage:
  ./copilot-review-loop.zsh [options]

Options:
  -t, --task-file FILE            Task description file (default: task.md)
  -n, --max-iters NUM             Max review loop iterations (default: 5)
  --test-cmd CMD                  Test command (simple argv-style only)
  --lint-cmd CMD                  Lint command (simple argv-style only)
  --typecheck-cmd CMD             Typecheck command (simple argv-style only)
  --log-dir DIR                   Log directory (default: .copilot-loop-logs)
  --implementer-agent NAME        Custom agent name for implementation
  --reviewer-agent NAME           Custom agent name for review
  --commit-message MSG            Commit message after PASS
  --no-auto-commit                Do not auto-commit even if PASS
  --include-task-file             Include task file in commit
  -h, --help                      Show help

Command parsing policy:
- Supported:
    --message "hello world"
    pytest -q
    pnpm test -- --runInBand
- Rejected intentionally:
    echo hi | sed ...
    FOO=bar pytest
    python -c "..."
    cmd > out.txt
    $(subcommand)
    $HOME/bin/tool
- If you need shell composition, wrap it in a script file and call that script.
EOF
}

log() { print -r -- "[INFO] $*"; }
warn() { print -r -- "[WARN] $*" >&2; }
err() { print -r -- "[ERROR] $*" >&2; }

require_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || {
    err "Required command not found: $cmd"
    exit 1
  }
}

timestamp() {
  date +"%Y-%m-%dT%H:%M:%S%z"
}

run_cmd_string() {
  local label="$1"
  local cmd_str="$2"
  local -a argv

  if [[ -z "$cmd_str" ]]; then
    return 0
  fi

  if [[ "$cmd_str" == *$'\n'* ||
        "$cmd_str" == *"|"* ||
        "$cmd_str" == *";"* ||
        "$cmd_str" == *"&"* ||
        "$cmd_str" == *">"* ||
        "$cmd_str" == *"<"* ||
        "$cmd_str" == *'$'* ||
        "$cmd_str" == *'`'* ]]; then
    err "${label}: unsupported shell syntax in *_CMD value."
    err "${label}: use a wrapper script if you need pipes, redirects, env expansion, or command substitution."
    return 90
  fi

  argv=(${(z)cmd_str})

  if (( ${#argv} == 0 )); then
    return 0
  fi

  {
    print -- "===== ${label} ====="
    print -- "\$ ${cmd_str}"
    "${argv[@]}"
    print
  }
}

git_diff_file() {
  local out="$1"
  local raw_untracked=""
  local -a candidate_untracked
  local -a untracked_files
  local file_path
  local fsize
  local log_dir_filter_local="$LOG_DIR_FILTER"

  raw_untracked="$(git ls-files --others --exclude-standard || true)"

  if [[ -n "$raw_untracked" ]]; then
    candidate_untracked=("${(@f)raw_untracked}")
  else
    candidate_untracked=()
  fi

  untracked_files=()
  for file_path in "${candidate_untracked[@]}"; do
    if [[ -n "$log_dir_filter_local" ]]; then
      if [[ "$file_path" == "$log_dir_filter_local" || "$file_path" == "$log_dir_filter_local/"* ]]; then
        continue
      fi
    fi
    untracked_files+=("$file_path")
  done

  {
    print -- "### git branch --show-current"
    git branch --show-current || true
    print

    print -- "### git status --short"
    git status --short || true
    print

    print -- "### git diff --stat (tracked unstaged)"
    git diff --stat || true
    print

    print -- "### git diff (tracked unstaged)"
    git diff || true
    print

    print -- "### git diff --cached (staged)"
    git diff --cached || true
    print

    print -- "### untracked files (LOG_DIR excluded)"
    if (( ${#untracked_files[@]} > 0 )); then
      printf '%s\n' "${untracked_files[@]}"
    else
      print -- "(none)"
    fi
    print

    print -- "### git diff for untracked files"
    if (( ${#untracked_files[@]} > 0 )); then
      for file_path in "${untracked_files[@]}"; do
        if [[ -f "$file_path" ]]; then
          fsize="$(wc -c < "$file_path" 2>/dev/null | tr -d '[:space:]' || true)"
          [[ -z "$fsize" ]] && fsize=0
          if [[ "$fsize" == <-> ]] && (( fsize > UNTRACKED_DIFF_MAX_BYTES )); then
            print -- "--- $file_path (skipped: file too large, ${fsize} bytes)"
            print
            continue
          fi
        fi

        print -- "--- $file_path"
        git diff --no-index -- /dev/null "$file_path" || true
        print
      done
    else
      print -- "(none)"
    fi
  } > "$out"
}

run_project_checks() {
  local out_file="$1"
  : > "$out_file"

  local had_any=0

  if [[ -n "${LINT_CMD}" ]]; then
    had_any=1
    run_cmd_string "LINT" "$LINT_CMD" >> "$out_file" 2>&1 || return 10
  fi

  if [[ -n "${TYPECHECK_CMD}" ]]; then
    had_any=1
    run_cmd_string "TYPECHECK" "$TYPECHECK_CMD" >> "$out_file" 2>&1 || return 11
  fi

  if [[ -n "${TEST_CMD}" ]]; then
    had_any=1
    run_cmd_string "TEST" "$TEST_CMD" >> "$out_file" 2>&1 || return 12
  fi

  if [[ "$had_any" -eq 0 ]]; then
    {
      print -- "No project checks configured."
      print -- "Set one or more of:"
      print -- "  TEST_CMD"
      print -- "  LINT_CMD"
      print -- "  TYPECHECK_CMD"
    } >> "$out_file"
  fi

  return 0
}

validate_review_json() {
  local review_file="$1"
  jq -e '
    type == "object" and
    (.status == "PASS" or .status == "FAIL") and
    (.issues | type == "array") and
    (
      .issues[]? |
      type == "object" and
      (.severity | type == "string") and
      (.severity | length > 0) and
      (.file | type == "string") and
      ((.line == null) or (.line | type == "number")) and
      (.title | type == "string") and
      (.detail | type == "string") and
      (.suggested_fix | type == "string")
    )
  ' "$review_file" >/dev/null
}

summarize_review() {
  local review_file="$1"
  local status issue_count
  status="$(jq -r '.status' "$review_file")"
  issue_count="$(jq -r '.issues | length' "$review_file")"

  print -- "status=${status}, issues=${issue_count}"

  if [[ "$issue_count" -gt 0 ]]; then
    jq -r '.issues[] | "- [\(.severity // "UNKNOWN")] \(.file):\(.line // 0) \(.title)"' "$review_file"
  fi
}

build_impl_prompt_initial() {
  local task_file="$1"
  cat <<EOF
Implement the task described in this repository file:
- ${task_file}

Instructions:
- Read ${task_file} directly from the repository before making changes.
- Edit code directly in the working tree.
- Keep changes minimal and localized.
- Do not modify unrelated files.
- Run relevant project checks when appropriate.
- If a check fails, fix the issue before concluding.
- Summarize changed files, what was fixed, checks run, and remaining risks.

Additional instructions:
${EXTRA_IMPL_INSTR}
EOF
}

build_impl_prompt_revision() {
  local task_file="$1"
  local review_json_file="$2"
  cat <<EOF
Revise the implementation using these sources:
- Task file: ${task_file}
- Review findings JSON: ${review_json_file}

Instructions:
- Read both files directly before making changes.
- Fix every material issue from the review findings.
- Keep changes minimal and localized.
- Do not modify unrelated files.
- Re-run relevant project checks when appropriate.
- If a check fails, fix the issue before concluding.
- Summarize changed files, what was fixed, checks run, and remaining risks.

Additional instructions:
${EXTRA_IMPL_INSTR}
EOF
}

build_review_prompt() {
  local task_file="$1"
  local checks_file="$2"
  local diff_file="$3"
  cat <<EOF
Review the current repository state using these files:
- Task file: ${task_file}
- Latest project checks: ${checks_file}
- Prepared git diff log: ${diff_file}

Instructions:
- Return JSON only, exactly matching the reviewer schema.
- Produce one final verdict only.
- Do not use iterative continuation behavior.
- Read the provided files directly as needed.
- Focus on material correctness, safety, maintainability, and task compliance.
- Prefer concrete findings over style nits.
- If there are no material issues, return:
  {"status":"PASS","issues":[]}
EOF
}

sanitize_commit_subject() {
  local raw="$1"
  raw="$(print -r -- "$raw" | tr '\r\n' ' ' | sed 's/[[:space:]]\+/ /g' | sed 's/^ //; s/ $//')"
  raw="$(print -r -- "$raw" | sed 's/["`$\\]//g')"
  print -r -- "$raw"
}

default_commit_message_from_task() {
  local first_line
  first_line="$(grep -m 1 -v '^[[:space:]]*$' "$TASK_FILE" || true)"
  if [[ -z "$first_line" ]]; then
    first_line="Implement task via Copilot review loop"
  fi
  first_line="$(sanitize_commit_subject "$first_line")"
  first_line="${first_line[1,72]}"
  print -r -- "feat: ${first_line}"
}

stage_files_for_commit() {
  git add -A

  if [[ -e "$LOG_DIR" ]]; then
    git reset -q HEAD -- "$LOG_DIR" 2>/dev/null || true
  fi

  if [[ "$INCLUDE_TASK_FILE_IN_COMMIT" -ne 1 && -e "$TASK_FILE" ]]; then
    git reset -q HEAD -- "$TASK_FILE" 2>/dev/null || true
  fi
}

has_staged_changes() {
  ! git diff --cached --quiet --exit-code
}

show_commit_summary() {
  print -- "### staged diff --stat"
  git diff --cached --stat || true
  print
  print -- "### staged files"
  git diff --cached --name-only || true
}

perform_auto_commit() {
  local commit_msg="$1"

  if ! has_staged_changes; then
    warn "No staged changes to commit after exclusions."
    return 20
  fi

  log "Committing changes..."
  git commit -m "$commit_msg"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -t|--task-file)
      TASK_FILE="$2"
      shift 2
      ;;
    -n|--max-iters)
      MAX_ITERS="$2"
      shift 2
      ;;
    --test-cmd)
      TEST_CMD="$2"
      shift 2
      ;;
    --lint-cmd)
      LINT_CMD="$2"
      shift 2
      ;;
    --typecheck-cmd)
      TYPECHECK_CMD="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --implementer-agent)
      IMPLEMENTER_AGENT="$2"
      shift 2
      ;;
    --reviewer-agent)
      REVIEWER_AGENT="$2"
      shift 2
      ;;
    --commit-message)
      COMMIT_MESSAGE="$2"
      shift 2
      ;;
    --no-auto-commit)
      AUTO_COMMIT=0
      shift 1
      ;;
    --include-task-file)
      INCLUDE_TASK_FILE_IN_COMMIT=1
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      err "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

require_cmd git
require_cmd jq
require_cmd cp
require_cmd copilot
require_cmd python3
require_cmd sed
require_cmd tr
require_cmd grep
require_cmd date
require_cmd wc

if ! [[ "$MAX_ITERS" == <-> ]] || (( MAX_ITERS <= 0 )); then
  err "MAX_ITERS must be a positive integer, got: $MAX_ITERS"
  exit 1
fi

if ! [[ "$UNTRACKED_DIFF_MAX_BYTES" == <-> ]] || (( UNTRACKED_DIFF_MAX_BYTES < 0 )); then
  err "UNTRACKED_DIFF_MAX_BYTES must be a non-negative integer, got: $UNTRACKED_DIFF_MAX_BYTES"
  exit 1
fi

LOG_DIR="${LOG_DIR%/}"
case "$LOG_DIR" in
  ./*)
    LOG_DIR_FILTER="${LOG_DIR#./}"
    ;;
  *)
    LOG_DIR_FILTER="$LOG_DIR"
    ;;
esac

[[ -f "$TASK_FILE" ]] || {
  err "Task file not found: $TASK_FILE"
  exit 1
}

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
  err "Current directory is not a git repository."
  exit 1
}

mkdir -p "$LOG_DIR"

if [[ -z "$COMMIT_MESSAGE" ]]; then
  COMMIT_MESSAGE="$(default_commit_message_from_task)"
fi

REVIEW_JSON=""
PASS_FLAG=0

log "Started at $(timestamp)"
log "Task file: $TASK_FILE"
log "Max iterations: $MAX_ITERS"
log "Log dir: $LOG_DIR"
log "Log dir filter: $LOG_DIR_FILTER"
log "Untracked diff size limit: $UNTRACKED_DIFF_MAX_BYTES bytes"
log "Implementer agent: $IMPLEMENTER_AGENT"
log "Reviewer agent: $REVIEWER_AGENT"
log "Auto commit: $AUTO_COMMIT"
log "Commit message: $COMMIT_MESSAGE"

integer iter
for (( iter = 1; iter <= MAX_ITERS; iter++ )); do
  log "============================================"
  log "Iteration $iter / $MAX_ITERS"

  ITER_DIR="$LOG_DIR/iter-$iter"
  mkdir -p "$ITER_DIR"

  IMPL_PROMPT_FILE="$ITER_DIR/impl_prompt.txt"
  IMPL_OUT_FILE="$ITER_DIR/impl_output.txt"
  CHECKS_FILE="$ITER_DIR/checks.txt"
  DIFF_FILE="$ITER_DIR/diff.txt"
  REVIEW_PROMPT_FILE="$ITER_DIR/review_prompt.txt"
  REVIEW_RAW_FILE="$ITER_DIR/review_raw.txt"
  REVIEW_JSON_FILE="$ITER_DIR/review.json"

  local_impl_prompt=""
  local_review_prompt=""

  if [[ -z "$REVIEW_JSON" ]]; then
    local_impl_prompt="$(build_impl_prompt_initial "$TASK_FILE")"
  else
    print -r -- "$REVIEW_JSON" > "$REVIEW_JSON_FILE"
    local_impl_prompt="$(build_impl_prompt_revision "$TASK_FILE" "$REVIEW_JSON_FILE")"
  fi
  print -r -- "$local_impl_prompt" > "$IMPL_PROMPT_FILE"

  log "Running implementation agent..."
  {
    print -- "### IMPLEMENTER START $(timestamp)"
    copilot \
      --agent "$IMPLEMENTER_AGENT" \
      --autopilot \
      --no-ask-user \
      --allow-tool='write, shell(git:*), shell(cat:*), shell(pytest:*), shell(python:*), shell(uv:*), shell(poetry:*), shell(pip:*), shell(ruff:*), shell(mypy:*), shell(npm:*), shell(pnpm:*), shell(yarn:*), shell(bun:*), shell(go:*), shell(cargo:*), shell(mvn:*), shell(gradle:*), shell(make:*), shell(just:*)' \
      -p "$local_impl_prompt"
    print -- "### IMPLEMENTER END $(timestamp)"
  } > "$IMPL_OUT_FILE" 2>&1 || {
    err "Implementation agent failed in iteration $iter. See: $IMPL_OUT_FILE"
    exit 2
  }

  log "Running local project checks..."
  if run_project_checks "$CHECKS_FILE"; then
    log "Project checks completed."
  else
    warn "Some project checks failed. Review will still inspect current state."
  fi

  log "Capturing git diff..."
  git_diff_file "$DIFF_FILE"

  local_review_prompt="$(build_review_prompt "$TASK_FILE" "$CHECKS_FILE" "$DIFF_FILE")"
  print -r -- "$local_review_prompt" > "$REVIEW_PROMPT_FILE"

  log "Running reviewer agent..."
  {
    copilot \
      --agent "$REVIEWER_AGENT" \
      --no-ask-user \
      --allow-tool='shell(git:*), shell(cat:*), shell(pytest:*), shell(python:*), shell(uv:*), shell(poetry:*), shell(pip:*), shell(ruff:*), shell(mypy:*), shell(npm:*), shell(pnpm:*), shell(yarn:*), shell(bun:*), shell(go:*), shell(cargo:*), shell(mvn:*), shell(gradle:*), shell(make:*), shell(just:*)' \
      -s \
      -p "$local_review_prompt"
  } > "$REVIEW_RAW_FILE" 2>&1 || {
    err "Reviewer agent failed in iteration $iter. See: $REVIEW_RAW_FILE"
    exit 3
  }

  if jq -e . "$REVIEW_RAW_FILE" >/dev/null 2>&1; then
    cp "$REVIEW_RAW_FILE" "$REVIEW_JSON_FILE"
  else
    python3 - "$REVIEW_RAW_FILE" "$REVIEW_JSON_FILE" <<'PY' || {
import json, sys
src, dst = sys.argv[1], sys.argv[2]
text = open(src, "r", encoding="utf-8", errors="ignore").read()

decoder = json.JSONDecoder()
best = None
best_end = -1
for i, ch in enumerate(text):
    if ch != "{":
        continue
    try:
        obj, end = decoder.raw_decode(text[i:])
        if end > best_end:
            best = obj
            best_end = end
    except Exception:
        pass

if best is None:
    raise SystemExit(1)

with open(dst, "w", encoding="utf-8") as f:
    json.dump(best, f, ensure_ascii=False, indent=2)
PY
      err "Failed to extract JSON object from reviewer output."
      err "  raw output: $REVIEW_RAW_FILE"
      exit 4
    }
  fi

  if ! validate_review_json "$REVIEW_JSON_FILE"; then
    err "Reviewer output is not valid schema JSON."
    err "  raw : $REVIEW_RAW_FILE"
    err "  json: $REVIEW_JSON_FILE"
    exit 4
  fi

  REVIEW_JSON="$(cat "$REVIEW_JSON_FILE")"

  log "Review summary:"
  summarize_review "$REVIEW_JSON_FILE"

  STATUS="$(jq -r '.status' "$REVIEW_JSON_FILE")"
  ISSUE_COUNT="$(jq -r '.issues | length' "$REVIEW_JSON_FILE")"

  if [[ "$STATUS" == "PASS" && "$ISSUE_COUNT" -eq 0 ]]; then
    PASS_FLAG=1
    log "Review passed with zero findings."
    break
  fi
done

log "============================================"

if [[ "$PASS_FLAG" -ne 1 ]]; then
  warn "Reached MAX_ITERS without PASS."
  warn "Inspect logs under: $LOG_DIR"
  exit 5
fi

if [[ "$AUTO_COMMIT" -eq 1 ]]; then
  log "Preparing auto-commit..."
  stage_files_for_commit

  if has_staged_changes; then
    show_commit_summary
    perform_auto_commit "$COMMIT_MESSAGE"
    log "Auto-commit completed."
    print -- ""
    print -- "Last commit:"
    git --no-pager log -1 --stat
    exit 0
  else
    warn "Auto-commit skipped because nothing remained to commit."
    exit 0
  fi
else
  log "PASS reached. Auto-commit disabled."
  print -- ""
  print -- "Next suggested commands:"
  print -- "  git add -A"
  print -- "  git commit -m \"$COMMIT_MESSAGE\""
  exit 0
fi
