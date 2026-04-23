#!/usr/bin/env bash
# lab_watchdog.sh — DuoNeural lab monitor
# Archon, 2026-04-23 (update: comparison run + Telegram notification on completion)
# Check 1: kilonova — active experiment log, process alive, GPU busy
#           Auto-notify via Telegram when comparison run completes
# Check 2: BitDelta pod (root@213.192.2.107 -p 40019) — bitdelta log + python3

set -uo pipefail

LOGFILE="/home/ai/duoneural/A26B/lab_watchdog.log"
SSH_KEY="$HOME/.ssh/id_ed25519"
SSH_BASE="-i $SSH_KEY -o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=no"
TS=$(date '+%Y-%m-%d %H:%M:%S')

log() { echo "[$TS] $*" | tee -a "$LOGFILE"; }
sep() { echo "────────────────────────────────────────────────────────" >> "$LOGFILE"; }

# Telegram push helper (uses archon_telegram.py --push)
tg_push() {
  /home/ai/duoneural/A26B/archon_env/bin/python3.11 \
    /home/ai/duoneural/A26B/archon_telegram.py --push "$1" 2>/dev/null || true
}

sep
log "=== LAB WATCHDOG CHECK ==="

# ── CHECK 1: kilonova ─────────────────────────────────────────────────────────
log "[kilonova] connecting to ai@10.0.0.190..."

KILO_OUT=$(ssh $SSH_BASE ai@10.0.0.190 "
  GPU=\$(cat /sys/class/drm/card*/device/gpu_busy_percent 2>/dev/null | head -1)
  # auto-select active experiment log — comparison takes priority
  LOG_COMP='/home/ai/duoneural/ctm_baseline_comparison.log'
  LOG_300M='/home/ai/duoneural/ctm_300m_longrun.log'
  if [ -f \"\$LOG_COMP\" ]; then
    CTM_LOG=\"\$LOG_COMP\"
    CTM_PROC_PAT='[c]tm_baseline_comparison'
    CTM_MODE='comparison'
  else
    CTM_LOG=\"\$LOG_300M\"
    CTM_PROC_PAT='[c]tm_300m_longrun'
    CTM_MODE='longrun'
  fi
  CTM_TAIL=\$(tail -5 \"\$CTM_LOG\" 2>/dev/null || echo '(log not found)')
  # bracket trick: first char in [] prevents pgrep matching its own cmdline
  CTM_PROC=\$(pgrep -af \"\$CTM_PROC_PAT\" 2>/dev/null | grep -v 'pgrep' | tr '\n' ' ')
  if [ -n \"\$CTM_PROC\" ]; then
    CTM_PID_STATUS='ALIVE'
  else
    CTM_PID_STATUS='DEAD'
  fi
  echo \"GPU_BUSY=\$GPU\"
  echo \"CTM_PID_STATUS=\$CTM_PID_STATUS\"
  echo \"CTM_MODE=\$CTM_MODE\"
  echo \"CTM_PROCS=\${CTM_PROC:-none}\"
  echo 'LOG_START'
  echo \"\$CTM_TAIL\"
  echo 'LOG_END'
" 2>&1)

if echo "$KILO_OUT" | grep -qiE 'refused|timeout|Permission denied|^ssh:'; then
  log "[kilonova] SSH FAILED: $(echo "$KILO_OUT" | head -3)"
else
  GPU=$(echo "$KILO_OUT"        | grep '^GPU_BUSY='        | cut -d= -f2)
  PID_STATUS=$(echo "$KILO_OUT" | grep '^CTM_PID_STATUS='  | cut -d= -f2)
  CTM_MODE=$(echo "$KILO_OUT"   | grep '^CTM_MODE='        | cut -d= -f2)
  CTM_PROCS=$(echo "$KILO_OUT"  | grep '^CTM_PROCS='       | cut -d= -f2-)
  LOG_TAIL=$(echo "$KILO_OUT"   | sed -n '/^LOG_START$/,/^LOG_END$/p' | grep -v '^LOG_\(START\|END\)$')

  log "[kilonova] GPU busy: ${GPU:-unknown}%"
  log "[kilonova] CTM process: ${PID_STATUS:-unknown} (mode: ${CTM_MODE:-unknown})"
  [ "${CTM_PROCS:-none}" != "none" ] && log "[kilonova] CTM processes: $CTM_PROCS"

  log "[kilonova] --- log tail ---"
  echo "$LOG_TAIL" | grep -v 'UserWarning\|attn_output\|experimental\|Triggered internally\|enable it with' | while IFS= read -r line; do
    [ -n "$line" ] && log "[kilonova]   $line"
  done

  # ── COMPARISON COMPLETE: notify + log result ──────────────────────────────
  # Fires once when ctm_baseline_comparison finishes (sentinel in /tmp)
  COMP_SENTINEL="/tmp/ctm_comparison_done"
  if [ "${PID_STATUS:-ALIVE}" = "DEAD" ] \
    && [ "${CTM_MODE:-longrun}" = "comparison" ] \
    && echo "$LOG_TAIL" | grep -qiE 'HYPOTHESIS CONFIRMED|HYPOTHESIS REJECTED|INCONCLUSIVE|experiment complete' \
    && [ ! -f "$COMP_SENTINEL" ]; then
    touch "$COMP_SENTINEL"
    # Extract result line
    RESULT=$(echo "$LOG_TAIL" | grep -iE 'HYPOTHESIS|INCONCLUSIVE|gap|best_ppl' | tail -3 | tr '\n' ' ')
    log "[kilonova] ╔══════════════════════════════════════════╗"
    log "[kilonova] ║  CTM BASELINE COMPARISON COMPLETE        ║"
    log "[kilonova] ║  $RESULT"
    log "[kilonova] ╚══════════════════════════════════════════╝"
    tg_push "🧪 DuoNeural Lab: CTM Baseline Comparison COMPLETE — $RESULT"
  fi

  # ── 300M LONGRUN VIZ (legacy — fires only once if longrun just finished) ──
  # Note: this ran once and failed (wrong script). sentinel prevents re-fire.
  LONGRUN_SENTINEL="/tmp/ctm_300m_viz_triggered"
  if [ "${PID_STATUS:-ALIVE}" = "DEAD" ] \
    && [ "${CTM_MODE:-comparison}" = "longrun" ] \
    && echo "$LOG_TAIL" | grep -qiE 'experiment complete|LONG-RUN COMPLETE' \
    && [ ! -f "$LONGRUN_SENTINEL" ]; then
    touch "$LONGRUN_SENTINEL"
    log "[kilonova] *** 300M longrun complete — sentinel set, no auto-viz (use nano_ctm_viz.py manually) ***"
    tg_push "📊 DuoNeural Lab: CTM 300M longrun finished — check lab for results"
  fi
fi

# ── CHECK 2: BitDelta pod ─────────────────────────────────────────────────────
log "[bitdelta] connecting to root@213.192.2.107 -p 40019..."

BD_OUT=$(ssh $SSH_BASE -p 40019 root@213.192.2.107 "
  BD_LOG='/workspace/bitdelta_fixed.log'
  BD_TAIL=\$(tail -5 \"\$BD_LOG\" 2>/dev/null || echo '(log not found)')
  BD_PROC=\$(pgrep -a python3 2>/dev/null | grep -v 'pgrep' | head -3 || echo 'none')
  echo 'BD_LOG_START'
  echo \"\$BD_TAIL\"
  echo 'BD_LOG_END'
  echo 'BD_PROC_START'
  echo \"\$BD_PROC\"
  echo 'BD_PROC_END'
" 2>&1)

if echo "$BD_OUT" | grep -qiE 'refused|timeout|Permission denied|^ssh:|No route'; then
  log "[bitdelta] SSH FAILED: $(echo "$BD_OUT" | head -3)"
else
  BD_PROCS=$(echo "$BD_OUT" | sed -n '/^BD_PROC_START$/,/^BD_PROC_END$/p' | grep -v '^BD_PROC_\(START\|END\)$')
  if [ -z "$BD_PROCS" ] || echo "$BD_PROCS" | grep -q '^none$'; then
    log "[bitdelta] python3: NOT RUNNING"
  else
    log "[bitdelta] python3 processes alive:"
    echo "$BD_PROCS" | while IFS= read -r line; do
      [ -n "$line" ] && log "[bitdelta]   $line"
    done
  fi

  log "[bitdelta] --- bitdelta_fixed.log tail ---"
  echo "$BD_OUT" | sed -n '/^BD_LOG_START$/,/^BD_LOG_END$/p' | grep -v '^BD_LOG_\(START\|END\)$' | while IFS= read -r line; do
    [ -n "$line" ] && log "[bitdelta]   $line"
  done
fi

log "=== WATCHDOG CHECK COMPLETE ==="
sep
