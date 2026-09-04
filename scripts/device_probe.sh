#!/data/data/com.termux/files/usr/bin/bash
# Phase 3.5a device feasibility probe. Run INSIDE Termux on the target phone.
#
#   bash device_probe.sh 2>&1 | tee /sdcard/device_probe.log
#
# Diagnostic only. Captures no trace, writes no controller code, changes no
# constant. It answers seven questions with evidence, and two of them can sink a
# phase rather than delay it (Q1 hnswlib, Q2 battery current) -- those are marked
# CRITICAL in the output so a skim cannot miss them.
#
# Every check prints the raw command output. "Probably works" is not an answer
# this script is capable of giving.

OUT=/sdcard/device_probe.log
JSON=/sdcard/device_feasibility_raw.json
say() { printf '\n=== %s ===\n' "$*"; }

say "0. DEVICE IDENTITY"
getprop ro.product.manufacturer 2>&1
getprop ro.product.model 2>&1
getprop ro.build.version.release 2>&1
getprop ro.board.platform 2>&1
uname -a 2>&1
echo "termux packages: $(pkg list-installed 2>/dev/null | wc -l)"

say "1. CRITICAL -- hnswlib build from source"
# Upstream publishes no aarch64-android wheels, and there is no chroma-hnswlib
# fallback on this platform the way there was on Windows. It builds or it does
# not.
pkg install -y clang python git >/dev/null 2>&1
echo "clang: $(clang --version 2>&1 | head -1)"
echo "python: $(python --version 2>&1)"
pip install pybind11 2>&1 | tail -2
rm -rf ~/hnswlib_probe && git clone --depth 1 https://github.com/nmslib/hnswlib ~/hnswlib_probe 2>&1 | tail -2
cd ~/hnswlib_probe || exit 1
pip install . -v > /sdcard/hnswlib_build.log 2>&1
echo "build exit code: $?"
echo "--- last 25 lines of build log (the actual compiler error, if any) ---"
tail -25 /sdcard/hnswlib_build.log
echo "--- import check ---"
python -c "import hnswlib; print('hnswlib import OK')" 2>&1
python - <<'PY' 2>&1
try:
    import hnswlib, numpy as np
    i = hnswlib.Index(space="l2", dim=8)
    i.init_index(max_elements=100, M=8, ef_construction=32)
    i.add_items(np.random.rand(100, 8).astype("float32"), np.arange(100))
    i.set_ef(16)
    lab, dist = i.knn_query(np.random.rand(1, 8).astype("float32"), k=5)
    print("hnswlib functional: returned", lab.shape, dist.dtype)
except Exception as e:
    print("hnswlib FUNCTIONAL CHECK FAILED:", type(e).__name__, e)
PY
cd ~ || exit 1

say "2. CRITICAL -- battery current without root"
# If current_now is denied AND Termux:API's current is unavailable or does not
# move under load, there is no measured energy anywhere in this project.
for f in current_now current_avg voltage_now capacity status; do
    printf 'power_supply/battery/%s: ' "$f"
    cat "/sys/class/power_supply/battery/$f" 2>&1
done
echo "--- other power_supply nodes ---"
ls /sys/class/power_supply/ 2>&1
echo "--- termux-battery-status (needs Termux:API app + termux-api pkg) ---"
termux-battery-status 2>&1
echo "--- does current MOVE under load? two reads, 10 s of CPU between them ---"
B1=$(termux-battery-status 2>/dev/null | grep -i current)
echo "before: ${B1:-unavailable}"
timeout 10 python -c "
while True:
    _ = sum(i * i for i in range(100000))
" 2>/dev/null
B2=$(termux-battery-status 2>/dev/null | grep -i current)
echo "after:  ${B2:-unavailable}"
echo "(identical or zero on both reads => not a usable energy source)"

say "3. CPU CLUSTERS -- freq_ratio was specified as one scalar"
echo "logical cpus: $(nproc 2>&1)"
for i in $(seq 0 $(($(nproc 2>/dev/null || echo 1) - 1))); do
    echo "cpu$i max_freq: $(cat /sys/devices/system/cpu/cpu$i/cpufreq/cpuinfo_max_freq 2>&1)"
done
echo "--- distinct max frequencies = number of clusters ---"
for i in $(seq 0 $(($(nproc 2>/dev/null || echo 1) - 1))); do
    cat "/sys/devices/system/cpu/cpu$i/cpufreq/cpuinfo_max_freq" 2>/dev/null
done | sort -u
echo "--- which cpus is this process allowed on ---"
grep Cpus_allowed_list /proc/self/status 2>&1
echo "--- which cpu does a busy python thread actually land on ---"
python - <<'PY' 2>&1
import os, time
t0 = time.time()
while time.time() - t0 < 3:
    _ = sum(i * i for i in range(50000))
try:
    with open("/proc/self/stat") as f:
        print("last cpu executed on (field 39):", f.read().split()[38])
except Exception as e:
    print("could not read /proc/self/stat:", e)
PY

say "4. THERMAL ZONES"
ls /sys/class/thermal/ 2>&1
for z in /sys/class/thermal/thermal_zone*/; do
    [ -d "$z" ] || continue
    echo "$z type=$(cat "${z}type" 2>&1) temp=$(cat "${z}temp" 2>&1)"
done
echo "(zone naming is device-specific; which one is the SoC package is an"
echo " AMBIGUITY to record, not a mapping to guess)"

say "5. PROCESS SURVIVAL -- 2 min smoke test, not the real 30"
echo "--- termux-wake-lock (does it need Termux:API?) ---"
termux-wake-lock 2>&1 && echo "wake-lock acquired" || echo "wake-lock FAILED"
python -c "
import time
t0 = time.time()
while time.time() - t0 < 120:
    _ = sum(i * i for i in range(100000))
" &
PID=$!
echo "started pid $PID"
for t in 30 60 90 120; do
    sleep 30
    if kill -0 $PID 2>/dev/null; then
        echo "t=${t}s: alive"
    else
        echo "t=${t}s: KILLED -- process did not survive"
        break
    fi
done
wait $PID 2>/dev/null
termux-wake-unlock 2>&1

say "7. EXPERIMENTAL CONTROLS -- record, do not enforce"
# Phase 3.5's 30-minute trace and Phase 9's validation are only comparable to
# each other if these match across runs. A phone on a bed throttles differently
# from one on a desk.
termux-battery-status 2>&1 | grep -Ei 'percentage|status|plugged|temperature'
echo "ambient temperature: NOT CONTROLLED (record manually if known)"
echo "physical setup: RECORD MANUALLY -- on a hard surface / in a case / in hand"
echo "charging during probe: see 'plugged'/'status' above"

say "PROBE COMPLETE"
echo "logs: $OUT and /sdcard/hnswlib_build.log"
echo "Transfer both off the device and paste the output back."
