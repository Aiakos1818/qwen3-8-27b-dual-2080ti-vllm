#!/usr/bin/env python
"""Crash-atomicity check for the SSD session store.

Parent spawns a child that starts a large store and SIGKILLs itself mid-write.
The parent then checks that (a) no half-written *final* file is discoverable
(the in-process index is gone anyway) and (b) a fresh store with
``clean_start=True`` wipes the leftover directory.
"""
import mmap
import os
import signal
import subprocess
import sys
import time

ROOT = "/dev/shm/ssd_crash"
ROW = 1 << 20
SLOTS = 512
ENGINE = "crash"


def child() -> None:
    from vllm.v1.core.host_tier_ssd import HostTierSSDStore

    buf = mmap.mmap(-1, SLOTS * ROW)
    view = memoryview(buf).cast("B", shape=(SLOTS, ROW))
    store = HostTierSSDStore(
        root_dir=ROOT,
        quota_bytes=1 << 40,
        kv_view=view,
        engine_id=ENGINE,
        read_threads=1,
        write_threads=1,
        use_o_direct=False,
        row_bytes=ROW,
    )
    store.submit_store(
        "victim", list(range(SLOTS)), [[b"\x01" * 16]], b"\x02" * 16, 1
    )
    time.sleep(0.03)
    os.kill(os.getpid(), signal.SIGKILL)


def session_dir() -> str:
    engine_dirs = os.listdir(ROOT)
    assert engine_dirs, "engine dir missing"
    return os.path.join(ROOT, engine_dirs[0], "sessions")


def main() -> int:
    import shutil

    shutil.rmtree(ROOT, ignore_errors=True)
    subprocess.run([sys.executable, __file__, "child"], check=False)
    time.sleep(0.1)

    files = []
    for root, _, names in os.walk(session_dir()):
        for n in names:
            files.append(os.path.join(root, n))
    finals = [f for f in files if f.endswith(".bin")]
    temps = [f for f in files if not f.endswith(".bin")]
    print(f"after kill: finals={len(finals)} temps={len(temps)}")
    partial = 0 < len(finals) < SLOTS
    print(f"partial session dir present: {partial}")

    # A fresh store with clean_start wipes whatever the crash left behind.
    from vllm.v1.core.host_tier_ssd import HostTierSSDStore

    buf = mmap.mmap(-1, ROW)
    view = memoryview(buf).cast("B", shape=(1, ROW))
    HostTierSSDStore(
        root_dir=ROOT,
        quota_bytes=1 << 20,
        kv_view=view,
        engine_id=ENGINE,
        use_o_direct=False,
        clean_start=True,
        row_bytes=ROW,
    )
    left = sum(len(n) for _, _, n in os.walk(session_dir()))
    print(f"files after clean_start: {left}")
    ok = partial and left == 0
    print("SSD-CRASH-" + ("OK" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child()
    else:
        sys.exit(main())
