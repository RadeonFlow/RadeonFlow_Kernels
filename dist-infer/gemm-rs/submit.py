#!/usr/bin/env python3
import sys
import os
import subprocess
import datetime
import time

def main():
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M:%S-%f")
    logfile = f"logs/gemm-rs-{timestamp}.log"

    os.makedirs("logs", exist_ok=True)

    print(f"submiting, log file: {logfile}")

    cmd = [
        "popcorn-cli", "submit",
        "--gpu", "MI300x8",
        "--leaderboard", "amd-gemm-rs",
        "--mode", "benchmark",
        "submission.py",
        "-o", logfile,
    ]

    start = time.time()

    timeout = 180
    try:
        # Use default stdio; remove invalid stdout=subprocess.STDOUT
        subprocess.run(cmd, timeout=timeout, check=True)
    except subprocess.TimeoutExpired:
        print(f"Error: Command timed out after {timeout}s", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Command not found: {cmd[0]}", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)

    with open(logfile) as f:
        print(f.read())

    end = time.time()
    print(f"submit done, time cost: {end - start:.2f}s")

if __name__ == "__main__":
    main()

