#!/usr/bin/env python3
import sys
import os
import subprocess
import datetime
import time
import re
import math

def gen_submission():
    with open("template.py") as f:
        template = f.read()
    with open("all2all.cpp") as f:
        code = f.read()
    ret = template.replace("{{}}", code)
    ret = ret.replace("\\", "@")
    return ret

def extract_and_geom_mean(text: str):
    pattern = re.compile(r'(\d+(?:\.\d+)?)\s*±\s*\d+(?:\.\d+)?')
    values = [float(m.group(1)) for m in pattern.finditer(text)]
    
    if not values:
        return None, []
    
    log_sum = sum(math.log(v) for v in values)
    geom_mean = math.exp(log_sum / len(values))
    return geom_mean, values

def main():
    if len(sys.argv) > 1 and sys.argv[1] != "local_test":
        pyfile = sys.argv[1]
    else:
        pyfile = "submission.py"
        code = gen_submission()
        if "local_test" not in sys.argv:
            code = code.replace("#define LOCAL_TEST", "")
        with open(pyfile, "w") as f:
            f.write(code)
        if "local_test" in sys.argv:
            return

    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M:%S-%f")
    logfile = f"logs/all2all-{timestamp}.log"

    os.makedirs("logs", exist_ok=True)

    print(f"submiting, log file: {logfile}")

    cmd = [
        "popcorn-cli", "submit",
        "--gpu", "MI300x8",
        "--leaderboard", "amd-all2all",
        "--mode", "benchmark",
        pyfile,
        "-o", logfile,
    ]

    start = time.time()

    timeout = 180
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout, check=True)
    except subprocess.TimeoutExpired:
        print(f"Error: Command timed out after {timeout}s", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)

    with open(logfile) as f:
        output = f.read()
    geom_mean, values = extract_and_geom_mean(output)
    print(output)

    extra_log = "\n"
    if geom_mean:
        extra_log += f"geom_mean: {geom_mean:.2f}, values: {values}\n"

    print(extra_log, end="")
    with open(logfile, "a") as f:
        f.write(extra_log)

    end = time.time()
    print(f"submit done, time cost: {end - start:.2f}s")

if __name__ == "__main__":
    main()

