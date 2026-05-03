"""Submit jobs to the running scheduler — like sbatch for our system.

Usage:
    python3 -m scheduler.submit job1.py job2.py    # submit individual files
    python3 -m scheduler.submit eval_data/jobs/    # submit a directory
"""

import json
import socket
import sys
from pathlib import Path

HOST = "localhost"
PORT = 9321


def collect_scripts(paths):
    """Resolve files and directories into .py paths."""
    scripts = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            scripts.extend(sorted(p.rglob("*.py")))
        elif p.is_file() and p.suffix == ".py":
            scripts.append(p)
        else:
            print(f"  skipping {p}")
    return scripts


def submit(scripts, host=HOST, port=PORT):
    """Send script paths to the scheduler server."""
    msg = json.dumps({"scripts": [str(s.resolve()) for s in scripts]})

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((host, port))
    except ConnectionRefusedError:
        print(f"Error: scheduler not running on {host}:{port}")
        print(f"Start it with: python3 -m scheduler.main")
        sys.exit(1)

    sock.sendall(msg.encode())
    response = sock.recv(8192).decode()
    sock.close()

    results = json.loads(response)
    for r in results:
        name = Path(r["path"]).name
        if r["status"] == "queued":
            print(f"  submitted {name} -> queued (k={r['k']:.3f})")
        else:
            print(f"  failed {name}: {r['error']}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 -m scheduler.submit <script.py | directory> ...")
        sys.exit(1)

    scripts = collect_scripts(sys.argv[1:])
    if not scripts:
        print("No .py scripts found.")
        sys.exit(1)

    print(f"Submitting {len(scripts)} job(s)...\n")
    submit(scripts)


if __name__ == "__main__":
    main()
