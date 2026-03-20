#!/usr/bin/env python3
import os
import subprocess
import sys

password = os.environ.get("SSH_PASSWORD")
if not password:
    print("Set SSH_PASSWORD in the environment (never commit it).", file=sys.stderr)
    sys.exit(1)
host = os.environ.get("SSH_HOST", "d3@100.89.206.96")

# Commands to search for FAANTRA
commands = [
    "find ~ -iname '*faantra*' 2>/dev/null",
    "find ~/Desktop ~/Documents ~/Downloads -iname '*faantra*' 2>/dev/null",
    "find ~ -type f \\( -iname '*.safetensors' -o -iname '*.bin' -o -iname '*.pt' -o -iname '*.pth' \\) -path '*faantra*' 2>/dev/null",
    "pwd",
    "ls -la | grep -i faantra || echo 'No FAANTRA in current directory'"
]

print("Searching for FAANTRA model on remote server...")
print("=" * 60)

for cmd in commands:
    print(f"\nRunning: {cmd}")
    print("-" * 60)
    try:
        result = subprocess.run(
            ['sshpass', '-p', password, 'ssh', '-o', 'StrictHostKeyChecking=no', 
             '-o', 'UserKnownHostsFile=/dev/null', host, cmd],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(f"Error: {result.stderr}", file=sys.stderr)
    except subprocess.TimeoutExpired:
        print("Command timed out")
    except Exception as e:
        print(f"Error executing command: {e}", file=sys.stderr)
