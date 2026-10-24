"""Kaggle script kernel: the 100-scenario evaluation with 50 gemma3:27b memos on GPU T4 x2.

Push (from the repo root; the branch must be on GitHub first):
    kaggle kernels push -p scripts/kaggle
Follow and download:
    kaggle kernels status aditya32211/kpi-rca-gemma3-27b-eval
    kaggle kernels output aditya32211/kpi-rca-gemma3-27b-eval -p outputs/kaggle_27b
Another member: change "id" in kernel-metadata.json to your own Kaggle username.

The kernel clones the public repo, generates synthetic scenarios and calls only a local
Ollama, so it needs no key and sends no data anywhere. Results land in /kaggle/working.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import urllib.request

REPO = "https://github.com/nishida09datkar/DATA-SCIENCE-CP.git"
BRANCH = "ticket-reporting"
MODEL = "gemma3:27b"
SRC = "/tmp/repo"
WORK = "/kaggle/working"
OLLAMA = "http://127.0.0.1:11434"


def sh(cmd: str, cwd: str | None = None, env: dict | None = None) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd, env=env)


def wait_for_ollama(timeout_s: float = 120) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{OLLAMA}/api/version", timeout=5)
            return
        except OSError:
            time.sleep(2)
    raise RuntimeError("ollama serve did not start")


sh("nvidia-smi")
sh(f"git clone --depth 1 --branch {BRANCH} {REPO} {SRC}")
sh("git log -1 --oneline", cwd=SRC)

# Project: Python 3.12 and the locked dependencies (on Linux, PyPI torch is the CUDA build).
sh("curl -LsSf https://astral.sh/uv/install.sh | sh")
uv = shutil.which("uv") or os.path.expanduser("~/.local/bin/uv")  # root installs to /usr/local/bin
sh(f"{uv} sync --locked --extra dev --extra embeddings", cwd=SRC)
python = f"{SRC}/.venv/bin/python"

# Ollama: the installer needs zstd for the archive and lspci/lshw to see the GPUs.
sh("apt-get -qq update && apt-get -qq install -y zstd pciutils lshw > /dev/null")
sh("curl -fsSL https://ollama.com/install.sh | sh")
log = open(f"{WORK}/ollama.log", "w")
server = subprocess.Popen(["ollama", "serve"], stdout=log, stderr=subprocess.STDOUT)
wait_for_ollama()
sh(f"ollama pull {MODEL} > /dev/null 2>&1 && ollama list")
sh(f"ollama run {MODEL} 'Reply with OK.'")  # loads the model once so the log shows the GPU split
sh(f"ollama ps | tee {WORK}/ollama_ps.txt")

# MiniLM on the GPU if this torch build runs on a T4 (sm_75); otherwise hide the GPUs from Python only.
env = dict(os.environ)
probe = subprocess.run([python, "-c", "import torch; torch.zeros(1).cuda()"], capture_output=True)
if probe.returncode != 0:
    print("torch cannot use the T4; embeddings run on the CPU", flush=True)
    env["CUDA_VISIBLE_DEVICES"] = ""

start = time.time()
sh(
    f"{python} -m kpi_rca -v evaluate --config config/kaggle_27b.yaml --scenarios 100 --with-llm --out {WORK}/outputs",
    cwd=SRC,
    env=env,
)
print(f"evaluate took {time.time() - start:.0f} s", flush=True)
sh(f"ollama ps | tee -a {WORK}/ollama_ps.txt")
sh(f"nvidia-smi | tee {WORK}/nvidia_smi.txt")

server.terminate()
log.close()
shutil.make_archive(f"{WORK}/evaluation", "zip", f"{WORK}/outputs/evaluation")
