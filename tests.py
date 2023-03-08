from train_agent import *
from subprocess import Popen, PIPE

if __name__ == "__main__":
    """Play a number of games with an agent and return the experience."""
    # Environment process
    p = Popen(
        ["build/Debug/train_env.exe"], stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True
    )
    # Separate inputs with newlines
    p.stdin.write("a".encode("utf-8") + b"\n")
    p.stdin.flush()
    line = p.stdout.readline()
    pass
