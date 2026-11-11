import subprocess

subprocess.run(
    ["python", "../md.py"],
    check=True
)

print("finished!")