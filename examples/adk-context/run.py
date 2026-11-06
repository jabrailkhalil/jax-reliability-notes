"""Run the same public-API lab against two immutable ADK Go revisions."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from check_results import check


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--race", action="store_true", help="enable Go's race detector")
    args = parser.parse_args()
    for tool in ("git", "go", "gofmt"):
        if shutil.which(tool) is None:
            parser.error(f"{tool} must be installed; CI uses Go 1.26.10")

    root = Path(__file__).resolve().parents[2]
    lab = root / ".lab"
    source = lab / "adk-go"
    lab.mkdir(exist_ok=True)
    env = os.environ | {"GOMAXPROCS": "2", "OTEL_SDK_DISABLED": "true"}

    def run(command: list[str], cwd: Path = root) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command, cwd=cwd, env=env, check=True, capture_output=True,
            text=True, encoding="utf-8", timeout=900,
        )

    program = root / "examples/adk-context/main.go"
    assert not run(["gofmt", "-l", str(program)]).stdout.strip(), "Go source needs formatting"
    if source.exists():
        assert (source / ".git").is_dir(), "existing lab path is not a Git checkout"
        assert not run(["git", "status", "--porcelain"], source).stdout.strip(), "lab checkout is dirty"
    else:
        run(["git", "init", str(source)])

    revisions = {
        "baseline": "a1603c7f179af285b08ef4f03e757a6deb0630b9",
        "fixed": "6caa3b853ea4d179124f089daf6f39498dc93425",
    }
    for revision in revisions.values():
        run(["git", "fetch", "--depth=1", "https://github.com/google/adk-go.git", revision], source)

    binary = lab / "adk-context-lab.exe"
    for mode, revision in revisions.items():
        run(["git", "checkout", "--detach", revision], source)
        assert run(["git", "rev-parse", "HEAD"], source).stdout.strip() == revision
        run(["go", "vet", "-p=2", "-mod=readonly", str(program)], source)
        run(["go", "build", "-p=2", *(["-race"] if args.race else []),
             "-mod=readonly", "-o", str(binary), str(program)], source)
        correct = run([str(binary), "--expect", mode])
        (lab / f"adk-{mode}.json").write_text(correct.stdout, encoding="utf-8")
        wrong = subprocess.run(
            [str(binary), "--expect", "fixed" if mode == "baseline" else "baseline"],
            cwd=root, env=env, capture_output=True, text=True, encoding="utf-8", timeout=120,
        )
        assert wrong.returncode == 1, "opposite revision expectation must fail"
        (lab / f"adk-{mode}-wrong-expectation.json").write_text(wrong.stdout, encoding="utf-8")
        invalid = subprocess.run(
            [str(binary), "--expect", "invalid"], cwd=root, env=env,
            capture_output=True, text=True, encoding="utf-8", timeout=120,
        )
        assert invalid.returncode == 2 and not invalid.stdout, "invalid mode must fail before the lab"
        run(["git", "diff", "--exit-code", "--", "go.mod", "go.sum"], source)
        print(json.dumps({"revision": revision, "expected": mode, "race": args.race}))
    check(lab)


if __name__ == "__main__":
    main()
