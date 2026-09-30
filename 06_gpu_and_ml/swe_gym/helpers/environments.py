"""SWE-Gym execution in Modal; hidden tests run in a separate fresh sandbox."""

import contextlib
import json
import shlex
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal
from swegym.harness.constants import APPLY_PATCH_PASS, MAP_REPO_VERSION_TO_SPECS
from swegym.harness.grading import get_eval_report, get_logs_eval
from swegym.harness.test_spec import TestSpec, make_eval_script_list

APP_NAME = "example-swe-gym-sandboxes"


class InfrastructureError(RuntimeError):
    pass


def test_spec(instance):
    inst = dict(instance)
    inst["repo"] = inst["repo"].lower()
    inst["instance_id"] = inst["instance_id"].lower()
    specs = MAP_REPO_VERSION_TO_SPECS[inst["repo"]][inst["version"]]
    commands = make_eval_script_list(
        inst, specs, "testbed", "/testbed", inst["base_commit"], inst["test_patch"]
    )
    return TestSpec(
        inst["instance_id"],
        inst["repo"],
        inst["version"],
        [],
        commands,
        [],
        "x86_64",
        inst["FAIL_TO_PASS"],
        inst["PASS_TO_PASS"],
    )


def command(sb, code, timeout=120):
    p = sb.exec("/bin/bash", "-lc", code, timeout=timeout)
    with ThreadPoolExecutor(2) as pool:
        out, err = pool.submit(p.stdout.read), pool.submit(p.stderr.read)
        stdout, stderr = out.result(), err.result()
    p.wait()
    return p.returncode, stdout + stderr


class Environments:
    def __init__(self):
        self.app = modal.App.lookup(APP_NAME, create_if_missing=True)
        self.images = {}
        self.lock = threading.Lock()

    def image(self, inst):
        key = inst["instance_id"]
        with self.lock:
            if key not in self.images:
                suffix = key.replace("__", "_s_").lower()
                # Record the resolved Modal image ID; upstream only publishes latest tags.
                self.images[key] = modal.Image.from_registry(
                    f"xingyaoww/sweb.eval.x86_64.{suffix}:latest"
                ).entrypoint([])
            return self.images[key]

    @contextlib.contextmanager
    def sandbox(self, inst):
        sb = modal.Sandbox.create(
            "sleep",
            "7200",
            app=self.app,
            image=self.image(inst),
            cpu=2,
            memory=8192,
            timeout=7200,
            block_network=True,
        )
        try:
            rc, output = command(
                sb,
                "cd /testbed && git reset --hard "
                + shlex.quote(inst["base_commit"])
                + " && git clean -fd",
                120,
            )
            if rc:
                raise InfrastructureError("Cannot reset repository: " + output[-2000:])
            yield sb
        finally:
            sb.terminate()

    def prepare_agent(self, sb):
        rc, out = command(
            sb,
            """set -e
rm -f /eval.sh /patch.diff /test.patch /root/patch.diff
cd /testbed
git checkout --detach HEAD
git remote | xargs -r -n1 git remote remove
git for-each-ref --format='%(refname)' | while read r; do git update-ref -d "$r"; done
git reflog expire --expire=now --all
git gc --prune=now
""",
            180,
        )
        if rc:
            raise InfrastructureError("Agent preparation failed: " + out[-2000:])

    def patch(self, sb):
        rc, out = command(
            sb,
            "cd /testbed && git add -A && git diff --cached --binary HEAD -- . ':(exclude)*.pyc' ':(exclude)__pycache__' ':(exclude).pytest_cache'",
            120,
        )
        if rc or len(out) > 2_000_000:
            raise InfrastructureError(
                f"Cannot capture patch: exit={rc}, bytes={len(out)}; {out[-2000:]}"
            )
        return out

    def grade(self, inst, patch, directory):
        directory = Path(directory) / inst["instance_id"].lower()
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "prediction.patch").write_text(patch)
        spec = test_spec(inst)
        with self.sandbox(inst) as sb:
            if patch:
                sb.filesystem.write_text(patch, "/prediction.patch")
                rc, out = command(
                    sb, "cd /testbed && git apply --whitespace=nowarn /prediction.patch"
                )
                if rc:
                    (directory / "apply.log").write_text(out)
                    return {"reward": 0.0, "invalid_patch": True}
            sb.filesystem.write_text(spec.eval_script, "/hidden_eval.sh")
            rc, out = command(sb, "bash /hidden_eval.sh", 1200)
            path = directory / "test_output.txt"
            path.write_text(f"{APPLY_PATCH_PASS} (pred)\n" + out)
            statuses, parsed = get_logs_eval(str(path))
            if not parsed or not statuses:
                raise InfrastructureError(
                    f"No trustworthy test results: {path}; exit={rc}"
                )
            report = get_eval_report(
                spec,
                {"instance_id": spec.instance_id, "model_patch": patch},
                str(path),
                True,
            )[spec.instance_id]
            report["reward"] = float(report["resolved"])
            (directory / "report.json").write_text(json.dumps(report, indent=2))
            return report
