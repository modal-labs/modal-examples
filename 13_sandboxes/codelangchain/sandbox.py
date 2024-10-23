"""Defines the logic for running agent code in a sandbox."""

import modal
from common import COLOR, agent_image, app


def run(code: str):
    print(
        f"{COLOR['HEADER']}📦: Running in sandbox{COLOR['ENDC']}",
        f"{COLOR['GREEN']}{code}{COLOR['ENDC']}",
        sep="\n",
    )
    sb = modal.Sandbox.create(
        "python",
        "-c",
        code,
        image=agent_image,
        timeout=60 * 10,  # 10 minutes
        secrets=[
            modal.Secret.from_name(
                "openai-secret"
            )  # could be a different secret!
        ],
        app=app,
    )

    sb.wait()

    if sb.returncode != 0:
        print(
            f"{COLOR['HEADER']}📦: Failed with exitcode {sb.returncode}{COLOR['ENDC']}"
        )

    return sb
