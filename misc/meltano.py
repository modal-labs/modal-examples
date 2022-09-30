import subprocess

import modal
from .meltano_project.utils import download_meltano_db, upload_meltano_db

stub = modal.Stub(
    image=modal.Image.from_dockerhub(tag="meltano/meltano")
    # Pre-install extractors and loaders in the image.
    .dockerfile_commands(
        """
            RUN /venv/bin/meltano init meltano_project
            WORKDIR /meltano_project
            RUN /venv/bin/meltano add extractor tap-github
            RUN /venv/bin/meltano add loader target-jsonl --variant andyh1203
            RUN /venv/bin/meltano install
            RUN pip install boto3
        """
    ),
    # Create a mount that contains `meltano.yml`, so that the local copy is synced inside the container.
    mounts=[
        modal.Mount(
            local_file="examples/misc/meltano_project/meltano.yml",
            remote_dir="/meltano_project",
        )
    ],
)


# For this example to work, the secret provides a valid Github personal access token
# under the key `TAP_GITHUB_ACCESS_TOKEN`. You may provide any other plugin configs that you wish via modal's Secrets.
@stub.function(
    secrets=[modal.ref("meltano-secrets"), modal.ref("meltano-s3-credentials")],
)
def run():
    download_meltano_db()

    try:
        subprocess.run(
            ["/venv/bin/meltano", "elt", "tap-github", "target-jsonl", "--job_id=test"],
            cwd="/meltano_project",
        )
    finally:
        upload_meltano_db()


if __name__ == "__main__":
    # Uncomment this line if you want to open a shell into the image.
    # This is useful for interfacing with the Meltano CLI to debug and/or experiment
    # but we recommend making persistent changes by editing the provided `meltano.yml` file.
    # e.g. to list all the entities for our tap:
    # $ cd /meltano_project
    # $ meltano select tap-github --list --all
    # and then add your selected entities to `meltano.yml`.

    # stub.interactive_shell(secrets=[modal.ref("meltano-secrets")])

    with stub.run():
        run()
