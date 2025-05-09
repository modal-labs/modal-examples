import modal
from pathlib import Path

# ## NuScenes Dataset Setup
# NuScenes is a large, complex dataset used for several autonomous vehicle tasks.

# To download NuScenes you need to make an account here: https://www.nuscenes.org/sign-up?prevpath=nuscenes&prevhash=download
# Then upload your username and password as a [Modal Secret](https://modal.com/secrets/). Click the Custom secret
# button, name the secret `nuscenes`, then enter two key/value pairs: [NUSCENES_USERNAME, NUSCENES_PASSWORD].
# Then the following line will import these values as environment variables:
nuscenes_secret = modal.Secret.from_name("nuscenes")

# We will download the data into a Modal.Volume with this name:
vol_name = "example-nuscenes"
# This is the location within the container where this Volume will be mounted:
vol_mnt = Path("/data")
vol_data_dir = "nuscenes"  # data subdir within the volume
# Finally, the Volume object can be created:
nuscenes_volume = modal.Volume.from_name(vol_name, create_if_missing=True)

# ### Define the image
nuscenes_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["git", "python3-opencv"])  # "libgl1-mesa-glx"])
    .run_commands("pip install --upgrade pip")
    .pip_install(["nuscenes-devkit", "tqdm", "requests", "numpy<2"])
    .run_commands(f"git clone https://github.com/beijbom/Sparse4D.git")
)

# Initialize the app
app = modal.App(
    "train-nuscenes",
    image=nuscenes_image,
    volumes={vol_mnt: nuscenes_volume},
)


@app.function(
    image=nuscenes_image, secrets=[nuscenes_secret], volumes={vol_mnt: nuscenes_volume}
)
def download_nuscenes(
    volume_subdir: str,
    region: str = "us",  # or "asia"
):
    """
    Inspired by:
    https://github.com/li-xl/nuscenes-download/blob/master/download_nuscenes.py
    """
    import sys

    import json
    import os
    import subprocess
    import sys
    import tarfile

    import requests
    from tqdm import tqdm

    # ───────────────────────────────────────────────────────────────
    # CONFIG: read these from env
    USEREMAIL = os.getenv("NUSCENES_USERNAME")
    PASSWORD = os.getenv("NUSCENES_PASSWORD")
    OUTPUT_DIR = vol_mnt / volume_subdir
    MINI_ROOT = OUTPUT_DIR / "v1.0-mini"
    PICKLE_FILE = OUTPUT_DIR / "nuscenes_infos_mini.pkl"
    # ───────────────────────────────────────────────────────────────

    # Skip if already downloaded and extracted
    if not MINI_ROOT.exists():
        if not USEREMAIL or not PASSWORD:
            print(
                "Error: set NUSCENES_USERNAME and NUSCENES_PASSWORD in your env",
                file=sys.stderr,
            )
            sys.exit(1)

        # 1) Log in via Cognito to get a bearer token
        print(f"Setting up Nuscenes mini dataset in `{vol_name}/{volume_subdir}`.")
        resp = requests.post(
            "https://cognito-idp.us-east-1.amazonaws.com/",
            headers={
                "Content-Type": "application/x-amz-json-1.1",
                "X-Amz-Target": "AWSCognitoIdentityProviderService.InitiateAuth",
            },
            data=json.dumps(
                {
                    "AuthFlow": "USER_PASSWORD_AUTH",
                    "ClientId": "7fq5jvs5ffs1c50hd3toobb3b9",
                    "AuthParameters": {"USERNAME": USEREMAIL, "PASSWORD": PASSWORD},
                }
            ),
        )
        resp.raise_for_status()
        token = resp.json()["AuthenticationResult"]["IdToken"]
        print("\t✅ Logged in successfully")

        # 2) Fetch the mini archive URL
        api = (
            f"https://o9k5xn5546.execute-api.us-east-1.amazonaws.com/v1"
            f"/archives/v1.0/v1.0-mini.tgz?region={region}&project=nuScenes"
        )
        resp = requests.get(api, headers={"Authorization": f"Bearer {token}"})
        resp.raise_for_status()
        download_url = resp.json()["url"]
        print("\t✅ Got download URL for v1.0-mini.tgz")

        # 3) Download into OUTPUT_DIR
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        archive_path = os.path.join(OUTPUT_DIR, "v1.0-mini.tgz")

        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            with (
                open(archive_path, "wb") as f,
                tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as bar,
            ):
                for chunk in r.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    f.write(chunk)
                    bar.update(len(chunk))
        print(f"\t✅ Downloaded to {archive_path}")

        # 4) Extract the archive in place
        print(f"\t📦 Extracting to {OUTPUT_DIR}")
        with tarfile.open(archive_path, "r:gz") as tar:
            members = tar.getmembers()
            for member in tqdm(members, desc="Extracting", unit="file"):
                tar.extract(member, path=OUTPUT_DIR)
        print("\t✅ Extraction complete")
    else:
        print(f"\tnuScenes mini already present at {MINI_ROOT}, skipping download.")

    # 5) Generate metadata with nuScenes devkit
    if PICKLE_FILE.is_file():
        print("\tpickle file found!")
    else:
        print("\t🔧 Generating devkit metadata pickles...")
        script_path = Path("/Sparse4D") / "tools/nuscenes_converter.py"
        # invoke the nuScenes setup script with our arguments
        subprocess.run(
            [
                sys.executable,
                script_path.as_posix(),
                "--root-path",
                str(OUTPUT_DIR),
                "--version",
                "v1.0-mini",
            ],
            check=True,
        )
    print("\t🎉 nuScenes v1.0-mini is ready in", MINI_ROOT)


@app.local_entrypoint()
def main():
    download_nuscenes.remote(vol_data_dir)
