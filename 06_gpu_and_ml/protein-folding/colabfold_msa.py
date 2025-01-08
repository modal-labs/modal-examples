import modal
from pathlib import Path
from urllib.parse import urljoin, urlparse
import os
import time
import logging as L
from itertools import islice

L.basicConfig(
    level=L.INFO,
    format=f"\033[0;32m%(asctime)s %(levelname)s [%(filename)s.%(funcName)-22s:%(lineno)-3d] %(message)s\033[0m",
    datefmt="%b %d %H:%M:%S",
)

here = Path(__file__).parent  # the directory of this file

MINUTES = 60  # seconds
HOURS = 60 * MINUTES  # seconds

GiB = 1024 # mebibytes

MMSEQS_FORCE_MERGE = "1"
MMSEQS_NO_INDEX = "1" #TODO Plug this in somewhere?
ARIA_NUM_CONNECTIONS = 8

# ColabFold uses this commit (May 28, 2023) to create the databases and perform searches.
mmseqs_commit_id = "71dd32ec43e3ac4dabf111bbc4b124f1c66a85f1"

app_name = "example-compbio-colabfold"
app = modal.App(app_name)

volume = modal.Volume.from_name(
    "example-compbio-colab", create_if_missing=True
)
volume_path = Path("/vol")
s3_bucket_path = Path("/s3")
data_path = volume_path / "data"

colabfold_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "cmake", "zlib1g-dev", "wget", "aria2", "rsync")
    .run_commands(
        "git clone https://github.com/soedinglab/MMseqs2",
        # f"cd MMseqs2 && git checkout {mmseqs_commit_id}",
        "cd MMseqs2 && mkdir build",
        "cd MMseqs2/build && cmake -DCMAKE_BUILD_TYPE=RELEASE -DHAVE_ZLIB=1 -DCMAKE_INSTALL_PREFIX=. ..",
        "cd MMseqs2/build && make -j4",
        "cd MMseqs2/build && make install ", # TODO GPU install
        "ln -s /MMseqs2/build/bin/mmseqs /usr/local/bin/mmseqs",
    )
    .pip_install(
        "colabfold[alphafold-minus-jax]==1.5.5",
        "aria2p==0.12.0",
        "tqdm==4.67.1",
    )
    # FIXME
    .copy_local_file(Path(__file__).parent / "expandaln.cpp", "/expandaln.cpp")
    .copy_local_file(Path(__file__).parent / "data/colabfold_search_default_input.fasta", "/input.fasta")
)

mmcif_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("rsync")
    .pip_install("tqdm==4.67.1")
)

with mmcif_image.imports():
    import shutil

def download_file(url, filename):
    import subprocess
    command = [
       "aria2c",
        "--log-level=warn",
        "-x", str(ARIA_NUM_CONNECTIONS),
        "-o", filename,
        "-c",
        "-d", data_path,
        url
    ]
    subprocess.run(command, check=True)

def extract_with_progress(
    filepath,
    with_pattern="",
    chunk_size=1024 * 1024
):
    import tarfile
    from tqdm import tqdm

    # with tarfile.open(filepath, 'r:*') as tar:
    # mode = "r:*"
    mode = "r|*"
    L.info(f"opening with tarfile mode {mode}")
    with tarfile.open(filepath, mode) as tar:
        L.info(f"opened")
        for member in tar:
            if not member.isfile() or with_pattern not in member.name:
                continue
            member_path = data_path / member.name

            if member_path.exists() and member_path.stat().st_size == member.size:
                L.info(f"already extracted {member.name}, skipping")
                continue

            extract_file = tar.extractfile(member)
            L.info(f"member size: {format_bytes(member.size)}")

            file_progress = tqdm(
                total=member.size, unit='B', desc=member.name, unit_scale=True
            )
            with open(member_path, 'wb') as f:
                while True:
                    chunk = extract_file.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    file_progress.update(len(chunk))

            file_progress.close()


@app.function(
    image=colabfold_image,
    volumes={volume_path: volume},
    memory=2 * GiB,
    timeout=4 * HOURS,
)
def setup_colabfold_db(url: str):
    import subprocess

    filename = urlparse(url).path.split("/")[-1]
    dest_filepath = data_path / filename
    extraction_complete_filepath = (
        dest_filepath.with_suffix("").with_suffix(".complete")
    )

    L.info(f"downloading from {url} to {dest_filepath}")
    download_file(url, filename)

    if not extraction_complete_filepath.exists():
        L.info(f"extracting {dest_filepath}")
        extract_with_progress(dest_filepath)
        extraction_complete_filepath.touch()
    else:
        L.info("extraction already complete, skipping")
    extracted_folder = data_path / Path(Path(filename).stem).stem

    db_foldername = extracted_folder.with_stem(extracted_folder.stem  + "_db")
    L.info(f"converting TSV to MMseqs2 DB: {db_foldername}")
    setup_env = os.environ.copy()
    setup_env["MMSEQS_FORCE_MERGE"] = MMSEQS_FORCE_MERGE
    command = [
        "mmseqs",
        "tsv2exprofiledb",
        extracted_folder,
        db_foldername,
    ]
    subprocess.run(command, check=True, env=setup_env)

@app.function(
    image=colabfold_image,
    volumes={volume_path: volume},
    memory=2 * GiB,
    timeout=4 * HOURS,
)
def setup_hhsuite_data(url: str):
    filename = urlparse(url).path.split("/")[-1]
    dest_filepath = data_path / filename

    L.info(f"downloading from {url} to {dest_filepath}")
    download_file(url, filename)

    L.info(f"extracting {dest_filepath}")
    extract_with_progress(data_path / filename, with_pattern="a3m")

@app.function(
    image=colabfold_image,
    volumes={volume_path: volume},
    memory=2 * GiB,
    timeout=1 * HOURS,
)
def setup_colabfold_fasta_db(url :str):
    import subprocess

    filename = urlparse(url).path.split("/")[-1]
    dest_filepath = data_path / filename

    L.info(f"downloading from {url} to {dest_filepath}")
    download_file(url, filename)

    L.info(f"creating MMseqs2 DB from {dest_filepath}")
    setup_env = os.environ.copy()
    setup_env["MMSEQS_FORCE_MERGE"] = MMSEQS_FORCE_MERGE
    command = [
        "mmseqs",
        "createdb",
        dest_filepath,
        dest_filepath.with_suffix("").with_suffix("")
    ]
    subprocess.run(command, check=True, env=setup_env)


def move_to_modal(mmcif_relative_path, s3_dir_path, modal_dir_path):
    def exists_and_same_size(a, b):
        return a.exists() and b.exists() and a.stat().st_size == b.stat().st_size

    retries = 3
    while retries > 0:
        try:
            source_filepath = s3_dir_path / mmcif_relative_path
            dest_filepath   = modal_dir_path / mmcif_relative_path

            if exists_and_same_size(source_filepath, dest_filepath):
                print(f"exists: {mmcif_relative_path.stem} ", end="")
                return True

            dest_filepath.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_filepath, dest_filepath)
            print(f"d/l {mmcif_relative_path.stem} ", end="")
            return True
        except Exception as e:
            print("")
            L.info(f"Warning: {e}")
            L.info("retrying d/l of {mmcif_relative_path.stem}")
            retries -= 1
    return False

def scan_directory(subdir_path):
    return list([f for f in subdir_path.rglob("*") if f.is_file()])

@app.function(
    image=mmcif_image,
    volumes={
        volume_path: volume,
        s3_bucket_path: modal.CloudBucketMount(
            bucket_name="pdbsnapshots",
            read_only=True
        ),
    },
    timeout=6 * HOURS,
)
def setup_mmcif_database(
    snapshot_id: Path,
    pdb_type: str,
    pdb_port=33444,
    pdb_server="rsync.wwpdb.org::ftp",
    num_workers_per_cpu=8,
):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm
    import subprocess

    assert pdb_type in ("divided", "obsolete")

    max_workers = num_workers_per_cpu * os.cpu_count()

    s3_dir_path = (
        s3_bucket_path / snapshot_id / "pub" / "pdb"
        / "data" / "structures" / pdb_type / "mmCIF"
    )
    modal_dir_path = data_path / "pdb" / pdb_type

    L.info(f"scanning: {s3_dir_path}")
    mmcif_paths = [f for f in s3_dir_path.iterdir() if f.is_file()]
    s3_subdir_paths = [d for d in s3_dir_path.iterdir() if d.is_dir()]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
         futures = [executor.submit(scan_directory, d) for d in s3_subdir_paths]

         for future in tqdm(futures, desc="scanning s3 directories"):
             mmcif_paths.extend(future.result())

    mmcif_relative_paths = [f.relative_to(s3_dir_path) for f in mmcif_paths]

    tasks_args = [
        (f, s3_dir_path, modal_dir_path) for f in mmcif_relative_paths
    ]

    L.info(f"starting process pool with max_workers={max_workers},"
          f" {len(tasks_args)} d/l tasks will be started."
    )
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(move_to_modal, *args) for args in tasks_args]
        results = []
        with tqdm(total=len(tasks_args), desc="D/L S3 PDB files") as pbar:
            for future in as_completed(futures):
                results.append(future)
                pbar.update(1)

    L.info(f"finalizying downloads with: {pdb_server}")
    command = [
        "rsync",
        "-rlpt", "-z",
        "--delete",
        f"--port={pdb_port}",
        f"{pdb_server}/data/structures/{pdb_type}/mmCIF",
        f"{modal_dir_path}"
    ]
    subprocess.run(command, check=True)

@app.function(
    image=colabfold_image,
    volumes={volume_path: volume},
    cpu=32,
    memory=64 * GiB,
    timeout=4 * HOURS,
)
def run_colabfold_search(
    args: str,
    fasta_content :str,
):
    import shlex
    import subprocess
    import tempfile

    fasta_filepath = Path(
        tempfile.NamedTemporaryFile(suffix='.fasta', mode='w', delete=False)
        .name
    )
    fasta_filepath.write_text(fasta_content)

    args = shlex.split(args)

    output_dir = Path("msas") #FIXME
    command = (
        ["colabfold_search"] + args +
        [str(x) for x in (fasta_filepath, data_path, output_dir)]
    )
    print(f"running command: {' '.join(command)}")
    subprocess.run(command, check=True)

    breakpoint()

@app.local_entrypoint()
def main(
    mmseqs_no_index: bool = False, # TODO
    mmseqs_force_merge: bool = False, # TODO
    pdb_aws_snapshot: str = "20240101",
    pdb_server: str = "x", # TODO
    colabfold_search_args: str = None,
    colabfold_search_fasta_filepath: str = None,
):

    if colabfold_search_args is None:
       colabfold_search_args = "--filter 0"

    if colabfold_search_fasta_filepath is None:
        colabfold_search_fasta_filepath = here / "data" / "colabfold_search_default_input.fasta"
    colabfold_search_fasta_content = Path(colabfold_search_fasta_filepath).read_text()

    run_colabfold_search.remote(
        colabfold_search_args,
        colabfold_search_fasta_content,
    )
    return

    # FIXME
    colabfold_url = "https://wwwuser.gwdg.de/~compbiol/colabfold/" # sic
    hhsuite_url = (
        "https://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/"
    )

    # settings = (

    fcs = [
        # Data from colabfold
        setup_colabfold_db.spawn(
            urljoin(colabfold_url, "uniref30_2302.tar.gz"),
            mmseqs_no_index,
            mmseqs_force_merge,
        ),
        setup_colabfold_db.spawn(
            urljoin(colabfold_url, "colabfold_envdb_202108.tar.gz"),
            mmseqs_no_index,
            mmseqs_force_merge,
        ),
        setup_colabfold_fasta_db.spawn(
            urljoin(colabfold_url,  "pdb100_230517.fasta.gz")
        ),

        # Data from HHSuite
        # setup_hhsuite_data.spawn(urljoin(hhsuite_url, "pdb100_foldseek_230517.tar.gz")),

        # MMCIF data
        # setup_mmcif_database.spawn(pdb_aws_snapshot, "divided"),
        # setup_mmcif_database.spawn(pdb_aws_snapshot, "obsolete"),
    ]

    for fc in fcs:
        L.info(fc.get(timeout=2 * HOURS))



def format_bytes(size):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    raise Exception("size too large")
