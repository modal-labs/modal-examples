import pathlib
import subprocess
import tarfile
import modal
from multiprocessing import Process

# TODO: use cloudbucketmount
volume = modal.Volume.from_name("RoseTTAFold", create_if_missing=True)
image = modal.Image.debian_slim().apt_install("wget")
app = modal.App("example-rosettafold-dataset-import", image=image)


def decompress_tar_gz(file_path: pathlib.Path, extract_dir: pathlib.Path) -> None:
    print(f"Decompressing {file_path} into {extract_dir}...")
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
        print(f"Decompressed {file_path} to {extract_dir}")


@app.function(
    volumes={"/vol": volume},
    timeout=60 * 60 * 2,  # 2 hours
    _allow_background_volume_commits=True,
)
def import_transform_load() -> None:
    uniref30 = pathlib.Path("/tmp/UniRef30_2020_06_hhsuite.tar.gz")
    bfd_dataset = pathlib.Path("/tmp/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz")
    structure_templates = pathlib.Path("/tmp/pdb100_2021Mar03.tar.gz")
    commands = []
    if not uniref30.exists():
        print("Downloading uniref30 [46G]")
        commands.append(f"wget http://wwwuser.gwdg.de/~compbiol/uniclust/2020_06/UniRef30_2020_06_hhsuite.tar.gz -O {uniref30}")

    if not bfd_dataset.exists():
        print("Downloading BFD [272G]")
        commands.append(f"wget https://bfd.mmseqs.com/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz -O {bfd_dataset}")

    if not structure_templates.exists():
        print("Downloading structure templates (including *_a3m.ffdata, *_a3m.ffindex)")
        commands.append(f"wget https://files.ipd.uw.edu/pub/RoseTTAFold/pdb100_2021Mar03.tar.gz -O {structure_templates}")

    # Start all downloads in parallel
    processes = [subprocess.Popen(cmd, shell=True) for cmd in commands]

    # Wait for all downloads to complete
    for p in processes:
        p.wait()
        volume.commit()

    # Check if all downloads were successful
    errors = []
    for p in processes:
        if p.returncode == 0:
            print("Download completed successfully.")
        else:
            errors.append(f"Error in downloading. {p.args} failed {p.returncode=}")
    if errors:
        raise RuntimeError(errors)

    uniref30_decompressed = pathlib.Path("/vol/rosettafold/UniRef30_2020_06_hhsuite")
    bfd_dataset_decompressed = pathlib.Path("/vol/rosettafold/bfd_metaclust_clu_complete_id30_c90_final_seq")
    structure_templates_decompressed = pathlib.Path("/vol/rosettafold/pdb100_2021Mar03/")
    decompression_jobs = {
        (uniref30, uniref30_decompressed),
        (bfd_dataset, bfd_dataset_decompressed),
        (structure_templates, structure_templates_decompressed),
    }
    processes = []
    # Create and start a process for each decompression task
    for file_path, extract_dir in decompression_jobs:
        process = Process(target=decompress_tar_gz, args=(file_path, extract_dir))
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()

    print("All decompression tasks completed.")
