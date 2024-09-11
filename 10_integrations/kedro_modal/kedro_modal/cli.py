import click
import modal

from .modal_functions import main_app, sync_app


@click.group(name="Kedro-Modal")
def commands():
    """Kedro plugin for running kedro pipelines on Modal"""
    pass


@commands.group(
    name="modal", context_settings=dict(help_option_names=["-h", "--help"])
)
def modal_group():
    """Interact with Kedro pipelines run on Modal"""


@modal_group.command(help="Run kedro project on Modal")
@click.pass_obj
def run(metadata):
    app, remote_project_mount_path, remote_data_path = main_app(
        metadata.project_path, metadata.project_name, metadata.package_name
    )
    with app.run() as app:
        app.sync_data(
            remote_project_mount_path / "data", remote_data_path, reset=False
        )
        app.run_kedro(remote_project_mount_path, remote_data_path)


@modal_group.command(help="Run kedro project on Modal")
@click.pass_obj
def debug(metadata):
    app, remote_project_mount_path, remote_data_path = main_app(
        metadata.project_path, metadata.project_name, metadata.package_name
    )
    app.interactive_shell()


@modal_group.command(
    help="Deploy kedro project to Modal, scheduling it to run daily"
)
@click.pass_obj
def deploy(metadata):
    app, remote_project_mount_point, remote_data_path = main_app(
        metadata.project_path, metadata.project_name, metadata.package_name
    )
    name = f"kedro.{metadata.project_name}"
    app.deploy(name)
    sync_data = modal.lookup(name, "sync_data")  # use the deployed function
    sync_data(remote_project_mount_point / "data", remote_data_path)


@modal_group.command(
    short_help="Sync the local data directory to Modal",
    help="Sync the local data directory to Modal, overwriting any existing data there",
)
@click.pass_obj
def reset(metadata):
    app, source_path, destination_path = sync_app(
        metadata.project_path, metadata.project_name
    )
    with app.run() as app:
        app.sync_data(source_path, destination_path, reset=True)
