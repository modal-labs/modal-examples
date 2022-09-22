# Kedro-Modal (A plugin for Kedro)

kedro-modal is an experimental Modal plugin for Kedro (https://kedro.readthedocs.io/en/stable/).

The plugin lets you run Kedro Python pipelines on Modal in an effortless way.

Data in your `<project-name>/data` directory will be synced to a persisted Modal [Shared Volume](https://modal.com/docs/guide/shared-volumes) and any datasets defined as local in your Kedro data catalog will be written to the same volume.

## Installation instructions

```bash
cd modal-examples/integrations/kedro-modal
pip install .
```

This installs the kedro-modal project cli commands, e.g. `kedro modal --help`

## Usage
The following commands assume you have navigated to your kedro project, e.g.:
`cd my_kedro_project`
### Run a Kedro pipeline on Modal:
```bash
kedro modal run
```

### Reset the remote data volume using local data
```bash
kedro modal reset
```

## Inspect and download output data
### List Kedro data volumes on Modal:
```bash
modal volume list | grep '^kedro.*storage$'
```

Example output:
```
kedro.Spaceflights.storage
```

### Download an output file (example):
```bash
modal volume get kedro.Spaceflights.storage data/02_intermediate/preprocessed_shuttles.pq .
```
