# Kedro-Modal (A plugin for Kedro)

kedro-modal is an experimental Modal plugin for Kedro (https://kedro.readthedocs.io/en/stable/).

The plugin lets you run Kedro Python pipelines on Modal in an effortless way.

Data in your `<project-name>/data` directory will be synced to a persisted Modal [Shared Volume](https://modal.com/docs/guide/shared-volumes) and any datasets defined as local in your Kedro data catalog will be written to the same volume.

## Installation instructions
First make sure that you have Modal installed (See https://modal.com/home)

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


## Features
At this point it only supports the basic use case of running a Kedro project (equivlalent of a basic `kedro run`) on Modal instead of running it locally.


* It pushes the project source code to Modal
* Installs requirements.txt of the project in a Modal image
* Sets up a modal Shared Volume for syncing local input data from project/data and writing any output from the Modal runs.

### Notably missing features at this point (but should not be too tricky to handle):
* External data processors (e.g. spark)
* Parallel runs using a custom Kedro Runner subclass
* Default dataset support using Modal distributed in-memory storage (i.e. modal.Dict) - mostly needed if running parallelism
* Using Modal functions as explicit Kedro nodes/steps (should technically be doable by just using Modal as is, but would probably need some kind of integration with this plugin)
