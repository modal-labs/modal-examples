# Embedding Wikipedia in 15 minutes

This example shows how we can embed the entirety of english wikipedia on Modal in just 15 minutes. We've published a detailed writeup which walks you through the implemenation [here](#todo).

## Description

There are a total of 2 files in this repository

- `download.py` : This showcases how to download the Wikipedia dataset into a `Modal` volume. We can take advantage of `Modal`'s high internet speeds to download large datasets quickly.

- `main.py`: This showcases how to run an embedding job on your downloaded dataset and run a parallelizable job using Modal's inbuilt parallelization abstraction.

## Getting Started

You'll need a few packages to get started - we recommend using a virtual environment to install all of the dependencies listed in the `requirements.txt`

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install modal
```

Once you've done so, you'll need to authenticate with Modal. To do so, run the command `modal token new`.

This will open up a new tab in your default browser and allow you to run, deploy and configure all of your Modal applications from your terminal.

## Downloading Our Dataset

Let's first download our Wikipedia dataset into a Modal volume. We can optimise the download time using the `num_proc ` keyword to parallelize some of the downloads.

From experience, this reduces the amount of time required by around 30-40% as long as we set a number between 4-10.

We can run our Download script using the command

```
modal run download.py
```

## Embedding our Dataset

Now that we've downloaded our wikipedia dataset, we can now embed the entire dataset using our `main.py` script. We can run it using the command

```
modal run main.py
```

Note that we utilize 2 volumes in our dataset script - one for reading from and another to write the files to upload to.

# Debugging

## Verifying that the Dataset has been downloaded

> Note that the `size` of the volume listed in the table for the directories. Our wikipedia directory is listed as having a size of 56B but the multiple .arrow files inside it should tell you that it in fact contains much larger files

Once we've downloaded the dataset, we can confirm that it has been downloaded and saved into our `embedding-wikipedia` volume at the path `/wikipedia` by runnning the command

```
modal volume ls embedding-wikipedia
```

This should produce a table that looks like this.

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ filename                                            ┃ type ┃ created/modified          ┃ size      ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ wikipedia                                           │ dir  │ 2023-12-02 10:57:44+01:00 │ 56 B      │
└─────────────────────────────────────────────────────┴──────┴───────────────────────────┴───────────┘
```

We can then view what this folder looks like inside by appending the `/wikipedia` to our command

```
modal volume ls embedding-wikipedia /wikipedia
```

This will then show the files inside the `/wikipedia`

```
Directory listing of '/wikipedia' in 'embedding-wikipedia'
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ filename                    ┃ type ┃ created/modified          ┃ size    ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ wikipedia/train             │ dir  │ 2023-12-02 10:58:12+01:00 │ 4.0 KiB │
│ wikipedia/dataset_dict.json │ file │ 2023-12-02 10:57:44+01:00 │ 21 B    │
└─────────────────────────────┴──────┴───────────────────────────┴─────────┘
```

## Removing Files

> Note that if you're looking to remove a directory, you need to supply the `--recursive` flag to the command for it to work.

If you'll like to save on storage costs when using volumes, you can use the modal cli to easily remove files.

```
modal volume rm embedding-wikipedia /wikipedia --recursive
```
