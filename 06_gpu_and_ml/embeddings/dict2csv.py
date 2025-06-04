import modal

# Create a persisted dict - the data gets retained between app runs
data = modal.Dict.from_name("laion2B", create_if_missing=True)
keys = sorted(list(data.keys()))
experiments = list({key.split("-")[0] for key in keys})
ms = ["exp-start", "embedder-init", "embedding-begin", "embedding-complete"]
for exp in experiments:
    try:
        msg = f"Experiment: {exp}"
        ms1 = ms[0]
        for idx in range(1, len(ms)):
            ms0 = ms[idx]
            dur = (data[f"{exp}-{ms0}"] - data[f"{exp}-{ms1}"]) / 1e9
            msg += f"\n\t{ms0} duration: {dur:.2E}s"
        if "M" in exp:
            n_images = 1e6
        else:
            n_images = 23410
        msg += f"\n\t\tThroughput: {n_images / dur: .2f} images/second"
        print(msg)
    except:
        if msg:
            msg = "Incomplete Experiment: " + f"\n\t {msg}"
            print(msg)
        continue
