# ---
# cmd: ["modal", "serve", "07_web_endpoints/badges.py"]
# ---

# # Serve a dynamic SVG badge

# In this example, we use Modal's [webhook](https://modal.com/docs/guide/webhooks) capability to host a dynamic SVG badge that shows
# you the current number of downloads for a Python package.

# First let's start off by creating a Modal app, and defining an image with the Python packages we're going to be using:

import modal

image = modal.Image.debian_slim().pip_install(
    "fastapi[standard]", "pybadges", "pypistats"
)

app = modal.App("example-badges", image=image)

# ## Defining the web endpoint

# In addition to using `@app.function()` to decorate our function, we use the
# [`@modal.fastapi_endpoint` decorator](https://modal.com/docs/guide/webhooks)
# which instructs Modal to create a REST endpoint that serves this function.
# Note that the default method is `GET`, but this can be overridden using the `method` argument.


@app.function()
@modal.fastapi_endpoint()
async def package_downloads(package_name: str):
    import json

    import pypistats
    from fastapi import Response
    from pybadges import badge

    stats = json.loads(pypistats.recent(package_name, format="json"))
    svg = badge(
        left_text=f"{package_name} downloads",
        right_text=str(stats["data"]["last_month"]),
        right_color="blue",
    )

    return Response(content=svg, media_type="image/svg+xml")


# In this function, we use `pypistats` to query the most recent stats for our package, and then
# use that as the text for a SVG badge, rendered using `pybadges`.
# Since Modal web endpoints are FastAPI functions under the hood, we return this SVG wrapped in a FastAPI response with the correct media type.
# Also note that FastAPI automatically interprets `package_name` as a [query param](https://fastapi.tiangolo.com/tutorial/query-params/).

# ## Running and deploying

# We can now run an ephemeral app on the command line using:

# ```shell
# modal serve badges.py
# ```

# This will create a short-lived web url that exists until you terminate the script.
# It will also hot-reload the code if you make changes to it.

# If you want to create a persistent URL, you have to deploy the script.
# To deploy using the Modal CLI by running `modal deploy badges.py`,

# Either way, as soon as we run this command, Modal gives us the link to our brand new
# web endpoint in the output:

# ![web badge deployment](./badges_deploy.png)

# We can now visit the link using a web browser, using a `package_name` of our choice in the URL query params.
# For example:
# - `https://YOUR_SUBDOMAIN.modal.run/?package_name=synchronicity`
# - `https://YOUR_SUBDOMAIN.modal.run/?package_name=torch`
