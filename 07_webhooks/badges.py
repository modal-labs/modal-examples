# # Serve a dynamic SVG badge

# In this example, we use Modal's [webhook](/docs/guide/webhooks) capability to host a dynamic SVG badge that shows
# you the current # of downloads for a Python package.
#
# First let's start off by creating a Modal stub, and defining an image with the Python packages we're going to be using:

import modal

stub = modal.Stub(
    "example-web-badges",
    image=modal.Image.debian_slim().pip_install("pybadges", "pypistats"),
)

# ## Defining the webhook
#
# Instead of using `@stub.function` to decorate our function, we use the
# `@modal.webhook` decorator ([learn more](/docs/guide/webhooks#webhook)), which instructs Modal
# to create a REST endpoint that serves this function. Note that the default method is `GET`, but this
# can be overridden using the `method` argument.


@stub.webhook
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
# Since Modal webhooks are FastAPI functions under the hood, we return this SVG wrapped in a FastAPI response with the correct media type.
# Also note that FastAPI automatically interprets `package_name` as a [query param](https://fastapi.tiangolo.com/tutorial/query-params/).

# ## Running and deploying
#
# We can now run this function as follows:

if __name__ == "__main__":
    stub.serve()

# You can run this script and it will create a short-lived web url that exists
# until you terminate the script.
#
# If you want to create a persistent URL, you have to deploy the script.
# To deploy using the Modal CLI by running `modal deploy web_badges.py`,
#
# Either way, as soon as we run this command, Modal gives us the link to our brand new
# webhook in the output:
#
# ![web badge deployment](./badges_deploy.png)
#
# We can now visit the link using a web browser, using a `package_name` of our choice in the URL query params.
# For example:
# - `https://YOUR_SUBDOMAIN.modal.run/?package_name=synchronicity`
# - `https://YOUR_SUBDOMAIN.modal.run/?package_name=torch`
