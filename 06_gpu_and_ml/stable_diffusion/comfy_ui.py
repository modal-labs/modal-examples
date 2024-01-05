import asyncio
import threading

import modal

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    # Here we place the latest ControlNet repository code into /root.
    # Because /root is almost empty, but not entirely empty
    # as it contains this comfy_ui.py script, `git clone` won't work,
    # so this `init` then `checkout` workaround is used.
    .run_commands(
        "cd /root && git init .",
        "cd /root && git remote add --fetch origin https://github.com/comfyanonymous/ComfyUI",
        "cd /root && git checkout master",
        "cd /root && pip install xformers!=0.0.18 -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://download.pytorch.org/whl/cu117",
    )
)
stub = modal.Stub(name="example-comfy-ui", image=image)

@stub.function(gpu="any")
@modal.asgi_app()
def f():
    print("hello")
    import server
    import execution
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = server.PromptServer(loop)
    server.add_routes()
    return server.app
