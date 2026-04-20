# ---
# cmd: ["python", "08_advanced/restricted_volumes.py"]
# pytest: false
# ---
import modal

app = modal.App.lookup(name="example-restricted-volumes", create_if_missing=True)
volume = modal.Volume.from_name(
    "example-restricted-volumes-data", create_if_missing=True, version=2
)

image = (
    modal.Image.debian_slim()
    .apt_install("sudo")
    .run_commands(
        "sudo adduser --disabled-password --gecos '' user1",
        "sudo adduser --disabled-password --gecos '' user2",
    )
)

sandbox = modal.Sandbox.create(app=app, image=image, volumes={"/data": volume})
sandbox_id = sandbox.object_id
print("Sandbox ID: ", sandbox_id)


def sandbox_restricted_exec(sandbox: modal.Sandbox, command: str, user: str):
    """Execute a command in a Sandbox as a restricted user."""
    p = sandbox.exec("su", "-", user, "-c", f"{command}")
    for line in p.stdout:
        print(line, end="")
    for line in p.stderr:
        print(line, end="")
    return p


print("⌛Setting up sandbox...")
p = sandbox.exec("sh", "-c", "mkdir -p /data/user1")
p.wait()
p = sandbox.exec("sh", "-c", "mkdir -p /data/user2")
p.wait()
p = sandbox.exec(
    "sh", "-c", "chown -R user1:user1 /data/user1  && chmod 700 /data/user1"
)
p.wait()
p = sandbox.exec(
    "sh", "-c", "chown -R user2:user2 /data/user2  && chmod 700 /data/user2"
)
p.wait()
print("Sandbox setup complete.")

print("\n🟢 Baseline exec (unrestricted, should succeed):")
p = sandbox.exec("sh", "-c", "ls -la /data/user1")
p.wait()
for line in p.stdout:
    print(line, end="")

print("\n🟢 Restricted user1 exec (should succeed):")
p = sandbox_restricted_exec(sandbox, "ls -la /data/user1", "user1")
p.wait()

print("\n🔴 Restricted user1 exec (should fail):")
p = sandbox_restricted_exec(sandbox, "ls -la /data/user2", "user1")
p.wait()

url = f"https://modal.com/id/{sandbox_id}"

print(
    f"\n☀️ Sandbox live! See: {url}\nYou can use modal.Sandbox.from_id('{sandbox_id}') to run additional commands."
)
# sandbox = modal.Sandbox.from_id(sandbox_id)
