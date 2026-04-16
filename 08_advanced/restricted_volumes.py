# ---
# cmd: ["python", "08_advanced/restricted_volumes.py"]
# pytest: false
# ---
import modal

app = modal.App.lookup(name="restricted-sb-app", create_if_missing=True)
volume = modal.Volume.from_name("restricted-sb-data", create_if_missing=True, version=2)

image = (
    modal.Image.debian_slim()
    .apt_install("sudo")
    .run_commands(
        "sudo adduser --disabled-password --gecos '' user1",
        "sudo adduser --disabled-password --gecos '' user2",
    )
)

sandbox = modal.Sandbox.create(app=app, image=image, volumes={"/data": volume})
print("Sandbox ID: ", sandbox.object_id)


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
p = sandbox.exec("sh", "-c", "mkdir -p /data/user2")
p = sandbox.exec(
    "sh", "-c", "chown -R user1:user1 /data/user1  && chmod 700 /data/user1"
)
p = sandbox.exec(
    "sh", "-c", "chown -R user2:user2 /data/user2  && chmod 700 /data/user2"
)
print("Sandbox setup complete.")

print("\n🟢 Baseline exec (unrestricted, should succeed):")
p = sandbox.exec("sh", "-c", "ls -la /data/user1")
for line in p.stdout:
    print(line, end="")

print("\n🟢 Restricted user1 exec (should succeed):")
p = sandbox_restricted_exec(sandbox, "ls -la /data/user1", "user1")

print("\n🔴 Restricted user1 exec (should fail):")
p = sandbox_restricted_exec(sandbox, "ls -la /data/user2", "user1")
