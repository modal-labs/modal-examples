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

print("⌛Setting up sandbox...")
sandbox.exec("sh", "-c", "mkdir -p /data/user1").wait()
sandbox.exec("sh", "-c", "mkdir -p /data/user2").wait()
sandbox.exec(
    "sh", "-c", "chown -R user1:user1 /data/user1 && chmod 700 /data/user1"
).wait()
sandbox.exec(
    "sh", "-c", "chown -R user2:user2 /data/user2 && chmod 700 /data/user2"
).wait()
print("Sandbox setup complete.")

print("\n🟢 Baseline exec (unrestricted, should succeed):")
p = sandbox.exec("sh", "-c", "ls -la /data/user1")
for line in p.stdout:
    print(line, end="")
p.wait()
assert p.returncode == 0, "Unrestricted exec should succeed"

print("\n🟢 Restricted user1 exec (should succeed):")
p = sandbox.exec("su", "-", "user1", "-c", "ls -la /data/user1")
for line in p.stdout:
    print(line, end="")
for line in p.stderr:
    print(line, end="")
p.wait()
assert p.returncode == 0, "user1 should access own directory"

print("\n🔴 Restricted user1 exec (should fail):")
p = sandbox.exec("su", "-", "user1", "-c", "ls -la /data/user2")
for line in p.stdout:
    print(line, end="")
for line in p.stderr:
    print(line, end="")
p.wait()
assert p.returncode != 0, "user1 should not access user2's directory"

url = f"https://modal.com/id/{sandbox_id}"
print(
    f"\n☀️ Sandbox live! See: {url}\nYou can use modal.Sandbox.from_id('{sandbox_id}') to run additional commands."
)
