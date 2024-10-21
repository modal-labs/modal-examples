# ---
# cmd: ["python", "13_sandboxes/safe_code_execution.py"]
# tags: ["use-case-sandboxed-code-execution"]
# pytest: false
# ---

# # Running arbitrary code in a sandboxed environment

# This example demonstrates how to run arbitrary code in a Modal [Sandbox](https://modal.com/docs/guide/sandbox).

# ## Setting up a multi-language environment

import modal

# Sandboxes allow us to run any kind of code in a safe environment. We'll use an image with a few
# different language runtimes to demonstrate this.

image = modal.Image.debian_slim(python_version="3.11").apt_install(
    "nodejs", "ruby", "php"
)
app = modal.App.lookup("safe-code-execution", create_if_missing=True)

# We'll now create a Sandbox with this image. We'll also enable output so we can see the image build
# logs. Note that we don't pass any commands to the Sandbox, so it will stay alive, waiting for us
# to send it commands.

with modal.enable_output():
    sandbox = modal.Sandbox.create(app=app, image=image)

print(f"Sandbox ID: {sandbox.object_id}")

# ## Running bash, Python, Node.js, Ruby, and PHP in a Sandbox

# We can now use [`Sandbox.exec`](https://modal.com/docs/reference/modal.Sandbox#exec) to run a few different
# commands in the Sandbox.

bash_ps = sandbox.exec("echo", "hello from bash")
python_ps = sandbox.exec("python", "-c", "print('hello from python')")
nodejs_ps = sandbox.exec("node", "-e", 'console.log("hello from nodejs")')
ruby_ps = sandbox.exec("ruby", "-e", "puts 'hello from ruby'")
php_ps = sandbox.exec("php", "-r", "echo 'hello from php';")

print(bash_ps.stdout.read(), end="")
print(python_ps.stdout.read(), end="")
print(nodejs_ps.stdout.read(), end="")
print(ruby_ps.stdout.read(), end="")
print(php_ps.stdout.read(), end="")
print()

# ```
# hello from bash
# hello from python
# hello from nodejs
# hello from ruby
# hello from php
# ```

# We can use multiple languages in tandem to build complex applications.
# Let's demonstrate this by piping data between Python and Node.js using bash. Here
# we generate some random numbers with Python and sum them with Node.js.

combined_process = sandbox.exec(
    "bash",
    "-c",
    """python -c 'import random; print(\" \".join(str(random.randint(1, 100)) for _ in range(10)))' |
    node -e 'const readline = require(\"readline\");
    const rl = readline.createInterface({input: process.stdin});
    rl.on(\"line\", (line) => {
      const sum = line.split(\" \").map(Number).reduce((a, b) => a + b, 0);
      console.log(`The sum of the random numbers is: ${sum}`);
      rl.close();
    });'""",
)

result = combined_process.stdout.read().strip()
print(result)

# ```
# The sum of the random numbers is: 501
# ```

# For long-running processes, you can use stdout as an iterator to stream the output.

slow_printer = sandbox.exec(
    "ruby",
    "-e",
    """
    10.times do |i|
      puts "Line #{i + 1}: #{Time.now}"
      STDOUT.flush
      sleep(0.5)
    end
    """,
)

for line in slow_printer.stdout:
    print(line, end="")

# ```
# Line 1: 2024-10-21 15:30:53 +0000
# Line 2: 2024-10-21 15:30:54 +0000
# Line 3: 2024-10-21 15:30:54 +0000
# Line 4: 2024-10-21 15:30:55 +0000
# Line 5: 2024-10-21 15:30:55 +0000
# Line 6: 2024-10-21 15:30:56 +0000
# Line 7: 2024-10-21 15:30:56 +0000
# Line 8: 2024-10-21 15:30:57 +0000
# Line 9: 2024-10-21 15:30:57 +0000
# Line 10: 2024-10-21 15:30:58 +0000
# ```

# Since sandboxes are safely separated from the rest of our system,
# we can run very dangerous code in them!

sandbox.exec("rm", "-rf", "/", "--no-preserve-root")

# This command has deleted the entire filesystem, so we can't run any more commands.
# Let's terminate the sandbox to finish the example.

sandbox.terminate()
