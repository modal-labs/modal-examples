# SSH Proxy

A containerized SSH proxy service, orchestrated and deployed with [Modal](https://modal.com/), for secure, authenticated, and dynamic interactive SSH access to running sandboxes and applications.

---

## Overview

This project is a Rust-based SSH server, wrapped with a Python `modaldeploy.py` script for easy cloud deployment using Modal's serverless execution environment. It provides controlled SSH access, managed keypairs, and sandboxes for remote sessions, suitable for cloud-native workflows and bot automation.

**Key features:**
- Rust SSH server (`ssh-proxy`) for robust, performant connections.
- Modal-based deployment for seamless cloud hosting and port forwarding.
- Runtime dynamic management of authorized keys and sandboxes.
- Secure key generation, fingerprinting, and authentication workflows.
- Persistent container environments for apps.
- PTY (pseudo-terminal) connection management for terminal support.

---

## Directory Structure

```
ssh-proxy/
├── src/
│   ├── key_handler.rs          # SSH keypair and fingerprint logic
│   ├── main.rs                 # SSH server setup & main event loop
│   ├── pty_connection_manager.rs # PTY (terminal) connection/session manager
│   ├── server_ops.rs           # API/client credentials & server RPC ops
│   └── ...                     # Other supporting modules
├── Cargo.toml                  # Rust dependencies
modaldeploy.py                  # Modal app spec & deployment script
README.md                       # This file
```

---

## Quick Start

### 1. Generate your SSH keypair

To access the proxy, you must generate an SSH keypair and provide the **public key** to the Modal app (the containerized server), while keeping the **private key** secure on your local machine. This replaces or supplements password authentication with key-based authentication.

**How to generate a keypair with `ssh-keygen`:**

```sh
ssh-keygen -t ed25519 -C "your_email@example.com" -f ~/.ssh/ssh-proxy-test
```

- This will generate two files:
  - `~/.ssh/ssh-proxy-test` (your private key, *do not share*)
  - `~/.ssh/ssh-proxy-test.pub` (your public key)

**IMPORTANT:**  
- Keep your private key safe, and never share it.
- You will use the public key for Modal deployment as shown below.

---

### 2. Set environment variables

In addition to Modal API authentication, you must provide your SSH **public key** to the Modal app so it can authorize your login.

Set the following environment variables before running the deployment:

- `MODAL_TOKEN_ID`  
- `MODAL_TOKEN_SECRET`  
- `SSH_PROXY_PUB_KEY` — the contents of your `.pub` file (single line).

**Example (.env or shell export):**

```sh
export MODAL_TOKEN_ID=your_token_id
export MODAL_TOKEN_SECRET=your_token_secret
export SSH_PROXY_PUB_KEY="$(cat ~/.ssh/ssh-proxy-test.pub)"
```

- The Modal app will read the `SSH_PROXY_PUB_KEY` environment variable, and configure the SSH server to accept logins using that key.

---

### 3. Deploy and run

```sh
python modaldeploy.py
```

This will:
- Build and run the Rust SSH server inside a Modal container.
- Expose port 22 for SSH traffic.
- Output the SSH endpoint via a secure Modal tunnel.

---

### 4. SSH into your proxy

Use your **private key** (`~/.ssh/ssh-proxy-test`) with the SSH client and the tunnel URL printed by the deployment script:

```sh
ssh -i ~/.ssh/ssh-proxy-test <user>@<tunnel_url>
```

Replace `<user>` with the appropriate username expected by the server (refer to your implementation; often `modal` or `sshproxy`).

---

## SSH Key Authentication and Workflow

1. **You** generate an SSH keypair on your machine.
2. **You** set the public key as `SSH_PROXY_PUB_KEY` in the Modal deployment environment.
3. **You** connect via SSH using your corresponding private key.
4. **The server** verifies your connection against the public key it was provided.

---

## Development

### Local Rust Build

You can build and run the SSH server with Cargo:

```sh
cd ssh-proxy
cargo build
cargo run --release --bin ssh_proxy
```

### Modal Development Notes

- The Docker-like environment uses a Debian slim base image with Rust installed.
- Local source code is injected into the Modal container at build time with `.add_local_dir`.
- PTY and networking support are provided for interactive SSH sessions.

---

## Security

- **Key Management:** See `src/key_handler.rs` for generation and fingerprinting logic. The authorized public key is supplied via `SSH_PROXY_PUB_KEY` at deployment.
- **API Authentication:** Credentials injected into API requests (`server_ops.rs`) using environment variables and custom headers.
- **Timeouts & Session Management:** Inactivity and authentication timeouts are enforced (`main.rs`, `pty_connection_manager.rs`).

---

## Main Components

### 1. `key_handler.rs`
Handles generation and fingerprinting of SSH keypairs:
- `generate_key_pair`: Deterministic (mock/testing) or random keypair creation.
- `key_fingerprint_sha256`: SHA-256 fingerprinting for public keys.

### 2. `pty_connection_manager.rs`
Manages PTY connection/session logic for sandboxed SSH sessions.

### 3. `server_ops.rs`
- Manages credentials, header injection, and main RPC client logic for interacting with the Modal API.

### 4. `modaldeploy.py`
Python entrypoint for Modal deployment:
- Defines the app, builds Rust project in the container, starts the SSH server, and forwards port 22.
- Handles Modal port forwarding (tunnel provisioning).
- **Reads `SSH_PROXY_PUB_KEY` from the environment and injects it into the server config.**

---

## Example Session

```sh
$ python modaldeploy.py
Tunnel created at  tunnel-12345.tunnel.modal.run
...

$ ssh -i ~/.ssh/ssh-proxy-test modal@tunnel-12345.tunnel.modal.run
# Interactive terminal session
```

---

## Configuration

Environment variables consumed:
- `MODAL_TOKEN_ID` (API token)
- `MODAL_TOKEN_SECRET` (API secret)
- `SSH_PROXY_PUB_KEY` (authorized SSH public key)

Other SSH user/session configuration is dynamic and backed by the Modal API.

---

## Troubleshooting

- If you have connection issues, check the logs printed by `modaldeploy.py` and the Modal dashboard.
- Ensure the correct Rust version and dependencies are installed inside the Modal container (handled by `modaldeploy.py`).
- If SSH authentication fails, confirm that the correct public key is supplied via `SSH_PROXY_PUB_KEY`, and you are using the corresponding private key in your SSH command.
- For debugging, run `cargo run` locally to verify the Rust server.

---

## Contributing

Pull requests and issues are welcome! Please make sure to test inside both Modal and local Rust environments.

---

## License

MIT

---


