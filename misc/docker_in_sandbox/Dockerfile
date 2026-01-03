FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists \
    apt-get update && \
    apt-get install -y --no-install-recommends gnupg lsb-release ca-certificates wget curl net-tools iproute2


RUN install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc && \
    chmod a+r /etc/apt/keyrings/docker.asc && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo \"${UBUNTU_CODENAME:-$VERSION_CODENAME}\") stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    mkdir build


RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    docker-ce=5:27.5.0-1~ubuntu.22.04~jammy \
    docker-ce-cli=5:27.5.0-1~ubuntu.22.04~jammy \
    containerd.io docker-buildx-plugin docker-compose-plugin

RUN rm $(command -v runc) && \
    wget https://github.com/opencontainers/runc/releases/download/v1.3.0/runc.amd64 && \
    chmod +x runc.amd64 && \
    mv runc.amd64 /usr/local/bin/runc && \
    update-alternatives --set iptables /usr/sbin/iptables-legacy && \
    update-alternatives --set ip6tables /usr/sbin/ip6tables-legacy

COPY start-dockerd.sh /start-dockerd.sh
RUN chmod +x /start-dockerd.sh
CMD [ "/start-dockerd.sh" ]
