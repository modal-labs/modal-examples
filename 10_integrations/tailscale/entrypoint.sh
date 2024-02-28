#!/bin/sh

# Custom entrypoint [1] used to login into Tailscale and start both SOCKS5 and HTTP
# proxies. This requires the env var `TAILSCALE_AUTHKEY` to be populated with a 
# Tailscale auth key. [2]
#
# [1] https://modal.com/docs/guide/custom-container#entrypoint
# [2] https://tailscale.com/kb/1111/ephemeral-nodes

set -e

tailscaled --tun=userspace-networking --socks5-server=localhost:1080 --outbound-http-proxy-listen=localhost:1080 &
tailscale up --authkey=${TAILSCALE_AUTHKEY} --hostname=${MODAL_TASK_ID}

# Loop until the maximum number of retries is reached
retry_count=0
while [ $retry_count -lt 5 ]; do
    http_status=$(curl -x socks5://localhost:1080 -o /dev/null -L -s -w '%{http_code}' https://www.google.com)

    # Check if the HTTP status code is 200 (OK)
    if [ $http_status -eq 200 ]; then
        echo "Successfully started SOCKS5 proxy, HTTP proxy, and connected to Tailscale."
        exec "$@" # Runs the command passed to the entrypoint script.
        exit 0
    else
        echo "Attempt $((retry_count+1))/$MAX_RETRIES failed: SOCKS5 proxy returned HTTP $http_status"
    fi

    retry_count=$((retry_count+1))
    sleep 1
done

echo "Failed to start Tailscale."
exit 1