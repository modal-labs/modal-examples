import json
import os
import subprocess
import sys


def load_event():
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        print("GITHUB_EVENT_PATH not set", file=sys.stderr)
        sys.exit(1)
    try:
        with open(event_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading event JSON: {e}", file=sys.stderr)
        sys.exit(1)


def determine_diff_range(event, event_name):
    if event_name == "pull_request":
        try:
            base = event["pull_request"]["base"]["sha"]
            head = event["pull_request"]["head"]["sha"]
        except KeyError as e:
            print(f"Missing key in pull_request event: {e}", file=sys.stderr)
            sys.exit(1)
    elif event_name == "push":
        base = event.get("before")
        head = event.get("after")

    elif event_name == "workflow_dispatch":
        try:
            subprocess.run(["git", "fetch", "origin", "main"], check=True)

            base = (
                subprocess.check_output(["git", "rev-parse", "origin/main"])
                .decode()
                .strip()
            )

            head = (
                subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
            )
        except subprocess.CalledProcessError as e:
            print(f"Git error while determining diff range: {e}", file=sys.stderr)
            sys.exit(1)

    else:
        print(f"Unsupported event type: {event_name}", file=sys.stderr)
        sys.exit(1)

    if not base or not head:
        print("Could not determine base and head commits", file=sys.stderr)
        sys.exit(1)
    return base, head


def get_changed_files(base, head):
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", base, head],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.splitlines()
    except subprocess.CalledProcessError as e:
        print(f"Error running git diff: {e}", file=sys.stderr)
        sys.exit(1)


def filter_files(files):
    return [
        f
        for f in files
        if f.endswith(".py")
        and not (f.startswith("internal/") or f.startswith("misc/"))
    ]


def write_output(key, value):
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        try:
            with open(github_output, "a") as out:
                out.write(f"{key}={value}\n")
        except Exception as e:
            print(f"Error writing to GITHUB_OUTPUT: {e}", file=sys.stderr)


def main():
    event = load_event()
    event_name = event.get("event_name") or os.environ.get("GITHUB_EVENT_NAME")
    base, head = determine_diff_range(event, event_name)
    changed_files = get_changed_files(base, head)
    filtered_files = filter_files(changed_files)
    json_output = json.dumps(filtered_files)
    write_output("all_changed_files", json_output)
    print(json_output)


if __name__ == "__main__":
    main()
