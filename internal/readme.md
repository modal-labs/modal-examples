## `Internal/`

This is internal repository and documentation management code. It does not
contain examples.

### Continuous Integration and Continuous Deployment

Modal cares deeply about the correctness of our examples -- we have also
suffered from janky, poorly-maintained documentation and we do our best to
ensure that our examples don't pay that forward.

This document explains the CI/CD process we use. It is primarily intended for
Modal engineers, but if you're contributing an example and have the bandwidth to
set up the testing as well, we appreciate it!

#### Frontmatter

Examples can include a small frontmatter block in YAML format that controls
testing and deployment behavior.

Fields:

- `deploy`: If `true`, the example is deployed as a Modal application with
  `modal deploy`. If `false`, it is not. Default is `false`.
- `cmd`: The command to run the example for testing. Default is
  `["modal", "run", "<filename>"]`.
- `args`: Arguments to pass to the command. Default is `[]`.
- `lambda-test`: If `true`, the example is tested with the cli command provided
  in `cmd`. If `false`, it is not. Default is `true`. Note that this controls
  execution in the CI/CD of this repo and in the monitor-based testing.
- `runtimes`: Control which runtimes the example is executed on in synthetic
  monitoring. Default is `["runc", "gvisor"]`.
- `env`: A dictionary of environment variables to include when testing.
  Default is `{}`, but note that the environment can be modified in the CI/CD of this repo
  or in the monitor-based testing.

Below is an example frontmatter for deploying a web app. Note that here we
`modal serve` in the test so as to not deploy to prod when testing. Note that in
testing environments, the `MODAL_SERVE_TIMEOUT` environment variable is set so
that the command terminates.

```yaml
---
deploy: true
cmd: ["modal", "serve", "10_integrations/pushgateway.py"]
---
# example prose and code begins here
```

#### Testing in GitHub Actions

When a PR is opened, any changed examples are run via GitHub Actions.

This workflow is intended to catch errors at the time a PR is made -- incuding
both errors in the example and issues with the execution of the example in the
monitoring system.

#### Continual Monitoring

Examples are executed regularly and at random to check for regressions. The
results are monitored.

Modal engineers, see `synthetic_monitoring` in the `modal` repo for details.
