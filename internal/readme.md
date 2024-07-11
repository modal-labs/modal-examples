## `Internal/`

This is internal repository and documentation management code. It does not
contain examples.

### Frontmatter

Examples have frontmatter that control testing and deployment behavior. Fields:

- `deploy`: If `true`, the example is deployed to the website with
  `modal deploy`. If `false`, it is not. Default is `false`.
- `cmd`: The command to run the example for testing. Default is
  `["modal", "run", "<filename>"]`.
- `args`: Arguments to pass to the command. Default is `[]`.
- `lambda-test`: If `true`, the example is tested with the cli command provided
  in `cmd`. If `false`, it is not. Default is `true`.
- `runtimes`: Control which runtimes the example is executed on in synthetic
  monitoring. Default is `["runc", "gvisor"]`.

Example for a web app. Note that here we `modal serve` in the test so as to not
deploy to prod when testing.

```yaml
deploy: true
cmd: ["modal", "serve", "10_integrations/pushgateway.py"]
```
