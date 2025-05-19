<!--
  ✍️ Write a short summary of your work. Screenshots and videos are welcome!
-->

## Type of Change

<!--
  ☑️ Check one of the top-level boxes and delete the others.
-->

- [ ] New example for the GitHub repo
  - [ ] New example for the documentation site (Linked from a discoverable page, e.g. via the sidebar in `/docs/examples`)
- [ ] Example updates (Bug fixes, new features, etc.)
- [ ] Other (changes to the codebase, but not to examples)

## Documentation Site Checklist

<!--
  ☑️ Review the checklist below if the example is intended for the documentation site.
  All boxes should be checked!
  Otherwise, set `lambda-test: false` in the frontmatter and delete the checklist
  or move the file into `misc/`.
  
  See `internal/README.md` for details on the CI.
-->

### ☑️ Monitoring
  - [ ] Example is configured for testing in the synthetic monitoring system
    - [ ] Example is tested by executing with `modal run`, or an alternative `cmd` is provided in the example frontmatter (e.g. `cmd: ["modal", "serve"]`)
    - [ ] Example is tested by running the `cmd` with no arguments, or the `args` are provided in the example frontmatter (e.g. `args: ["--prompt", "Formula for room temperature superconductor:"]`
    - [ ] Example does _not_ require third-party dependencies besides `fastapi` to be installed locally (e.g. does not import `requests` or `torch` in the global scope or in the scope of a `local_entrypoint`)

### ☑️ Content
  - [ ] Example is documented with comments throughout, in a [_Literate Programming_](https://en.wikipedia.org/wiki/Literate_programming) style
  - [ ] Example media assets are retrieved from `modal-cdn.com`

### ☑️ Build Stability
  - [ ] Example pins all dependencies in container images
    - [ ] Example pins container images to a stable tag like `v1`, not a dynamic tag like `latest`
    - [ ] Example specifies a `python_version` for the base image, if it is used 
    - [ ] Example pins all dependencies to at least [SemVer](https://semver.org/) minor version, `~=x.y.z` or `==x.y`, or we expect this example to work across major versions of the dependency and are committed to maintenance across those versions
      - [ ] Example dependencies with `version < 1` are pinned to patch version, `==0.y.z`

## Outside Contributors

You're great! Thanks for your contribution.
