<!--
  ✍️ Write a short summary of your work. Screenshots and videos are welcome!
-->

### Type of Change

- [ ] New example
- [ ] Example updates (Bug fixes, new features, etc.)
- [ ] Other (changes to the codebase, but not to examples)

## Checklist

- [ ] Example is testable in synthetic monitoring system, or `lambda-test: false` is added to example frontmatter (`---`)
  - [ ] Example is tested by executing with `modal run` or an alternative `cmd` is provided in the example frontmatter (e.g. `cmd: ["modal", "deploy"]`)
  - [ ] Example is tested by running with no arguments or the `args` are provided in the example frontmatter (e.g. `args: ["--prompt", "Formula for room temperature superconductor:"]`
- [ ] Example is documented with comments throughout, in a [_Literate Programming_](https://en.wikipedia.org/wiki/Literate_programming) style.
- [ ] Example does _not_ require third-party dependencies to be installed locally
- [ ] Example pins its dependencies
  - [ ] Example pins container images to a stable tag, not a dynamic tag like `latest`
  - [ ] Example specifies a `python_version` for the base image, if it is used
  - [ ] Example pins all dependencies to at least minor version, `~=x.y.z` or `==x.y`
  - [ ] Example dependencies with `version < 1` are pinned to patch version, `==0.y.z`

## Outside contributors

You're great! Thanks for your contribution.
