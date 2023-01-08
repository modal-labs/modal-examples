# Text to Pokémon Card

This demo application is an example of the new 'model stacking' applications enabled by
the StableDiffusion release. Two or more ML models are combined with other code (and even human)
guidance to produce images that would take hours for a skilled practitioner to create by hand.

The resulting generated images are rendered with [simeydotme's](https://github.com/simeydotme/pokemon-cards-css)
excellent CSS work.

## Developing

### Frontend

```bash
cd "$(git rev-parse --show-toplevel)/06_gpu_and_ml/text-to-pokemon/text_to_pokemon/frontend"
npx vite build --watch
```

### Backend

Run this to `stub.serve()` the web API and iterate on the application backend:

```bash
cd "$(git rev-parse --show-toplevel)/06_gpu_and_ml/text-to-pokemon"
python3 -m text_to_pokemon.main
```

Sending <kbd>Ctrl</kbd>+<kbd>C</kbd> will stop your app.

### Tests

```
python3 -m pytest
```

## Deploy

```bash
cd "$(git rev-parse --show-toplevel)/06_gpu_and_ml/text-to-pokemon/text_to_pokemon/frontend"
npx vite build
cd "$(git rev-parse --show-toplevel)/06_gpu_and_ml/text-to-pokemon/"
modal app deploy text_to_pokemon.main
```
