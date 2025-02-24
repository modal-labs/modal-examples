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
npm install
npx vite build --watch
```

### Backend

Run this to create an ephemeral app live reloading the web API. That way you can iterate on the application backend:

```bash
cd "$(git rev-parse --show-toplevel)/06_gpu_and_ml/text-to-pokemon"
modal serve -m text_to_pokemon.main
```

Sending <kbd>Ctrl</kbd>+<kbd>C</kbd> will stop your app.

### Tests

```bash
python3 -m pytest
```

## Deploy

```bash
cd "$(git rev-parse --show-toplevel)/06_gpu_and_ml/text-to-pokemon/text_to_pokemon/frontend"
npm install
npx vite build
cd "$(git rev-parse --show-toplevel)/06_gpu_and_ml/text-to-pokemon/"
modal -m deploy text_to_pokemon.main
```
