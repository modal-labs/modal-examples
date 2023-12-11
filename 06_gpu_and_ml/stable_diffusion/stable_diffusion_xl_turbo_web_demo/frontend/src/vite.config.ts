import postcss from './postcss.config.js'
import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'
import path from 'path';

export default defineConfig({
  plugins: [svelte()],
  css:{
	  postcss
  },
  build: {
    minify: true
  },
  resolve: {
    alias: {
      $lib: path.resolve('./src/lib'),
    },
  },
})
