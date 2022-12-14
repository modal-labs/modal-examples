<!--
  @component

  A 'dumb' progress bar shown to users when the backend is busy generating their
  requested Pokémon cards.

  The bar is 'dumb' because it doesn't change based on information returned by the
  backend API. It has a hardcoded progression policy based on the typical performance
  of the backend API. 
  
  TODO(Jonathon): Add progress info into backend API response and feed it into this component.
-->
<script>
    import { onMount } from "svelte";
    import { tweened } from "svelte/motion";
    import { cubicOut } from "svelte/easing";

    export let finished = false;

    let ticks = 0;

    onMount(() => {
        const interval = setInterval(() => {
            if (finished) return;
            let K = 15;
            let currProgress = 1 - Math.exp(-(ticks / K));
            progress.set(currProgress);
            ticks += 1;
        }, 1000);

        return () => {
            clearInterval(interval);
        };
    });

    const progress = tweened(0, {
        duration: 400,
        easing: cubicOut,
    });

    $: if (finished === true) progress.set(1);
</script>

<div class="wrapper">
    <progress value={$progress} />
</div>

<style>
    .wrapper {
        width: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    progress {
        display: block;
        width: 50%;
        height: 2em;
        margin-top: 2em;
    }
</style>
