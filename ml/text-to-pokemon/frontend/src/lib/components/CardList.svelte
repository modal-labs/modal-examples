<!--
  @component

  Simple component list, organizing a series of Card components
  into a grid.
-->



<script>
	import { activeCard } from "../stores/activeCard.js";

	let thisGrid;

	$: active = thisGrid && thisGrid.contains($activeCard);
</script>

<section class="card-grid" class:active bind:this={thisGrid}>
	<slot />
</section>

<style>
	.card-grid {
		display: grid;
		grid-template-columns: 1fr;
		grid-gap: 2vw;
		transform-style: preserve-3d;
		height: 100%;
		max-width: 1200px;
		margin: auto;
		margin-top: 4.5em;
		padding: 50px;
		padding-top: 0;
		position: relative;
	}

	.card-grid.active {
		z-index: 99;
	}

	@media screen and (min-width: 600px) {
		.card-grid {
			grid-template-columns: 1fr 1fr;
		}
	}

	@media screen and (min-width: 900px) {
		.card-grid {
			grid-template-columns: 1fr 1fr 1fr 1fr;
		}
	}
</style>
