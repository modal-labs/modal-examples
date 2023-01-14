<script>
	import Card from "./lib/components/Card.svelte";
	import Footer from "./lib/components/Footer.svelte";
	import Pokeball from "./lib/components/Pokeball.svelte";
	import Generator from "./lib/components/Generator.svelte";
	import { onMount } from "svelte";

	onMount(() => {
		const $headings = document.querySelectorAll("h1,h2,h3");
		const $anchor = [...$headings].filter((el) => {
			const id = el.getAttribute("id")?.replace(/^.*?-/g, "");
			const hash = window.location.hash?.replace(/^.*?-/g, "");
			return id === hash;
		})[0];
		if ($anchor) {
			setTimeout(() => {
				$anchor.scrollIntoView();
			}, 100);
		}
	});
</script>

<main>
	<header>
		<div class="title">
			<Pokeball
				ballBottomColor="#BBFFAA"
				ballTopColor="#111"
				buttonFlashColor="green"
			/>
			<h1 id="⚓-top">StableDiffusion Pokémon Cards</h1>
		</div>

		<section class="intro" id="⚓-intro">
			<h3>Generate Pokémon from a text description.</h3>
			<p>
				This demo application uses a <a
					href="https://stability.ai/blog/stable-diffusion-public-release"
					target="_blank"
					rel="noopener noreferrer">Stable Diffusion</a
				>
				model trained by
				<a
					href="https://github.com/LambdaLabsML/lambda-diffusers"
					target="_blank"
					rel="noopener noreferrer">LambdaLabs</a
				> to make new Pokémon types.
			</p>
			<p>
				For added effect, further image transforms are applied in the
				backend to composite your make-believe Pokémon into a card, and
				the resulting images are rendered with <a
					href="https://github.com/simeydotme/pokemon-cards-css"
					target="_blank"
					rel="noopener noreferrer">simeydotme's</a
				> beautiful CSS work!
			</p>
		</section>

		<div class="showcase">
			<Card
				name=""
				img={"https://i.imgur.com/IczIg4r.png"}
				number={"123"}
				supertype={"Pokémon"}
				subtypes={["Basic", "V"]}
				rarity={"Rare Ultra"}
				showcase={true}
			/>
		</div>

		<section class="info">
			<ul>
				<li>
					👆🏼 Try <strong>clicking</strong> or <strong>tapping</strong>
					a card to take a closer look!
				</li>
				<li>
					Generations may take up-to two minutes in event of
					cold-start.
				</li>
			</ul>
		</section>
	</header>

	<section class="generator">
		<Generator />
	</section>
</main>

<Footer />

<style>
	main {
		/* Need more space for 'Built with Modal' footer badge. */
		padding-bottom: 150px;
	}

	@media screen and (min-width: 600px) {
		h1 {
			font-size: 2.5em;
		}

		h3 {
			font-size: 1.5em;
		}

		.title {
			display: flex;
			flex-direction: column;
			align-items: flex-start;
		}

		.title h1 {
			margin-top: 0.2em;
		}

		.intro p {
			font-size: 1.25em;
		}
	}
	.info {
		opacity: 0.8;
		border: solid 1px;
		border-radius: 0.5em;
	}

	.info ul {
		padding-inline-start: 30px;
	}

	.info ul li {
		line-height: 1.5em;
	}

	.generator {
		max-width: 100%;
	}
</style>
