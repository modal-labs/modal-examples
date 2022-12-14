<!--
  @component

  A Pokéball graphic, with customizable colors.
-->

<script>
    export let ballBottomColor = 'white';
	export let ballTopColor = 'red';
    export let buttonFlashColor = '#e74c3c';

	$: cssVarStyles = `--ball-top-color:${ballTopColor};--ball-bottom-color:${ballBottomColor};--button-flash-color:${buttonFlashColor}`;
</script>

<div class="pokeball" style="{cssVarStyles}">
    <div class="pokeball__button" />
</div>

<style>
    /* Poké Styles */
    .pokeball {
        position: relative;
        width: 50px;
        height: 50px;
        background: var(--ball-bottom-color, white);
        border: 2.5px solid #000;
        border-radius: 50%;
        overflow: hidden;
        box-shadow: inset -2.5px 2.5px 0 2.5px #ccc;
        animation: fall 0.25s ease-in-out,
            shake 1.25s cubic-bezier(0.36, 0.07, 0.19, 0.97) 3;
    }
    .pokeball::before,
    .pokeball::after {
        content: "";
        position: absolute;
    }
    .pokeball::before {
        background: var(--ball-top-color, red);
        left: 0;
        width: 100%;
        height: 50%;
    }
    .pokeball::after {
        top: calc(50% - 2.5px);
        width: 100%;
        height: 5px;
        left: 0;
        background: #000;
    }
    .pokeball__button {
        position: absolute;
        top: calc(50% - 7.5px);
        left: calc(50% - 7.5px);
        width: 15px;
        height: 15px;
        background: #7f8c8d;
        border: 2.5px solid #fff;
        border-radius: 50%;
        z-index: 10;
        box-shadow: 0 0 0 2.5px black;
        animation: blink 0.5s alternate 7;
    }

    /* Animation */
    @keyframes blink {
        from {
            background: #eee;
        }
        to {
            background: var(--button-flash-color, '#e74c3c');
        }
    }
    @keyframes shake {
        0% {
            transform: translate(0, 0) rotate(0);
        }
        20% {
            transform: translate(-10px, 0) rotate(-20deg);
        }
        30% {
            transform: translate(10px, 0) rotate(20deg);
        }
        50% {
            transform: translate(-10px, 0) rotate(-10deg);
        }
        60% {
            transform: translate(10px, 0) rotate(10deg);
        }
        100% {
            transform: translate(0, 0) rotate(0);
        }
    }
    @keyframes fall {
        0% {
            top: -200px;
        }
        60% {
            top: 0;
        }
        80% {
            top: -20px;
        }
        100% {
            top: 0;
        }
    }
</style>
