<!--
  @component

  A simple and small Modal branding footer. Can be copy-pasted into other Svelte-based examples,
  and there's a React version of this component in the 'Modal Whisper Transcriber' example.
-->

<div class="footer-wrapper">
    <a href="https://modal.com" target="_blank" rel="noopener noreferrer">
        <footer class="shadow-lg">
            <span class="p-1 text-md"><strong>built with</strong></span>
            <img
                alt="modal logo"
                class="h-12 w-24"
                src="../img/modal-logo.svg"
            />
        </footer>
    </a>
</div>

<style>
    .footer-wrapper {
        width: 100%;
        position: fixed;
        display: flex;
        justify-content: center;
        bottom: 0px;
    }

    footer {
        display: flex;
        flex-direction: row;
        align-items: center;
        padding: 0.25em;
        background: rgb(39 39 42); /* bg-zinc-800 */
        margin-bottom: 1.5em;
        color: white;
        text-align: center;
        border-radius: 0.5rem;
        text-decoration: none;
    }

    a {
        text-decoration: none;
        cursor: pointer;
    }

    a:hover {
        color: white;
    }

    span {
        margin-left: 0.25em;
        margin-right: 0.25em;
    }

    footer img {
        height: 3rem;
        width: 6rem;
    }

    .shadow-lg {
        --tw-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1),
            0 4px 6px -4px rgb(0 0 0 / 0.1);
        --tw-shadow-colored: 0 10px 15px -3px var(--tw-shadow-color),
            0 4px 6px -4px var(--tw-shadow-color);
        box-shadow: var(--tw-ring-offset-shadow, 0 0 #0000),
            var(--tw-ring-shadow, 0 0 #0000), var(--tw-shadow);
    }
</style>
