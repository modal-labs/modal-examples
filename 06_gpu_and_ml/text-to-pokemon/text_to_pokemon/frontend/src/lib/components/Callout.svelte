<script>
    export let type;

    if (type != "error" && type != "warn")
        console.warn("Callout.svelte only supports types {error, warn}");

    const iconBackgroundColor = type === "error" ? "#D72828" : "#DBCA44";
</script>

<div class="callout {type}">
    <svg
        height="24"
        style="overflow:visible;enable-background:new 0 0 32 32"
        viewBox="0 0 32 32"
        width="24"
        xml:space="preserve"
        ><g
            ><g id="Error_1_"
                ><g id="Error"
                    ><circle
                        cx="16"
                        cy="16"
                        id="BG"
                        r="16"
                        style="fill:{iconBackgroundColor};"
                    /><path
                        d="M14.5,25h3v-3h-3V25z M14.5,6v13h3V6H14.5z"
                        id="Exclamatory_x5F_Sign"
                        style="fill:#E6E6E6;"
                    /></g
                ></g
            ></g
        ></svg
    >
    <slot />
</div>

<style>
    svg {
        margin-right: 0.5em;
    }

    .callout {
        display: inline-flex;
        align-items: center;
        font-size: 1em;
        padding: 1em 1.5em;
        margin: 5% 10%;
        border: 1px solid #dce6e9;
        border-radius: 0.5em;
    }

    .error {
        background-color: #ff2e2e;
    }

    .warn {
        background-color: #111; /* Need to show white text clearly */
    }
</style>
