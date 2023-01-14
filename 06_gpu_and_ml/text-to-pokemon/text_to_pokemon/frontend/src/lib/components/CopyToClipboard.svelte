<script lang="ts">
    import { createEventDispatcher } from "svelte";
    export let text;

    const dispatch = createEventDispatcher();
    const copy = () => {
        navigator.clipboard.writeText(text).then(
            () => dispatch("copy", text),
            (e) => dispatch("fail")
        );
    };
</script>

<slot {copy} />
