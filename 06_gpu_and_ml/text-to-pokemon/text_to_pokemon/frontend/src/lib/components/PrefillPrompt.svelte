<!--
  @component

  A line item in the prompt box's autocomplete dropdown list.
-->
<script>
    export let itemLabel;
    export let highlighted;
</script>

<li class="autocomplete-items" class:autocomplete-active={highlighted} on:click on:keyup>
    {@html itemLabel}
</li>

<style>
    li.autocomplete-items {
        color: black;
        list-style: none;
        z-index: 99;
        /*position the autocomplete items to be the same width as the container:*/
        top: 100%;
        left: 0;
        right: 0;
        padding: 10px;
        cursor: pointer;
    }

    li.autocomplete-items:hover {
        /*when hovering an item:*/
        color: DodgerBlue !important;
        font-weight: bold;
    }

    li.autocomplete-items:active {
        /*when navigating through the items using the arrow keys:*/
        color: DodgerBlue !important;
    }

    .autocomplete-active {
        /*when navigating through the items using the arrow keys:*/
        color: DodgerBlue !important;
    }
</style>
