<script lang="ts">
    import { Eraser } from "lucide-svelte";

    import smallPaintIcon from "$lib/assets/sm-paint-icon.svg";
    import extraSmallPaintIcon from "$lib/assets/xs-paint-icon.svg";
    import mediumPaintIcon from "$lib/assets/md-paint-icon.svg";
    import largePaintIcon from "$lib/assets/lg-paint-icon.svg";
    import { createEventDispatcher } from "svelte";

    const dispatch = createEventDispatcher();

    export let paint: string;
    export let brushSize: string;

    const handleClearCanvas = () => {
        dispatch("clearCanvas");
    };

    const handleSetPaint = (paint: string) => {
        dispatch("setPaint", paint);
    };

    const handleSetBrushSize = (size: string) => {
        dispatch("setBrushSize", size);
    };
</script>

<div class="flex flex-col">
    <div class="container">
        <div class="flex justify-between w-full">
            <button
                class="flex items-center gap-2.5 w-full text-sm"
                on:click={() => handleClearCanvas()}
            >
                <Eraser size={12} /> Clear
            </button>
        </div>
    </div>

    <div class="container mt-3">
        <div class="flex justify-between w-full">
            <button
                on:click={() => handleSetPaint("#2613FD")}
                class="circle bg-[#2613FD]"
                class:circle-active={paint === "#2613FD"}
            ></button>
            <button
                on:click={() => handleSetPaint("#5E57B3")}
                class="circle bg-[#5E57B3]"
                class:circle-active={paint === "#5E57B3"}
            ></button>
        </div>
        <div class="flex justify-between w-full">
            <button
                on:click={() => handleSetPaint("#FD139F")}
                class="circle bg-[#FD139F]"
                class:circle-active={paint === "#FD139F"}
            ></button>
            <button
                on:click={() => handleSetPaint("#A4FD13")}
                class="circle bg-[#A4FD13]"
                class:circle-active={paint === "#A4FD13"}
            ></button>
        </div>
        <div class="flex justify-between w-full">
            <button
                on:click={() => handleSetPaint("#FD1321")}
                class="circle bg-[#FD1321]"
                class:circle-active={paint === "#FD1321"}
            ></button>
            <button
                on:click={() => handleSetPaint("#F90")}
                class="circle bg-[#F90]"
                class:circle-active={paint === "#F90"}
            ></button>
        </div>
        <div class="flex justify-between w-full">
            <button
                on:click={() => handleSetPaint("black")}
                class="circle bg-black"
                class:circle-active={paint === "black"}
            ></button>
            <button
                on:click={() => handleSetPaint("white")}
                class="circle bg-white"
                class:circle-active={paint === "white"}
            ></button>
        </div>

        <div
            class="flex border-white/10 border-t pt-2.5 justify-between w-full"
        >
            <button on:click={() => handleSetBrushSize("xs")}>
                <img
                    class="brush"
                    class:brush-active={brushSize === "xs"}
                    src={extraSmallPaintIcon}
                    alt="extrasmall paint icon"
                />
            </button>
            <button on:click={() => handleSetBrushSize("sm")}>
                <img
                    class="brush"
                    class:brush-active={brushSize === "sm"}
                    src={smallPaintIcon}
                    alt="small paint icon"
                />
            </button>
        </div>
        <div class="flex justify-between w-full">
            <button on:click={() => handleSetBrushSize("md")}>
                <img
                    class="brush"
                    class:brush-active={brushSize === "md"}
                    src={mediumPaintIcon}
                    alt="medium paint icon"
                />
            </button>
            <button on:click={() => handleSetBrushSize("lg")}>
                <img
                    class="brush"
                    class:brush-active={brushSize === "lg"}
                    src={largePaintIcon}
                    alt="large paint icon"
                />
            </button>
        </div>
    </div>
</div>

<style lang="postcss">
    .container {
        @apply flex flex-col items-center gap-2.5 py-2 px-3 border rounded-[10px] border-white/5 bg-white/10;
        width: 88px;
    }

    .circle {
        @apply w-5 h-5 rounded-full;
    }

    .circle-active {
        @apply border-2 border-primary;
    }

    .brush {
        filter: invert(100%) sepia(6%) saturate(7487%) hue-rotate(293deg)
            brightness(103%) contrast(118%);
    }

    .brush-active {
        filter: invert(84%) sepia(34%) saturate(768%) hue-rotate(51deg)
            brightness(97%) contrast(92%);
    }
</style>
