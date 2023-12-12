<script lang="ts">
  import { Github, Loader, Upload, Undo, Redo, ArrowDownToLine, ArrowLeftSquare } from "lucide-svelte";
  import { onMount } from "svelte";
  import paper from "paper";
  import { throttle, debounce } from "throttle-debounce";

  import BackgroundGradient from "$lib/BackgroundGradient.svelte";
  import modalLogoWithText from "$lib/assets/logotype.svg";
  import Paint from "$lib/Paint.svelte";
  import defaultInputImage from "$lib/assets/mocha_outside.png";

  let value: string = "star wars, 8k, disney";
  let imgInput: HTMLImageElement;
  let imgOutput: HTMLImageElement;
  let canvasDrawLayer: HTMLCanvasElement;

  let isImageUploaded = false;
  let firstImageGenerated = false;
  let isDrawing = false;
  let lastUpdatedAt = 0;
  $: isLoading = false;

  $: brushSize = "sm";
  $: paint = "#2613FD"; // can be hex or "eraser" for removing paint
  const radiusByBrushSize: Record<string, number> = {
    xs: 1,
    sm: 2,
    md: 3,
    lg: 4,
  };
  const setPaint = (e: CustomEvent<string>) => {
    paint = e.detail;
  };
  const setBrushSize = (e: CustomEvent<string>) => {
    brushSize = e.detail;
  };

  onMount(() => {
    paper.setup(canvasDrawLayer);
    const tool = new paper.Tool();

    let path: paper.Path;

    tool.onMouseDown = (event: paper.ToolEvent) => {
      path = new paper.Path();
      path.strokeColor = new paper.Color(paint);
      path.strokeWidth = radiusByBrushSize[brushSize] * 4;
      path.add(event.point);

      isDrawing = true;
      throttledGetNextFrameLoop();
    };

    tool.onMouseDrag = (event: paper.ToolEvent) => {
      path.add(event.point);

      isDrawing = true;
      throttledGetNextFrameLoop();
    };

    imgInput.onload = () => {
      resizeImage(imgInput);
      isImageUploaded = true;
    };
    imgInput.src = defaultInputImage;
  });

  const resizeImage = (img: HTMLImageElement) => {
    let newWidth;
    let newHeight;
    if (img.width > img.height) {
      const aspectRatio = img.height / img.width;
      newWidth = 320;
      newHeight = newWidth * aspectRatio;
    } else {
      const aspectRatio = img.width / img.height;
      newHeight = 320;
      newWidth = newHeight * aspectRatio;
    }

    img.style.width = `${newWidth}px`;
    img.style.height = `${newHeight}px`;
  };

  function loadImage(e: Event) {
    const target = e.target as HTMLInputElement;
    if (!target || !target.files) return;
    const file = target.files[0];

    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        if (e?.target?.result && typeof e.target.result === "string") {
          isImageUploaded = true;
          imgInput.src = e?.target.result;
          resizeImage(imgInput);
        }
      };

      reader.readAsDataURL(file);

      getNextFrameLoop();
    }
  }

  function getCombinedImageData() {
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = 320;
    tempCanvas.height = 320;
    const tempCtx = tempCanvas.getContext("2d");
    if (!tempCtx) return;

    tempCtx.drawImage(imgInput, 0, 0, 320, 320);
    tempCtx.drawImage(canvasDrawLayer, 0, 0, 320, 320);
    return tempCanvas.toDataURL("image/jpeg");
  }

  const throttledGetNextFrameLoop = throttle(
    250,
    () => {
      getNextFrameLoop();
    },
    { noLoading: false, noTrailing: false },
  );

  const debouncedGetNextFrameLoop = debounce(
    100,
    () => {
      getNextFrameLoop();
    },
    { atBegin: false },
  );

  const movetoCanvas = () => {
    imgInput.src = imgOutput.src;
  }

  const downloadImage = () => {
    let a = document.createElement("a");
    a.href = imgOutput.src;
    a.download = "modal-generated-image.jpeg";
    a.click();
  }

  const getNextFrameLoop = () => {
    if (!isDrawing && firstImageGenerated) {
      return;
    }

    isLoading = true;
    const data = getCombinedImageData();
    const sentAt = new Date().getTime();
    fetch(window.INFERENCE_BASE_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: data,
        prompt: value,
      }),
    })
      .then((res) => res.text())
      .then((text) => {
        if (sentAt > lastUpdatedAt) {
          imgOutput.src = text;
          resizeImage(imgOutput);
          lastUpdatedAt = sentAt;
        }

        firstImageGenerated = true;
      })
      .finally(() => (isLoading = false));
  };
</script>

<BackgroundGradient />
<main class="flex flex-col items-center pt-16">
  <div class="container">
    <div class="flex items-center justify-between">
      <div>
        <h2 class="text-2xl font-medium">SDXL Turbo Image2Image</h2>
        <div class="font-sm">example description</div>
      </div>
      <div class="button py-2 px-5"><Github size={16} />View Code</div>
    </div>

    <div class="mt-6">
      <div class="mb-6 font-medium">Input</div>
      <input
        type="file"
        accept="image/*"
        id="file-upload"
        hidden
        on:change={loadImage}
      />
      <label for="file-upload" class="button flex-col w-[446px] h-[76px]">
        <div class="flex items-center gap-2 font-medium">
          <Upload size={16} />
          Upload Image
        </div>
        <span>PNG, JPEG</span>
      </label>
    </div>

    <div class="mt-6">
      <h3 class="mb-6">Prompt</h3>
      <input
        class="rounded-lg border border-white/20 bg-white/10 py-4 px-6 outline-none w-full"
        bind:value
        on:input={debouncedGetNextFrameLoop}
      />
    </div>

    <div class="mt-6 flex items-center">
      <div class="pr-[72px] border-r border-white/10 flex items-center">
        <div>
          <div>
            <div class="mb-2">Canvas</div>
            <div>Draw on the image and generate a new one</div>
          </div>

          <img
            alt="input"
            bind:this={imgInput}
            class="absolute w-[320px] h-[320px] bg-[#D9D9D9] pointer-events-none z-[-1]"
            class:hidden={!isImageUploaded}
          />
          <canvas
            bind:this={canvasDrawLayer}
            width={320}
            height={320}
            class="w-[320px] h-[320px] z-1"
          />
        </div>

        <div class="ml-6">
          <Paint
            {paint}
            {brushSize}
            on:clearCanvas={() => {
              paper.project.activeLayer.removeChildren();
              paper.view.update();
              getNextFrameLoop();
            }}
            on:setPaint={setPaint}
            on:setBrushSize={setBrushSize}
          />
        </div>
      </div>

      <div class="pl-[72px]">
        <div>
          <div class="mb-2 flex items-center gap-1">
            Output
            {#if isLoading}
              <Loader size={14} class="animate-spin" />
            {/if}
          </div>
          <div class="flex justify-between">
            <div>Generated Image</div>
            <div class="flex">
              <button on:click={movetoCanvas}><ArrowLeftSquare size={16}/>Move to Canvas</button>
              <button><Undo size={16} /></button>
              <button><Redo size={16} /></button>
              <button on:click={downloadImage}><ArrowDownToLine size={16} /></button>
            </div>
          </div>

        </div>

        <img
          alt="output"
          bind:this={imgOutput}
          class="w-[320px] h-[320px] bg-[#D9D9D9]"
          class:hidden={!firstImageGenerated}
        />
        {#if !firstImageGenerated}
          <div class="w-[320px] h-[320px] bg-[#D9D9D9]" />
        {/if}
      </div>
    </div>
  </div>

  <div class="w-full container flex my-4 justify-between items-center">
    <div class="flex items-center gap-2">
      Built with <img
        class="modal-logo"
        alt="Modal logo"
        src={modalLogoWithText}
      />
    </div>
    <a href="https://modal.com/docs/guide" class="button px-5 py-[6px]"
      >Get Started</a
    >
  </div>
</main>

<style lang="postcss">
  .container {
    @apply bg-white/10 border border-white/20 rounded-lg p-6 max-w-screen-lg;
  }

  .button {
    @apply border border-primary bg-primary/20 rounded-lg justify-center items-center flex gap-2 cursor-pointer;
  }

  .modal-logo {
    width: 108px;
    height: 32px;
  }
</style>
