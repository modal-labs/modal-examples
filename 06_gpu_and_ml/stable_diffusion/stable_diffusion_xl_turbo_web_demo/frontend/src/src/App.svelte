<script lang="ts">
  import {
    Github,
    Loader,
    Upload,
    Undo,
    Redo,
    ArrowDownToLine,
    ArrowLeftSquare,
    MoveUpRight,
  } from "lucide-svelte";
  import { onMount } from "svelte";
  import paper from "paper";
  import { throttle, debounce } from "throttle-debounce";

  import BackgroundGradient from "$lib/BackgroundGradient.svelte";
  import modalLogoWithText from "$lib/assets/logotype.svg";
  import Paint from "$lib/Paint.svelte";
  import defaultInputImage from "$lib/assets/mocha_outside.png";

  let value: string = "studio ghibli, 8k, wolf";
  let imgInput: HTMLImageElement;
  let imgOutput: HTMLImageElement;
  let canvasDrawLayer: HTMLCanvasElement;

  let isImageUploaded = false;
  let firstImageGenerated = false;
  
  // we track lastUpdatedAt so that expired requests don't overwrite the latest
  let lastUpdatedAt = 0;

  // used for undo/redo functionality 
  let outputImageHistory: string[] = [];
  $: currentOutputImageIndex = -1;

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
    /* 
      Setup paper.js for canvas which is a layer above our input image.
      Paper is used for drawing/paint functionality.
    */
    paper.setup(canvasDrawLayer);
    const tool = new paper.Tool();

    let path: paper.Path;

    tool.onMouseDown = (event: paper.ToolEvent) => {
      path = new paper.Path();
      path.strokeColor = new paper.Color(paint);
      path.strokeWidth = radiusByBrushSize[brushSize] * 4;
      path.add(event.point);

      throttledGenerateOutputImage();
    };

    tool.onMouseDrag = (event: paper.ToolEvent) => {
      path.add(event.point);

      throttledGenerateOutputImage();
    };

    imgInput.onload = () => {
      resizeImage(imgInput);
      isImageUploaded = true;

      // kick off an inference on first image load so output image is populated as well
      // otherwise it will be empty
      if (!firstImageGenerated) {
        GenerateOutputImage();
        firstImageGenerated = true;
      }
    };

    imgOutput.onload = () => {
      resizeImage(imgOutput);
    };
    imgInput.src = defaultInputImage;
  });

  // Our images need to be sized 320x320 for both input and output
  // This is important because we combine the canvas layer with the image layer
  // so the pixels need to matchup.
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

      GenerateOutputImage();
    }
  }

  // combines the canvas with the input image so that the 
  // generated image contains edits made by paint brush
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

  const throttledGenerateOutputImage = throttle(
    250,
    () => {
      GenerateOutputImage();
    },
    { noLoading: false, noTrailing: false },
  );

  const debouncedGenerateOutputImage = debounce(
    100,
    () => {
      GenerateOutputImage();
    },
    { atBegin: false },
  );

  const movetoCanvas = () => {
    imgInput.src = imgOutput.src;
  };

  const downloadImage = () => {
    let a = document.createElement("a");
    a.href = imgOutput.src;
    a.download = "modal-generated-image.jpeg";
    a.click();
  };

  const redoOutputImage = () => {
    if (currentOutputImageIndex > 0 && outputImageHistory.length > 1) {
      currentOutputImageIndex -= 1;
      imgOutput.src = outputImageHistory[currentOutputImageIndex];
    }
  };

  const undoOutputImage = () => {
    if (currentOutputImageIndex < outputImageHistory.length - 1) {
      currentOutputImageIndex += 1;
      imgOutput.src = outputImageHistory[currentOutputImageIndex];
    }
  };

  const GenerateOutputImage = () => {
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
          outputImageHistory = [text, ...outputImageHistory];
          if (outputImageHistory.length > 10) {
            outputImageHistory = outputImageHistory.slice(0, 10);
          }
          imgOutput.src = text;
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
      <div class="button py-2 px-5 font-medium">
        <Github size={16} />View Code
      </div>
    </div>

    <div class="mt-6">
      <div class="mb-6 font-semibold">Input</div>
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
      <h3 class="mb-6 font-semibold">Prompt</h3>
      <input
        class="rounded-lg border border-white/20 bg-white/10 py-4 px-6 outline-none w-full"
        bind:value
        on:input={debouncedGenerateOutputImage}
      />
    </div>

    <div class="mt-6 flex items-center">
      <div class="pr-7 border-r border-white/10 flex items-center">
        <div>
          <div class="pb-6">
            <div class="mb-2 font-medium">Canvas</div>
            <div>Draw on the image to generate a new one</div>
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

        <div class="ml-6 mt-4">
          <Paint
            {paint}
            {brushSize}
            on:clearCanvas={() => {
              paper.project.activeLayer.removeChildren();
              paper.view.update();
              GenerateOutputImage();
            }}
            on:setPaint={setPaint}
            on:setBrushSize={setBrushSize}
          />
        </div>
      </div>

      <div class="pl-7 flex">
        <div>
          <div class="pb-6">
            <div class="mb-2 flex items-center gap-1 font-medium">
              Output
              {#if isLoading}
                <Loader size={14} class="animate-spin" />
              {/if}
            </div>
            <div>Generated Image</div>
          </div>

          <img
            alt="output"
            bind:this={imgOutput}
            class="w-[320px] h-[320px] bg-[#D9D9D9]"
            class:hidden={!firstImageGenerated}
          />
        </div>
        <div class="flex justify-between ml-6 items-center">
          <div class="flex flex-col gap-4 mb-[108px]">
            <div class="btns-container justify-space-between">
              <button class="text-xs flex gap-1.5" on:click={undoOutputImage}
                ><Undo size={16} />Back</button
              >
              <div class="w-[1px] h-4 bg-white/10" />
              <button class="text-xs flex gap-1.5" on:click={redoOutputImage}
                ><Redo size={16} />Next</button
              >
            </div>
            <button class="text-xs btns-container" on:click={movetoCanvas}>
              <ArrowLeftSquare size={16} />Move to Canvas
            </button>

            <button class="text-xs btns-container" on:click={downloadImage}>
              <ArrowDownToLine size={16} /> Download
            </button>
          </div>
        </div>
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
    <a
      href="https://modal.com/docs/guide"
      class="button px-5 py-[6px] font-medium"
    >
      Get Started <MoveUpRight size={16} />
    </a>
  </div>
</main>

<style lang="postcss">
  .container {
    @apply bg-white/10 border border-white/20 rounded-lg p-6 max-w-screen-lg;
  }

  .btns-container {
    @apply flex items-center gap-2.5 py-2 px-3 border rounded-[10px] border-white/5 bg-white/10;
    width: 144px;
  }

  .button {
    @apply border border-primary bg-primary/20 rounded-lg justify-center items-center flex gap-2 cursor-pointer;
  }

  .modal-logo {
    width: 108px;
    height: 32px;
  }
</style>
