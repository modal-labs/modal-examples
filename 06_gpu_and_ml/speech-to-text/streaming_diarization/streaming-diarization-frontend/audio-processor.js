class AudioProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();

    const processorOptions = options.processorOptions || {};
    this.sourceSampleRate = processorOptions.sourceSampleRate || 48000;
    this.targetSampleRate = processorOptions.targetSampleRate || 16000;
    this.downsampleRatio = this.sourceSampleRate / this.targetSampleRate;

    this.bufferSize = 512; // 1 second of audio at 16kHz
    this.buffer = new Float32Array(this.bufferSize);
    this.bufferIndex = 0;

    this.sampleAccumulator = 0;
    this.sampleCounter = 0;
  }

  process(inputs, outputs) {
    const input = inputs[0];
    const channel = input[0];

    if (!channel) return true;

    // simple downsampling by averaging
    for (let i = 0; i < channel.length; i++) {
      this.sampleAccumulator += channel[i];
      this.sampleCounter++;

      if (this.sampleCounter >= this.downsampleRatio) {
        const downsampledValue = this.sampleAccumulator / this.sampleCounter;
        this.buffer[this.bufferIndex++] = downsampledValue;

        this.sampleAccumulator = 0;
        this.sampleCounter = 0;

        if (this.bufferIndex >= this.bufferSize) {
          const pcmData = new Int16Array(this.bufferSize);
          for (let j = 0; j < this.bufferSize; j++) {
            pcmData[j] = Math.max(
              -32768,
              Math.min(32767, Math.round(this.buffer[j] * 32768))
            );
          }

          this.port.postMessage(pcmData.buffer);

          this.bufferIndex = 0;
        }
      }
    }

    return true;
  }
}

registerProcessor("audio-processor", AudioProcessor);
