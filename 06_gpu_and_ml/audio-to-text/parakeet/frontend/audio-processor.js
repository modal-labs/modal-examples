class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.bufferSize = 16000;
        this.buffer = new Float32Array(this.bufferSize);
        this.bufferIndex = 0;
    }

    process(inputs, outputs) {
        const input = inputs[0];
        const channel = input[0];

        if (!channel) return true;

        // Fill our buffer
        for (let i = 0; i < channel.length; i++) {
            this.buffer[this.bufferIndex++] = channel[i];

            // When buffer is full, send it
            if (this.bufferIndex >= this.bufferSize) {
                // Convert to Int16Array
                const pcmData = new Int16Array(this.bufferSize);
                for (let j = 0; j < this.bufferSize; j++) {
                    pcmData[j] = Math.max(-32768, Math.min(32767, Math.round(this.buffer[j] * 32768)));
                }
                
                // Send the buffer
                this.port.postMessage(pcmData.buffer);
                
                // Reset buffer
                this.bufferIndex = 0;
            }
        }

        return true;
    }
}

registerProcessor('audio-processor', AudioProcessor); 