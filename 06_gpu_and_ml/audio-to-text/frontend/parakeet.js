let mediaRecorder;
let isRecording = false;
let ws;
let audioContext;
let sourceNode;
let workletNode;

const recordButton = document.getElementById('recordButton');
const transcriptionDiv = document.getElementById('transcription');

// Constants for audio processing
const BUFFER_SIZE = 64000; 
const SAMPLE_RATE = 16000; // Target sample rate

// Audio worklet processor code
const workletCode = `
class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.bufferSize = ${BUFFER_SIZE};
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
`;

async function setupMediaRecorder() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                channelCount: 1, // Mono
                sampleRate: SAMPLE_RATE
            } 
        });
        
        // Set up Web Audio API
        audioContext = new window.AudioContext({
            sampleRate: SAMPLE_RATE
        });

        // Create and load the audio worklet
        const blob = new Blob([workletCode], { type: 'application/javascript' });
        const workletUrl = URL.createObjectURL(blob);
        await audioContext.audioWorklet.addModule(workletUrl);
        URL.revokeObjectURL(workletUrl);
        
        sourceNode = audioContext.createMediaStreamSource(stream);
        workletNode = new AudioWorkletNode(audioContext, 'audio-processor');
        
        workletNode.port.onmessage = (event) => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(event.data);
            }
        };
        
        sourceNode.connect(workletNode);
        workletNode.connect(audioContext.destination);
        
        return true;
    } catch (err) {
        console.error('Error accessing microphone:', err);
        alert('Error accessing microphone. Please make sure you have granted microphone permissions.');
        return false;
    }
}

async function connectWebSocket() {
    ws = new WebSocket('/ws');
    
    ws.onopen = () => {
        console.log('WebSocket connected');
    };
    
    ws.onmessage = (event) => {
        // Display the transcription
        const transcription = event.data;
        transcriptionDiv.textContent += transcription + '\n';
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
    
    ws.onclose = () => {
        console.log('WebSocket disconnected');
    };
}

recordButton.addEventListener('click', async () => {
    if (!isRecording) {
        // Start recording
        const success = await setupMediaRecorder();
        if (success) {
            await connectWebSocket();
            isRecording = true;
            recordButton.textContent = 'Stop Recording';
            recordButton.classList.add('recording');
            transcriptionDiv.textContent = ''; // Clear previous transcription
        }
    } else {
        // Stop recording
        isRecording = false;
        if (sourceNode) {
            sourceNode.disconnect();
        }
        if (workletNode) {
            workletNode.disconnect();
        }
        if (audioContext) {
            audioContext.close();
        }
        if (ws) {
            ws.close();
        }
        recordButton.textContent = 'Start Recording';
        recordButton.classList.remove('recording');
    }
});
