let mediaRecorder;
let isRecording = false;
let ws;
let audioContext;
let sourceNode;
let processorNode;

const recordButton = document.getElementById('recordButton');
const transcriptionDiv = document.getElementById('transcription');

// Constants for audio processing
const CHUNK_SIZE = 64000; // Same as in the Python code
const SAMPLE_RATE = 16000; // Target sample rate

async function setupMediaRecorder() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                channelCount: 1, // Mono
                sampleRate: SAMPLE_RATE,
                sampleSize: 16
            } 
        });
        
        // Set up Web Audio API
        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: SAMPLE_RATE
        });
        
        sourceNode = audioContext.createMediaStreamSource(stream);
        processorNode = audioContext.createScriptProcessor(4096, 1, 1);
        
        processorNode.onaudioprocess = (e) => {
            if (!isRecording) return;
            
            const inputData = e.inputBuffer.getChannelData(0);
            // Convert Float32Array to Int16Array
            const pcmData = new Int16Array(inputData.length);
            for (let i = 0; i < inputData.length; i++) {
                // Convert float to 16-bit signed integer
                pcmData[i] = Math.max(-32768, Math.min(32767, Math.round(inputData[i] * 32768)));
            }
            
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(pcmData.buffer);
            }
        };
        
        sourceNode.connect(processorNode);
        processorNode.connect(audioContext.destination);
        
        return true;
    } catch (err) {
        console.error('Error accessing microphone:', err);
        alert('Error accessing microphone. Please make sure you have granted microphone permissions.');
        return false;
    }
}

async function connectWebSocket() {
    // Get the WebSocket URL from the server    
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
        if (processorNode) {
            processorNode.disconnect();
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
