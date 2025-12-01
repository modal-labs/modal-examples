let mediaRecorder;
let isRecording = false;
let ws;
let audioContext;
let sourceNode;
let workletNode;

const recordButton = document.getElementById("recordButton");
    const transcriptionDiv = document.getElementById("transcription"); // Kept for reference but not used in new UI

    // Speaker colors map to match the HTML
    const speakerColors = ['blue', 'red', 'green', 'orange'];

// Constants for audio processing
const BUFFER_SIZE = 16000;
const SAMPLE_RATE = 16000; // Target sample rate

// Get WebSocket URL (hardcoded to Parakeet)
function getWebSocketUrl() {
  return window.WS_BASE_URL || "/ws";
}

async function setupMediaRecorder() {
  try {
    // First get the microphone stream
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1, // Mono
      },
    });

    // Then set up Web Audio API with native sample rate
    audioContext = new window.AudioContext();

    // Load the audio worklet
    await audioContext.audioWorklet.addModule("/static/audio-processor.js");

    // Create source node and worklet node
    sourceNode = audioContext.createMediaStreamSource(stream);
    workletNode = new AudioWorkletNode(audioContext, "audio-processor", {
      processorOptions: {
        targetSampleRate: SAMPLE_RATE,
        sourceSampleRate: audioContext.sampleRate,
      },
    });

    workletNode.port.onmessage = (event) => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(event.data);
      }
    };

    sourceNode.connect(workletNode);
    workletNode.connect(audioContext.destination);

    return true;
  } catch (err) {
    console.error("Error in setupMediaRecorder:", err);
    if (err.name === "NotAllowedError") {
      alert(
        "Microphone access was denied. Please allow microphone access and try again."
      );
    } else if (err.name === "NotFoundError") {
      alert("No microphone found. Please connect a microphone and try again.");
    } else {
      alert("Error accessing microphone: " + err.message);
    }
    return false;
  }
}

async function connectWebSocket() {
  ws = new WebSocket(getWebSocketUrl());

  ws.onopen = () => {
    console.log("WebSocket connected");
  };

  ws.onmessage = (event) => {
    // Parse the probabilities
    try {
        const probs = JSON.parse(event.data);
        
        // Update each speaker indicator
        probs.forEach((prob, index) => {
            const speakerBox = document.getElementById(`speaker-${index}`);
            if (speakerBox) {
                const indicator = speakerBox.querySelector('.speaker-indicator');
                const probLabel = speakerBox.querySelector('.speaker-prob');
                
                // Update opacity based on probability
                // Minimum opacity 0.1 so it's faintly visible, max 1.0
                const opacity = Math.max(0.1, prob);
                indicator.style.opacity = opacity;
                
                // Update text label
                probLabel.textContent = prob.toFixed(2);
            }
        });
    } catch (e) {
        console.error("Error parsing message:", e);
    }
  };

  ws.onerror = (error) => {
    console.error("WebSocket error:", error);
  };

  ws.onclose = () => {
    console.log("WebSocket disconnected");
  };
}

recordButton.addEventListener("click", async () => {
  if (!isRecording) {
    // Start recording
    const success = await setupMediaRecorder();
    if (success) {
      await connectWebSocket();
      isRecording = true;
      recordButton.textContent = "Stop Diarizing";
      recordButton.classList.add("recording");
      // transcriptionDiv.textContent = ""; // Clear previous transcription
    }
  } else {
    // Stop recording
    isRecording = false;
    if (sourceNode) {
      sourceNode.disconnect();
      sourceNode = null;
    }
    if (workletNode) {
      workletNode.disconnect();
      workletNode = null;
    }
    if (audioContext) {
      audioContext.close();
      audioContext = null;
    }
    if (ws) {
      ws.close();
      ws = null;
    }
    recordButton.textContent = "Start Diarizing";
    recordButton.classList.remove("recording");
  }
});
