let mediaRecorder;
let isRecording = false;
let ws;
let audioContext;
let sourceNode;
let workletNode;

const recordButton = document.getElementById("recordButton");
const transcriptionDiv = document.getElementById("transcription");

// Constants for audio processing
const BUFFER_SIZE = 16000;
const SAMPLE_RATE = 16000; // Target sample rate

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
  ws = new WebSocket(`/ws`);

  ws.onopen = () => {
    console.log("WebSocket connected");
  };

  ws.onmessage = (event) => {
    // Display the transcription
    const transcription = event.data;
    transcriptionDiv.textContent += transcription + "\n";
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
      recordButton.textContent = "Stop Transcription";
      recordButton.classList.add("recording");
      transcriptionDiv.textContent = ""; // Clear previous transcription
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
    recordButton.textContent = "Start Transcribing Mic";
    recordButton.classList.remove("recording");
  }
});
