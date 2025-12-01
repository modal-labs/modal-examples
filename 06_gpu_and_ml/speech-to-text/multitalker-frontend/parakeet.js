let mediaRecorder;
let isRecording = false;
let ws;
let audioContext;
let sourceNode;
let workletNode;

const recordButton = document.getElementById("recordButton");
const transcriptionDiv = document.getElementById("transcription");

// Get WebSocket URL (hardcoded to Parakeet)
function getWebSocketUrl() {
    return window.WS_BASE_URL || "/ws";
}

// Transcription mode: 'concat' (default) or 'replace'
const TRANSCRIPTION_MODE = window.TRANSCRIPTION_MODE || 'concat';

// Constants for audio processing
const BUFFER_SIZE = 16000;
const SAMPLE_RATE = 16000; // Target sample rate

// ANSI color code to CSS color mapping
const ANSI_COLORS = {
  '30': '#000000', // black
  '31': '#cd3131', // red
  '32': '#0dbc79', // green
  '33': '#e5e510', // yellow
  '34': '#2472c8', // blue
  '35': '#bc3fbc', // magenta
  '36': '#11a8cd', // cyan
  '37': '#e5e5e5', // white
  '90': '#666666', // bright black (gray)
  '91': '#f14c4c', // bright red
  '92': '#23d18b', // bright green
  '93': '#f5f543', // bright yellow
  '94': '#3b8eea', // bright blue
  '95': '#d670d6', // bright magenta
  '96': '#29b8db', // bright cyan
  '97': '#ffffff', // bright white
};

// Convert ANSI color codes to HTML
function ansiToHtml(text) {
  // Replace ANSI escape sequences with HTML spans
  let html = text
    .replace(/\x1b\[0m/g, '</span>') // Reset
    .replace(/\x1b\[([0-9;]+)m/g, (match, codes) => {
      const codeList = codes.split(';');
      let styles = [];
      let isBold = false;
      let color = null;
      
      for (const code of codeList) {
        if (code === '0') {
          // Reset
          return '</span>';
        } else if (code === '1') {
          // Bold
          isBold = true;
        } else if (ANSI_COLORS[code]) {
          // Color
          color = ANSI_COLORS[code];
        }
      }
      
      if (isBold) styles.push('font-weight: bold');
      if (color) styles.push(`color: ${color}`);
      
      return styles.length > 0 ? `<span style="${styles.join('; ')}">` : '';
    });
  
  // Escape any remaining HTML and convert newlines to <br>
  html = html.replace(/\n/g, '<br>');
  
  return html;
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
  const wsUrl = getWebSocketUrl();
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log("WebSocket connected to:", wsUrl);
  };

  ws.onmessage = (event) => {
    // Display the transcription
    const transcription = event.data;
    const htmlTranscription = ansiToHtml(transcription);
    
    if (TRANSCRIPTION_MODE === 'replace') {
      transcriptionDiv.innerHTML = htmlTranscription;
    } else {
      transcriptionDiv.innerHTML += htmlTranscription;
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
    // Start microphone recording
    const success = await setupMediaRecorder();
    if (success) {
      await connectWebSocket();
      isRecording = true;
      recordButton.textContent = "Stop Transcription";
      recordButton.classList.add("recording");
      transcriptionDiv.innerHTML = ""; // Clear previous transcription
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
