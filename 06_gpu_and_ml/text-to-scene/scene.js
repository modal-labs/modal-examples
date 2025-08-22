let scene, camera, renderer, container;

function initScene() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x222222);
  container = document.getElementById("canvas-container");
  camera = new THREE.PerspectiveCamera(
    75,
    container.clientWidth / container.clientHeight,
    0.1,
    1000
  );
  renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true,
  });
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer.domElement);

  setupMouseControls();
}

// Movement variables
const moveSpeed = 0.01;
const maxDistance = 0.3; // Maximum movement distance from origin
const keys = {
  w: false,
  a: false,
  s: false,
  d: false,
};
let isMouseDown = false;
let previousMousePosition = {
  x: 0,
  y: 0,
};
let rotationSpeed = 0.0005;
let isRotating = true;
let sceneCenter = new THREE.Vector3();
let animationId = null;
let isAtLimit = false;

// Event listeners for keyboard controls
document.addEventListener("keydown", (event) => {
  switch (event.key.toLowerCase()) {
    case "w":
      keys.w = true;
      break;
    case "a":
      keys.a = true;
      break;
    case "s":
      keys.s = true;
      break;
    case "d":
      keys.d = true;
      break;
  }
  // Restart animation if it was stopped
  if (!animationId && (keys.w || keys.a || keys.s || keys.d)) {
    animate();
  }
});

document.addEventListener("keyup", (event) => {
  switch (event.key.toLowerCase()) {
    case "w":
      keys.w = false;
      break;
    case "a":
      keys.a = false;
      break;
    case "s":
      keys.s = false;
      break;
    case "d":
      keys.d = false;
      break;
  }
});

// Mouse drag for camera-relative rotation
function setupMouseControls() {
  renderer.domElement.addEventListener("mousedown", (event) => {
    isMouseDown = true;
    previousMousePosition = {
      x: event.clientX,
      y: event.clientY,
    };
    // Prevent default to avoid text selection
    event.preventDefault();
  });

  document.addEventListener("mouseup", () => {
    isMouseDown = false;
  });

  document.addEventListener("mousemove", (event) => {
    if (isMouseDown) {
      const deltaMove = {
        x: event.clientX - previousMousePosition.x,
        y: event.clientY - previousMousePosition.y,
      };

      const quaternion = new THREE.Quaternion();

      // Horizontal rotation (Y-axis)
      const yQuaternion = new THREE.Quaternion();
      yQuaternion.setFromAxisAngle(
        new THREE.Vector3(0, 1, 0),
        -deltaMove.x * 0.002
      );
      quaternion.multiply(yQuaternion);

      // Vertical rotation (X-axis)
      const xQuaternion = new THREE.Quaternion();
      xQuaternion.setFromAxisAngle(
        new THREE.Vector3(1, 0, 0),
        -deltaMove.y * 0.002
      );
      quaternion.multiply(xQuaternion);

      // Apply rotation
      camera.quaternion.multiply(quaternion);

      previousMousePosition = {
        x: event.clientX,
        y: event.clientY,
      };
    }
  });

  // Prevent context menu on canvas
  renderer.domElement.addEventListener("contextmenu", (event) => {
    event.preventDefault();
  });
}

// initialize all loader
const loader = new THREE.PLYLoader();
const dracoLoader = new THREE.DRACOLoader();
dracoLoader.setDecoderPath(
  "https://cdn.jsdelivr.net/npm/three@0.132.2/examples/js/libs/draco/"
);
let loadedCount = 0;
let totalFiles = 0;

// Load all layers of a scene
function loadSceneFromURLs(urls) {
  document.getElementById("loading").style.display = "block";
  totalFiles = urls.length;
  loadedCount = 0;

  // Clear existing models from scene
  scene.children.slice().forEach((child) => {
    if (child instanceof THREE.Mesh) {
      if (child.geometry) child.geometry.dispose();
      if (child.material) child.material.dispose();
      scene.remove(child);
    }
  });

  // Load each PLY file
  urls.forEach((url) => {
    fetch(url)
      .then((response) => response.arrayBuffer())
      .then((data) => {
        try {
          const geometry = loader.parse(data);
          const material = new THREE.MeshBasicMaterial({
            side: THREE.DoubleSide,
            vertexColors: true,
          });
          const mesh = new THREE.Mesh(geometry, material);
          mesh.rotateX(-Math.PI / 2);
          mesh.rotateZ(-Math.PI / 2);
          scene.add(mesh);

          loadedCount++;
          if (loadedCount === totalFiles) {
            document.getElementById("loading").style.display = "none";
            positionCamera();
            isRotating = true;
            document.getElementById("rotate-toggle").textContent =
              "Pause Rotation";
            if (!animationId) animate();
          }
        } catch (error) {
          console.error("Error loading PLY file:", error);
          loadedCount++;
          if (loadedCount === totalFiles) {
            document.getElementById("loading").style.display = "none";
            if (
              scene.children.filter((c) => c instanceof THREE.Mesh).length > 0
            ) {
              positionCamera();
              isRotating = true;
              document.getElementById("rotate-toggle").textContent =
                "Pause Rotation";
              if (!animationId) animate();
            }
          }
        }
      })
      .catch((error) => {
        console.error("Error fetching PLY file:", error);
        loadedCount++;
        if (loadedCount === totalFiles) {
          document.getElementById("loading").style.display = "none";
        }
      });
  });
}

// Load available scenes and group them by timestamp (generation)
async function loadSceneList() {
  try {
    const response = await fetch("/list-scenes");
    const scenes = await response.json();

    const sceneList = document.getElementById("scene-list");
    sceneList.innerHTML = "";

    const sceneGroups = {};
    scenes.forEach((scene) => {
      const timestamp = scene.timestamp;
      if (!sceneGroups[timestamp]) {
        sceneGroups[timestamp] = [];
      }
      sceneGroups[timestamp].push(scene);
    });

    // Create buttons for each timestamp (generation), sorted by most recent first
    Object.keys(sceneGroups)
      .sort((a, b) => b.localeCompare(a)) // Sort timestamps in reverse order (newest first)
      .forEach((timestamp) => {
        const group = sceneGroups[timestamp];
        const item = document.createElement("span");
        item.className = "scene-item";
        // Format timestamp for display (YYYYMMDD_HHMMSS -> readable)
        const month = timestamp.slice(4, 6);
        const day = timestamp.slice(6, 8);
        const hour = timestamp.slice(9, 11);
        const minute = timestamp.slice(11, 13);
        const displayName = `${month}/${day} ${hour}:${minute}`;
        item.textContent = displayName;
        item.title = `Generated: ${timestamp}`;
        item.onclick = () => {
          const urls = group.map((s) => s.url);
          loadSceneFromURLs(urls);
        };
        sceneList.appendChild(item);
      });

    // Load the most recent scene group automatically if available
    const sortedGroups = Object.keys(sceneGroups).sort((a, b) =>
      b.localeCompare(a)
    );
    if (sortedGroups.length > 0) {
      const mostRecent = sortedGroups[0];
      const urls = sceneGroups[mostRecent].map((s) => s.url);
      loadSceneFromURLs(urls);
    }
  } catch (error) {
    console.error("Error loading scene list:", error);
  }
}

// Generate new scene
document
  .getElementById("generate-btn")
  .addEventListener("click", async function () {
    const prompt = document.getElementById("prompt-input").value.trim();
    const negative_prompt = document
      .getElementById("negative-prompt-input")
      .value.trim();
    const labels_fg1 = document.getElementById("fg1-input").value.trim();
    const labels_fg2 = document.getElementById("fg2-input").value.trim();
    const classes = document.getElementById("class-select").value;

    if (!prompt) {
      alert("Please enter a prompt");
      return;
    }

    this.disabled = true;
    document.getElementById("loading").style.display = "block";
    document.getElementById("loading").textContent =
      "Generating 3D world... This may take a few minutes.";

    try {
      const response = await fetch("/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          negative_prompt,
          labels_fg1,
          labels_fg2,
          classes,
        }),
      });

      if (!response.ok) {
        throw new Error("Generation failed");
      }

      // Reload scene list
      await loadSceneList();

      document.getElementById("loading").textContent = "Loading...";
    } catch (error) {
      console.error("Error generating scene:", error);
      alert("Error generating scene: " + error.message);
    } finally {
      this.disabled = false;
      document.getElementById("loading").style.display = "none";
    }
  });

document.getElementById("refresh-btn").addEventListener("click", function () {
  loadSceneList();
});

// Position camera reset
function positionCamera() {
  // Initial camera position
  scene.rotation.y = 0;
  camera.position.set(0, 0, 0);
  camera.lookAt(0, 0, -10);
}

// Control button events
document.getElementById("rotate-toggle").addEventListener("click", function () {
  isRotating = !isRotating;
  this.textContent = isRotating ? "Pause Rotation" : "Start Rotation";
});

document.getElementById("reset-view").addEventListener("click", function () {
  positionCamera();
  isAtLimit = false;
  isRotating = true;
  if (!animationId) {
    animate();
  }
});

// Function to limit movement distance
function limitMovement(position) {
  const distance = Math.sqrt(position.x * position.x + position.z * position.z);
  if (distance > maxDistance) {
    const ratio = maxDistance / distance;
    position.x *= ratio;
    position.z *= ratio;
    return true; // Reached limit
  }
  return false; // Not at limit
}

// Animation loop
function animate() {
  // Calculate movement direction based on camera orientation
  if (keys.w || keys.a || keys.s || keys.d) {
    // Get camera's forward vector (negative Z-axis)
    const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(
      camera.quaternion
    );
    // Get camera's right vector (positive X-axis)
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion);

    // Ignore Y component to keep movement horizontal
    forward.y = 0;
    right.y = 0;
    forward.normalize();
    right.normalize();

    // Calculate movement direction
    const movement = new THREE.Vector3();

    if (keys.w) movement.add(forward); // Forward
    if (keys.s) movement.sub(forward); // Backward
    if (keys.a) movement.sub(right); // Left
    if (keys.d) movement.add(right); // Right

    // Only normalize if there's actual movement
    if (movement.length() > 0) {
      movement.normalize().multiplyScalar(moveSpeed);
    }

    // Store current Y position
    const currentY = camera.position.y;

    // Apply movement
    camera.position.add(movement);

    // Restore Y position to keep movement horizontal
    camera.position.y = currentY;

    // Check if reached movement limit
    isAtLimit = limitMovement(camera.position);
  }

  // Auto-rotation if enabled
  if (isRotating && scene.children.some((c) => c instanceof THREE.Mesh)) {
    scene.rotation.y += rotationSpeed;
  }

  // Create a target position slightly in front of the camera
  const targetPosition = new THREE.Vector3();
  targetPosition.copy(camera.position);
  targetPosition.add(
    new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion)
  );

  // Smoothly look at a point slightly in front of the camera
  camera.lookAt(targetPosition);

  renderer.render(scene, camera);

  // keep running
  animationId = requestAnimationFrame(animate);
}

// Window resize handler
window.addEventListener("resize", function () {
  const container = document.getElementById("canvas-container");
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
});

// Initialize and load scene list when page loads
window.addEventListener("DOMContentLoaded", function () {
  initScene();
  animate(); // Start animation loop
  loadSceneList();
});
