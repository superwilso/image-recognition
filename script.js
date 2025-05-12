const videoElement = document.getElementById('video');
const canvasElement = document.getElementById('canvas');
const canvasCtx = canvasElement.getContext('2d');
const statusElement = document.getElementById('status');

let cocoSsdModel = null;
let faceDetector = null;
let rafId = null; // requestAnimationFrame ID

// --- Configuration ---
const faceDetectionConfig = {
  modelType: 'short', // or 'full' for potentially better accuracy but slower
  maxFaces: 5,        // Detect up to 5 faces
};
const cocoSsdConfig = {
  base: 'lite_mobilenet_v2', // 'mobilenet_v2' is more accurate but larger/slower
};

const DRAW_BOX_COLOR_FACE = '#00FF00'; // Green
const DRAW_BOX_COLOR_PERSON = '#FF0000'; // Red
const DRAW_TEXT_COLOR = '#FFFFFF';
const DRAW_FONT = '16px Arial';
const DRAW_LINE_WIDTH = 2;

// --- Helper Functions ---

function updateStatus(message) {
  console.log(message);
  statusElement.innerText = message;
}

async function setupCamera() {
  updateStatus('Requesting camera access...');
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user' }, // Use front camera
      audio: false,
    });
    videoElement.srcObject = stream;
    return new Promise((resolve) => {
      videoElement.onloadedmetadata = () => {
        updateStatus('Camera access granted.');
        resolve(videoElement);
      };
    });
  } catch (error) {
    updateStatus(`Error accessing camera: ${error.message}. Please grant permission.`);
    console.error('getUserMedia error!', error);
    throw error; // Re-throw to prevent further execution if camera fails
  }
}

async function loadModels() {
  updateStatus('Loading detection models...');
  try {
    // Set backend to WebGL for performance
    await tf.setBackend('webgl');
    await tf.ready(); // Ensure backend is ready

    const [loadedCocoSsd, loadedFaceDetector] = await Promise.all([
      cocoSsd.load(cocoSsdConfig),
      faceDetection.createDetector(faceDetection.SupportedModels.MediaPipeFaceDetector, {
          runtime: 'tfjs', // or 'mediapipe'
          // modelType: faceDetectionConfig.modelType, // Use defaults or specify
          maxFaces: faceDetectionConfig.maxFaces
      })
    ]);

    cocoSsdModel = loadedCocoSsd;
    faceDetector = loadedFaceDetector;

    updateStatus('Models loaded successfully.');
  } catch (error) {
    updateStatus(`Error loading models: ${error.message}`);
    console.error('Model loading error!', error);
    throw error; // Re-throw
  }
}

function drawBoundingBox(prediction, color) {
    if (!prediction || !prediction.box) return; // Handle potential nulls or missing box

    const [x, y, width, height] = prediction.box; // coco-ssd format [x, y, width, height]
    const score = prediction.score ? (prediction.score * 100).toFixed(1) + '%' : ''; // coco-ssd has score
    const label = prediction.class || 'face'; // coco-ssd has class, face detection doesn't directly

    // Scale coordinates if canvas size differs from video intrinsic size (important!)
    const scaleX = canvasElement.width / videoElement.videoWidth;
    const scaleY = canvasElement.height / videoElement.videoHeight;

    const scaledX = x * scaleX;
    const scaledY = y * scaleY;
    const scaledWidth = width * scaleX;
    const scaledHeight = height * scaleY;

    // Draw the bounding box
    canvasCtx.strokeStyle = color;
    canvasCtx.lineWidth = DRAW_LINE_WIDTH;
    canvasCtx.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight);

    // Draw the label background
    canvasCtx.fillStyle = color;
    const textWidth = canvasCtx.measureText(label + ' ' + score).width;
    const textHeight = parseInt(DRAW_FONT, 10); // Estimate height from font size
    canvasCtx.fillRect(scaledX, scaledY > textHeight ? scaledY - textHeight : scaledY, textWidth + 4, textHeight + 4);

    // Draw label text
    canvasCtx.fillStyle = DRAW_TEXT_COLOR;
    canvasCtx.font = DRAW_FONT;
    canvasCtx.fillText(label + ' ' + score, scaledX + 2, scaledY > textHeight ? scaledY : scaledY + textHeight);
}


async function runDetection() {
  if (!cocoSsdModel || !faceDetector || videoElement.paused || videoElement.ended) {
    // Ensure models are loaded and video is playing
    rafId = requestAnimationFrame(runDetection); // Keep checking
    return;
  }

  // Match canvas dimensions to video dimensions (important for correct drawing)
  if (canvasElement.width !== videoElement.videoWidth || canvasElement.height !== videoElement.videoHeight) {
      canvasElement.width = videoElement.videoWidth;
      canvasElement.height = videoElement.videoHeight;
  }

  // Detect objects using Coco-SSD
  const cocoPredictions = await cocoSsdModel.detect(videoElement);

  // Detect faces using Face Detection API
  const facePredictions = await faceDetector.estimateFaces(videoElement, {
    flipHorizontal: false // Usually false for front camera
  });

  // Clear previous drawings
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  // --- Draw Coco-SSD Predictions (only 'person') ---
  cocoPredictions.forEach(prediction => {
    if (prediction.class === 'person') {
        // CocoSSD provides box as [x, y, width, height] already
        drawBoundingBox(prediction, DRAW_BOX_COLOR_PERSON);
    }
  });

  // --- Draw Face Predictions ---
  facePredictions.forEach(prediction => {
      // Face Detection API provides box slightly differently
      const facePredictionFormatted = {
          box: [
              prediction.box.xMin,
              prediction.box.yMin,
              prediction.box.width,
              prediction.box.height
          ],
          // score: prediction.detectionScore // Might not be available or needed here
      };
      drawBoundingBox(facePredictionFormatted, DRAW_BOX_COLOR_FACE);
  });


  // Loop the detection
  rafId = requestAnimationFrame(runDetection);
}

// --- Main Execution ---

async function main() {
  try {
    await setupCamera(); // Wait for camera setup
    await loadModels();  // Wait for models to load

    // Make sure video is playing before starting detection
    videoElement.play().then(() => {
        updateStatus('Detection running...');
        runDetection(); // Start the detection loop
    }).catch(playError => {
        // Autoplay might be blocked, prompt user? For now, log it.
        updateStatus('Could not autoplay video. Interaction might be needed.');
        console.error("Video play error:", playError);
        // You might add a button here for the user to click to start video/detection
    });

  } catch (error) {
    // Errors during setup/loading are caught here
    updateStatus(`Initialization failed: ${error.message}`);
    console.error("Initialization failed:", error);
    // Stop any potential animation frame loop if it somehow started
    if (rafId) {
        cancelAnimationFrame(rafId);
    }
  }
}

// Start the application
main();