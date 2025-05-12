const videoElement = document.getElementById('video');
const canvasElement = document.getElementById('canvas');
const canvasCtx = canvasElement.getContext('2d');
const statusElement = document.getElementById('status');
const startButton = document.getElementById('start-button');
const switchCameraButton = document.getElementById('switch-camera');
const videoCanvasContainer = document.getElementById('video-canvas-container');

let cocoSsdModel = null;
let faceDetector = null;
let rafId = null; // requestAnimationFrame ID
let currentStream = null; // To keep track of the active media stream
let currentFacingMode = 'user'; // Start with front camera ('user') or 'environment' for rear
let modelsLoaded = false; // Flag to prevent reloading models
let isDetecting = false; // Flag to track if detection loop is active

// --- Configuration ---
const faceDetectionConfig = {
    modelType: 'short', // Options: 'short', 'full'
    maxFaces: 5,        // Detect up to 5 faces
};
const cocoSsdConfig = {
    base: 'lite_mobilenet_v2', // Options: 'lite_mobilenet_v2', 'mobilenet_v2'
};

// Drawing Styles
const DRAW_BOX_COLOR_FACE = '#00FF00'; // Bright Green
const DRAW_BOX_COLOR_PERSON = '#FF00FF'; // Bright Magenta (more distinct from green)
const DRAW_TEXT_COLOR = '#FFFFFF'; // White text
const DRAW_FONT = '16px "Segoe UI", sans-serif'; // Clear font
const DRAW_LINE_WIDTH = 3; // Thicker lines
const DRAW_SHADOW_COLOR = 'rgba(0, 0, 0, 0.5)'; // Shadow for text legibility
const DRAW_SHADOW_BLUR = 4;

// --- Helper Functions ---

function updateStatus(message) {
    console.log(`[Status] ${message}`);
    statusElement.innerText = message;
}

function stopDetectionLoop() {
    if (rafId) {
        cancelAnimationFrame(rafId);
        rafId = null;
    }
    isDetecting = false;
    console.log("Detection loop stopped.");
    // Clear canvas when stopping detection
    if (canvasCtx) {
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    }
}

function stopCurrentStream() {
    stopDetectionLoop(); // Ensure detection stops when stream stops
    if (currentStream) {
        currentStream.getTracks().forEach(track => {
            track.stop();
        });
        console.log("Camera stream stopped.");
    }
    videoElement.srcObject = null;
    currentStream = null;
}

async function setupCamera(facingMode) {
    updateStatus(`Requesting ${facingMode} camera...`);
    stopCurrentStream(); // Stop any existing stream first

    const constraints = {
        video: {
            facingMode: facingMode,
            // Optional: Request specific dimensions - be aware browsers might ignore/adjust
            // width: { ideal: 1280 }, // Request HD if available
            // height: { ideal: 720 }
        },
        audio: false,
    };

    try {
        currentStream = await navigator.mediaDevices.getUserMedia(constraints);
        videoElement.srcObject = currentStream;
        currentFacingMode = facingMode; // Update current mode after successful acquisition

        // Flip video and canvas horizontally if using front camera for a natural mirror effect
        const scaleX = (facingMode === 'user') ? -1 : 1;
        videoElement.style.transform = `scaleX(${scaleX})`;
        canvasElement.style.transform = `scaleX(${scaleX})`; // Keep canvas flip matching video flip

        return new Promise((resolve, reject) => { // Added reject
            videoElement.onloadedmetadata = () => {
                updateStatus('Camera ready.');
                // Adjust canvas size to match the video's actual resolution after stream starts
                canvasElement.width = videoElement.videoWidth;
                canvasElement.height = videoElement.videoHeight;
                console.log(`Camera resolution: ${videoElement.videoWidth}x${videoElement.videoHeight}`);
                resolve(videoElement);
            };
            videoElement.onerror = (e) => {
                const errorMsg = `Video element error: ${e.message || e}`;
                updateStatus(errorMsg);
                console.error(errorMsg, e);
                reject(new Error(errorMsg)); // Reject promise on video error
            }
        });
    } catch (error) {
        console.error('getUserMedia error!', error.name, error.message);
        let userMessage = `Error accessing ${facingMode} camera.`;
        if (error.name === "NotAllowedError") {
             userMessage = "Camera permission denied. Please refresh and grant permission.";
        } else if (error.name === "NotFoundError" || error.name === "DevicesNotFoundError") {
             userMessage = `${facingMode === 'user' ? 'Front' : 'Rear'} camera not found.`;
             // We'll let the calling function decide whether to try the other camera
        } else if (error.name === "NotReadableError") {
            userMessage = "Camera is already in use or hardware error.";
        } else {
            userMessage = `Camera error: ${error.name}`;
        }
        updateStatus(userMessage);
        throw error; // Re-throw the error for the caller to handle
    }
}

async function loadModels() {
    if (modelsLoaded) {
        console.log("Models already loaded.");
        return;
    }
    updateStatus('Loading detection models...');
    try {
        await tf.setBackend('webgl'); // Use WebGL backend for performance
        await tf.ready(); // Ensure backend is ready

        console.time("ModelLoad"); // Start timing model loading
        const [loadedCocoSsd, loadedFaceDetector] = await Promise.all([
            cocoSsd.load(cocoSsdConfig),
            faceDetection.createDetector(faceDetection.SupportedModels.MediaPipeFaceDetector, {
                runtime: 'tfjs', // Use tfjs runtime
                modelType: faceDetectionConfig.modelType,
                maxFaces: faceDetectionConfig.maxFaces
            })
        ]);
        console.timeEnd("ModelLoad"); // End timing

        cocoSsdModel = loadedCocoSsd;
        faceDetector = loadedFaceDetector;
        modelsLoaded = true;
        updateStatus('Models loaded successfully.');

    } catch (error) {
        updateStatus(`Error loading models: ${error.message}`);
        console.error('Model loading error!', error);
        modelsLoaded = false; // Ensure flag is false on error
        throw error; // Re-throw
    }
}

function drawBoundingBox(prediction, color, labelPrefix = '') {
    let box, scoreText = '', labelText = '';

    // Adapt based on model output structure
    if (prediction.class && prediction.bbox && prediction.bbox.length === 4) { // COCO-SSD
        box = prediction.bbox; // [x, y, width, height]
        labelText = prediction.class;
        scoreText = prediction.score ? (prediction.score * 100).toFixed(0) + '%' : ''; // Integer percentage
    } else if (prediction.box && prediction.box.xMin !== undefined) { // Face Detection
        const fb = prediction.box; // { xMin, yMin, width, height, xMax, yMax }
        box = [fb.xMin, fb.yMin, fb.width, fb.height];
        labelText = 'face';
        // Face model might have keypoints, but we are not using them here
        // Score might be available depending on runtime/model, e.g., prediction.keypoints[0]?.score
    } else {
        // console.warn("Skipping draw for invalid prediction structure:", prediction);
        return; // Cannot draw without a valid box
    }

    const [x, y, width, height] = box;

    // Prevent drawing boxes with non-positive dimensions which can cause errors
    if (width <= 0 || height <= 0) {
        // console.warn(`Skipping draw for invalid box dimensions: w=${width}, h=${height}`);
        return;
    }

    // Scale coordinates if canvas size differs from video intrinsic size
    const scaleX = canvasElement.width / videoElement.videoWidth;
    const scaleY = canvasElement.height / videoElement.videoHeight;
    const scaledX = x * scaleX;
    const scaledY = y * scaleY;
    const scaledWidth = width * scaleX;
    const scaledHeight = height * scaleY;

    // --- Draw the Bounding Box ---
    canvasCtx.strokeStyle = color;
    canvasCtx.lineWidth = DRAW_LINE_WIDTH;
    canvasCtx.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight);

    // --- Draw the Label Text with Background ---
    const fullLabel = `${labelPrefix}${labelText} ${scoreText}`.trim();
    canvasCtx.font = DRAW_FONT;
    const textMetrics = canvasCtx.measureText(fullLabel);
    const textWidth = textMetrics.width;
    // Estimate text height based on font size (adjust multiplier as needed)
    const textHeight = parseInt(DRAW_FONT, 10) * 1.2;

    // Calculate label position (slightly above the top-left corner)
    const labelX = scaledX + (DRAW_LINE_WIDTH / 2);
    // Position text background above the box, ensuring it doesn't go off-screen top
    const labelY = Math.max(scaledY - textHeight - 2, textHeight); // At least textHeight from top

    // Draw background rectangle for the text
    canvasCtx.fillStyle = color; // Use box color for background
    canvasCtx.fillRect(labelX - 2, labelY - textHeight + 2, textWidth + 4, textHeight);

    // Draw the text itself (with shadow for better contrast)
    canvasCtx.fillStyle = DRAW_TEXT_COLOR;
    canvasCtx.shadowColor = DRAW_SHADOW_COLOR;
    canvasCtx.shadowBlur = DRAW_SHADOW_BLUR;
    canvasCtx.fillText(fullLabel, labelX, labelY);

    // Reset shadow for other drawings
    canvasCtx.shadowColor = 'transparent';
    canvasCtx.shadowBlur = 0;
}


async function runDetectionFrame() {
    // Check conditions required for detection
    if (!modelsLoaded || !currentStream || videoElement.paused || videoElement.ended || !videoElement.srcObject || !isDetecting) {
        console.log("Detection prerequisites not met or detection stopped. Exiting frame.");
        stopDetectionLoop(); // Ensure loop variable is cleared if stopped
        return;
    }

     // Ensure canvas dimensions are synced if video dimensions change (can happen!)
    if (canvasElement.width !== videoElement.videoWidth || canvasElement.height !== videoElement.videoHeight) {
        canvasElement.width = videoElement.videoWidth;
        canvasElement.height = videoElement.videoHeight;
        console.log(`Canvas resized to: ${canvasElement.width}x${canvasElement.height}`);
    }

    // --- Start Detection ---
    try {
        // Use tf.tidy to automatically clean up intermediate tensors
        const predictions = await tf.tidy(async () => {
            // Detect faces first (often faster or primary interest)
            const facePredictions = await faceDetector.estimateFaces(videoElement, {
                flipHorizontal: false // Detection is on the raw stream *before* CSS flipping
            });

            // Detect objects using Coco-SSD
            const cocoPredictions = await cocoSsdModel.detect(videoElement);

            return { faces: facePredictions, objects: cocoPredictions };
        });


        // --- Draw Results ---
        // Clear previous drawings
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

        // Draw Coco-SSD Predictions (only 'person')
        predictions.objects.forEach(prediction => {
            if (prediction.class === 'person') {
                drawBoundingBox(prediction, DRAW_BOX_COLOR_PERSON);
            }
            // Example: Draw other objects if needed
            // else if (prediction.class === 'cell phone') {
            //     drawBoundingBox(prediction, '#00BCD4'); // Cyan for cell phone
            // }
        });

        // Draw Face Predictions
        predictions.faces.forEach(prediction => {
            drawBoundingBox(prediction, DRAW_BOX_COLOR_FACE);
        });

    } catch (error) {
        console.error("Error during detection frame:", error);
        updateStatus("Error during detection. Check console.");
        stopDetectionLoop(); // Stop detection on error
        // Optionally: Show start button again or attempt recovery
    }

    // --- Loop ---
    // Request the next frame if detection is still supposed to be active
    if (isDetecting) {
        rafId = requestAnimationFrame(runDetectionFrame);
    }
}

function startDetectionLoop() {
    if (isDetecting) {
        console.log("Detection loop already running.");
        return;
    }
    if (!modelsLoaded || !currentStream || videoElement.paused) {
        console.warn("Cannot start detection loop: prerequisites not met.");
        updateStatus("Cannot start detection (models/camera/video not ready).")
        return;
    }
    isDetecting = true;
    console.log("Starting detection loop.");
    updateStatus("Detection running...");
    runDetectionFrame(); // Start the first frame
}


async function startVideoAndDetection() {
    if (!currentStream) {
        updateStatus("Camera not ready.");
        console.error("Attempted to start video without a stream.");
        return;
    }
    // Start the video playback (required for detection)
    try {
        await videoElement.play();
        console.log("Video playback started.");
        startDetectionLoop(); // Start the detection loop *after* video starts playing
    } catch (playError) {
        updateStatus('Error playing video. User interaction might be required.');
        console.error("Video play error:", playError);
        // Provide feedback or a button to try playing again
    }
}

async function initialize(requestedFacingMode) {
    // Disable buttons during initialization
    startButton.disabled = true;
    switchCameraButton.disabled = true;
    startButton.style.display = 'none'; // Hide start button immediately
    updateStatus("Initializing...");

    try {
        // 1. Setup Camera (with fallback)
        try {
            await setupCamera(requestedFacingMode);
        } catch (cameraError) {
            // If the primary camera fails (e.g., not found or permission denied), try the other one only if it wasn't NotFound
            if (cameraError.name !== 'NotAllowedError' && cameraError.name !== 'NotReadableError') {
                 const fallbackFacingMode = (requestedFacingMode === 'user') ? 'environment' : 'user';
                 updateStatus(`Initial camera failed (${cameraError.name}), trying ${fallbackFacingMode} camera...`);
                 try {
                    await setupCamera(fallbackFacingMode);
                 } catch (fallbackError) {
                    // If fallback also fails, throw the *original* error for clarity
                    updateStatus(`Fallback camera also failed (${fallbackError.name}).`);
                    throw cameraError;
                 }
            } else {
                 throw cameraError; // Re-throw permission denied or other critical errors
            }
        }

        // 2. Load Models (only if not already loaded)
        await loadModels(); // This handles the modelsLoaded check internally

        // 3. Show elements and start detection
        videoCanvasContainer.style.display = 'block';
        switchCameraButton.style.display = 'inline-block';

        await startVideoAndDetection(); // Play video and begin detection loop

    } catch (error) {
        // Handle errors during init (camera permission denied, model load failure etc.)
        updateStatus(`Initialization failed: ${error.message || 'Unknown error'}`);
        console.error("Initialization failed:", error);
        stopCurrentStream(); // Ensure cleanup
        // Show start button again, hide video/switch button
        startButton.style.display = 'inline-block';
        videoCanvasContainer.style.display = 'none';
        switchCameraButton.style.display = 'none';
    } finally {
        // Re-enable buttons *if* they are visible
        if (startButton.style.display !== 'none') startButton.disabled = false;
        if (switchCameraButton.style.display !== 'none') switchCameraButton.disabled = false;
    }
}

// --- Event Listeners ---

startButton.addEventListener('click', () => {
    console.log("Start button clicked.");
    initialize(currentFacingMode); // Start with the default facing mode
});

switchCameraButton.addEventListener('click', async () => {
    if (switchCameraButton.disabled) return; // Prevent double clicks

    console.log("Switch camera button clicked.");
    const newFacingMode = (currentFacingMode === 'user') ? 'environment' : 'user';
    updateStatus(`Switching to ${newFacingMode} camera...`);
    switchCameraButton.disabled = true; // Disable while switching

    stopDetectionLoop(); // Stop old detection loop explicitly before changing stream

    try {
        await setupCamera(newFacingMode); // Setup the new camera
        await startVideoAndDetection(); // Restart video play and detection loop
    } catch (error) {
         updateStatus(`Failed to switch camera: ${error.message || 'Unknown error'}`);
         console.error("Camera switch failed:", error);
         stopCurrentStream(); // Cleanup on failure
         // Reset UI to initial state
         videoCanvasContainer.style.display = 'none';
         switchCameraButton.style.display = 'none';
         startButton.style.display = 'inline-block';
         startButton.disabled = false;
    } finally {
         // Re-enable button only if it's still supposed to be visible
         if (switchCameraButton.style.display !== 'none') {
            switchCameraButton.disabled = false;
         }
    }
});

// Optional: Pause detection when tab is hidden, resume when visible
document.addEventListener("visibilitychange", () => {
  if (!modelsLoaded || !currentStream) return; // Only act if initialized

  if (document.hidden) {
    if (isDetecting) {
        stopDetectionLoop(); // Pause loop
        console.log("Page hidden, pausing detection loop.");
    }
  } else {
    if (!isDetecting && videoElement.readyState >= 3) { // Check if video is ready
        console.log("Page visible, resuming detection loop.");
        startDetectionLoop(); // Resume loop
    }
  }
});

// --- Initial Page Load ---
updateStatus('Ready. Click "Start Detection".'); // Set initial status
console.log("Page loaded. Waiting for user interaction.");