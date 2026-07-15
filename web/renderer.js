// Global State
let scene, camera, renderer, controls;
let faceMesh;
let metadata = null;
let meanPositions = null;
let identityBasis = null; // Float32Array (converted from float16)
let expressionBasis = null; // Float32Array (converted from float16)
let faceIndices = null;

// Animation & Audio State
let audioSync = new AudioSync();
let animController = new AnimationController();
let visemeTimeline = null;
let currentEmotionCoeffs = null;
let activeUtterancePlaying = false;

// Float16 to Float32 decoder
function decodeFloat16(uint16) {
    const s = (uint16 & 0x8000) >> 15;
    const e = (uint16 & 0x7C00) >> 10;
    const f = uint16 & 0x03FF;
    
    if (e === 0) {
        if (f === 0) {
            return s ? -0.0 : 0.0;
        } else {
            return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
        }
    } else if (e === 31) {
        return f ? NaN : (s ? -Infinity : Infinity);
    }
    
    return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

function float16BufferToFloat32Array(arrayBuffer) {
    const uint16View = new Uint16Array(arrayBuffer);
    const float32Array = new Float32Array(uint16View.length);
    for (let i = 0; i < uint16View.length; i++) {
        float32Array[i] = decodeFloat16(uint16View[i]);
    }
    return float32Array;
}

// Load GNM Buffers
async function loadGnmBuffers() {
    const overlay = document.getElementById("loading-overlay");
    try {
        console.log("Loading metadata...");
        const metaRes = await fetch("/data/web/metadata.json");
        metadata = await metaRes.json();
        
        console.log("Loading mean positions (float32)...");
        const meanRes = await fetch("/data/web/mean_positions.bin");
        meanPositions = new Float32Array(await meanRes.arrayBuffer());
        
        console.log("Loading face indices (uint32)...");
        const faceRes = await fetch("/data/web/face_indices.bin");
        faceIndices = new Uint32Array(await faceRes.arrayBuffer());

        console.log("Loading identity basis (float16)...");
        const idRes = await fetch("/data/web/identity_basis.bin");
        identityBasis = float16BufferToFloat32Array(await idRes.arrayBuffer());

        console.log("Loading expression basis (float16)...");
        const exprRes = await fetch("/data/web/expression_basis.bin");
        expressionBasis = float16BufferToFloat32Array(await exprRes.arrayBuffer());

        console.log("Buffers loaded and decoded.");
        console.log("Diagnostics - meanPositions length:", meanPositions.length);
        console.log("Diagnostics - faceIndices length:", faceIndices.length);
        console.log("Diagnostics - identityBasis length:", identityBasis.length);
        console.log("Diagnostics - expressionBasis length:", expressionBasis.length);
        overlay.style.opacity = 0;
        setTimeout(() => overlay.style.display = "none", 500);

        // Load Viseme Table
        await animController.loadVisemeTable();
        
        // Initialize Scene
        initScene();
    } catch (e) {
        console.error("Failed to load GNM buffers", e);
        overlay.querySelector("p").innerText = "Loading Failed! Please ensure server is running.";
    }
}

// Initialize Three.js Scene
function initScene() {
    const container = document.getElementById("canvas-container");
    
    // Scene setup
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x050508);
    
    // Camera
    camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 100);
    camera.position.set(0, 0.23, 0.45); // Centered on the face
    
    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    container.appendChild(renderer.domElement);
    
    // Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI / 2 + 0.1;
    controls.minDistance = 0.2;
    controls.maxDistance = 1.0;
    controls.target.set(0, 0.23, 0.03); // Focus on the center of the head
    
    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.15);
    scene.add(ambientLight);
    
    const keyLight = new THREE.DirectionalLight(0xffffff, 0.85);
    keyLight.position.set(0.5, 0.5, 0.5);
    scene.add(keyLight);
    
    const fillLight = new THREE.DirectionalLight(0xa5b4fc, 0.45); // cool blue fill
    fillLight.position.set(-0.5, 0.2, 0.3);
    scene.add(fillLight);
    
    const rimLight = new THREE.DirectionalLight(0xffffff, 0.5);
    rimLight.position.set(0, 0.8, -0.8);
    scene.add(rimLight);

    // Build GNM Head Geometry
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(meanPositions), 3));
    geometry.setIndex(new THREE.BufferAttribute(faceIndices, 1));
    geometry.computeVertexNormals();

    // Material with sleek, premium metallic/clay look
    const material = new THREE.MeshStandardMaterial({
        color: 0x222530,
        roughness: 0.5,
        metalness: 0.1,
        flatShading: false,
        side: THREE.DoubleSide
    });

    faceMesh = new THREE.Mesh(geometry, material);
    scene.add(faceMesh);

    // Initial neutral coefficient setup
    currentEmotionCoeffs = new Float32Array(metadata.expression_dim);
    
    // Auto-fetch default emotion on startup
    updateCurrentEmotion();

    // Start render loop
    requestAnimationFrame(renderLoop);
}

// Perform real-time mesh deformation
// Uses active-only sparse summation to achieve extreme efficiency
const currentPos = new Float32Array(17821 * 3);

function deformMesh(expressionCoeffs) {
    if (!faceMesh || !metadata) return 0;
    
    // Copy template positions to starting buffer
    currentPos.set(meanPositions);

    // 1. Gather active expression parameters (threshold check to keep loop tiny)
    const activeExprIndices = [];
    const activeExprWeights = [];
    for (let i = 0; i < metadata.expression_dim; i++) {
        const w = expressionCoeffs[i];
        if (Math.abs(w) > 1e-4) {
            activeExprIndices.push(i);
            activeExprWeights.push(w);
        }
    }

    const activeCount = activeExprIndices.length;
    const vCount = metadata.num_vertices;

    // 2. Perform additive deformation on GPU-ready array
    for (let i = 0; i < activeCount; i++) {
        const exprIdx = activeExprIndices[i];
        const weight = activeExprWeights[i];
        const offset = exprIdx * vCount * 3;

        for (let j = 0; j < vCount * 3; j++) {
            currentPos[j] += expressionBasis[offset + j] * weight;
        }
    }

    // 3. Update Three.js buffer
    const posAttr = faceMesh.geometry.attributes.position;
    posAttr.array.set(currentPos);
    posAttr.needsUpdate = true;
    faceMesh.geometry.computeVertexNormals();

    return activeCount;
}

// Render Loop
let lastTime = performance.now();
let frameCount = 0;
let fpsVal = document.getElementById("fps-val");
let msVal = document.getElementById("ms-val");
let activeVal = document.getElementById("active-val");

function renderLoop(time) {
    requestAnimationFrame(renderLoop);
    
    const t0 = performance.now();

    // 1. Query timeline & update mesh vertices
    let activeShapesCount = 0;
    if (activeUtterancePlaying && visemeTimeline) {
        const timeS = audioSync.getCurrentTime();
        
        // Update slider time text
        document.getElementById("audio-time").innerText = `${timeS.toFixed(2)}s / ${audioSync.getDuration().toFixed(2)}s`;
        
        const speechCoeffs = animController.getSpeechCoefficients(timeS, visemeTimeline);
        const blended = animController.blend(speechCoeffs, currentEmotionCoeffs);
        activeShapesCount = deformMesh(blended);
    } else {
        // Just keep emotion shape
        const speechCoeffs = new Array(182).fill(0.0);
        const blended = animController.blend(speechCoeffs, currentEmotionCoeffs);
        activeShapesCount = deformMesh(blended);
    }

    // 2. Render
    controls.update();
    renderer.render(scene, camera);

    const t1 = performance.now();
    
    // Performance stats
    frameCount++;
    if (t1 - lastTime >= 1000) {
        fpsVal.innerText = frameCount;
        frameCount = 0;
        lastTime = t1;
    }
    msVal.innerText = (t1 - t0).toFixed(1);
    activeVal.innerText = activeShapesCount;
}

// UI Handlers & API Sync
const speakBtn = document.getElementById("speak-btn");
const playPauseBtn = document.getElementById("play-pause-btn");
const textInput = document.getElementById("speech-text");
const emotionSelect = document.getElementById("emotion-select");
const intensityRange = document.getElementById("intensity-range");
const intensityVal = document.getElementById("intensity-val");

intensityRange.addEventListener("input", (e) => {
    intensityVal.innerText = e.target.value;
    updateCurrentEmotion();
});

emotionSelect.addEventListener("change", updateCurrentEmotion);

function updateCurrentEmotion() {
    if (!metadata || !animController.visemeTable) return;
    
    const name = emotionSelect.value;
    const intensity = parseFloat(intensityRange.value);
    
    speakBtn.disabled = true; // wait for fetch
    
    fetch("/api/emotion", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name, intensity: intensity })
    })
    .then(r => r.json())
    .then(data => {
        currentEmotionCoeffs = new Float32Array(data.coefficients);
        speakBtn.disabled = false;
    })
    .catch(e => {
        console.error("Failed to load emotion coefficients from server", e);
        speakBtn.disabled = false;
    });
}

// Speak Button clicked
speakBtn.addEventListener("click", async () => {
    const text = textInput.value.trim();
    if (!text) return;

    speakBtn.disabled = true;
    speakBtn.innerText = "Synthesizing...";
    playPauseBtn.disabled = true;

    try {
        const response = await fetch("/api/speak", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                text: text,
                emotion: emotionSelect.value,
                intensity: parseFloat(intensityRange.value)
            })
        });

        const data = await response.json();
        
        // Load synthesized audio
        await audioSync.loadAudioFromBase64(data.audio_base64);
        
        // Store viseme timeline
        visemeTimeline = data.visemes;
        
        // Update UI
        playPauseBtn.disabled = false;
        playPauseBtn.innerText = "Pause";
        
        // Play
        audioSync.play();
        activeUtterancePlaying = true;
        
        audioSync.onEnded = () => {
            playPauseBtn.innerText = "Play Audio";
            activeUtterancePlaying = false;
        };

    } catch (e) {
        console.error("Synthesize API failed", e);
    } finally {
        speakBtn.disabled = false;
        speakBtn.innerText = "Speak";
    }
});

// Play/Pause Button
playPauseBtn.addEventListener("click", () => {
    if (audioSync.isPlaying) {
        audioSync.pause();
        playPauseBtn.innerText = "Play Audio";
    } else {
        audioSync.play();
        playPauseBtn.innerText = "Pause";
    }
});

// Handle window resizing
window.addEventListener('resize', () => {
    const container = document.getElementById("canvas-container");
    if (!camera || !renderer) return;
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
});

// Boot GNM Web Engine
window.onload = loadGnmBuffers;
