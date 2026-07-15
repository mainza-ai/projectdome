let scene, camera, renderer, controls;
let faceMesh;
let metadata = null;
let meanPositions = null;
let identityBasis = null;
let expressionBasis = null;
let faceIndices = null;
let skinningWeights = null;
let jointRegressor = null;
let vertexBodyParts = null;

let neckRotation = [0, 0, 0];
let headRotation = [0, 0, 0];
let leftEyeRotation = [0, 0, 0];
let rightEyeRotation = [0, 0, 0];
let currentIdentityCoeffs = new Float32Array(253).fill(0.0);

let audioSync = new AudioSync();
let animController = new AnimationController();
let visemeTimeline = null;
let currentEmotionCoeffs = null;
let activeUtterancePlaying = false;

const BODY_PART = { SKIN: 0, SCLERAS: 1, IRISES: 2, TEETH: 3, TONGUE: 4, GUM: 5, MOUTH_SOCK: 6 };

function decodeFloat16(uint16) {
    const s = (uint16 & 0x8000) >> 15;
    const e = (uint16 & 0x7C00) >> 10;
    const f = uint16 & 0x03FF;
    if (e === 0) return f === 0 ? (s ? -0.0 : 0.0) : (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    if (e === 31) return f ? NaN : (s ? -Infinity : Infinity);
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

async function loadGnmBuffers() {
    const overlay = document.getElementById("loading-overlay");
    try {
        console.log("Loading metadata...");
        const metaRes = await fetch("/data/web/metadata.json");
        metadata = await metaRes.json();

        console.log("Loading mean positions...");
        const meanRes = await fetch("/data/web/mean_positions.bin");
        meanPositions = new Float32Array(await meanRes.arrayBuffer());

        console.log("Loading face indices...");
        const faceRes = await fetch("/data/web/face_indices.bin");
        faceIndices = new Uint32Array(await faceRes.arrayBuffer());

        console.log("Loading identity basis (float16)...");
        const idRes = await fetch("/data/web/identity_basis.bin");
        identityBasis = float16BufferToFloat32Array(await idRes.arrayBuffer());

        console.log("Loading expression basis (float16)...");
        const exprRes = await fetch("/data/web/expression_basis.bin");
        expressionBasis = float16BufferToFloat32Array(await exprRes.arrayBuffer());

        console.log("Loading skinning weights...");
        const skinRes = await fetch("/data/web/skinning_weights.bin");
        skinningWeights = new Float32Array(await skinRes.arrayBuffer());

        console.log("Loading joint regressor...");
        const regRes = await fetch("/data/web/joint_regressor.bin");
        jointRegressor = new Float32Array(await regRes.arrayBuffer());

        try {
            const vpRes = await fetch("/data/web/vertex_body_parts.bin");
            vertexBodyParts = new Int32Array(await vpRes.arrayBuffer());
            console.log("Loaded vertex body parts for material assignment.");
        } catch (e) {
            vertexBodyParts = null;
        }

        overlay.style.opacity = 0;
        setTimeout(() => overlay.style.display = "none", 500);

        await animController.loadVisemeTable();
        await loadBlinkCoefficients();
        initScene();
    } catch (e) {
        console.error("Failed to load GNM buffers", e);
        overlay.querySelector("p").innerText = "Loading Failed! Ensure server is running.";
    }
}

function getVertexColors() {
    if (!vertexBodyParts) return null;
    const colors = new Float32Array(metadata.num_vertices * 3);
    const palette = {
        [BODY_PART.SKIN]: [0.78, 0.70, 0.62],
        [BODY_PART.SCLERAS]: [0.95, 0.95, 0.97],
        [BODY_PART.IRISES]: [0.30, 0.45, 0.65],
        [BODY_PART.TEETH]: [0.92, 0.90, 0.85],
        [BODY_PART.TONGUE]: [0.80, 0.50, 0.50],
        [BODY_PART.GUM]: [0.75, 0.50, 0.50],
        [BODY_PART.MOUTH_SOCK]: [0.35, 0.20, 0.20],
    };
    for (let i = 0; i < metadata.num_vertices; i++) {
        const part = vertexBodyParts[i];
        const c = palette[part] || palette[BODY_PART.SKIN];
        colors[i * 3] = c[0];
        colors[i * 3 + 1] = c[1];
        colors[i * 3 + 2] = c[2];
    }
    return colors;
}

function initScene() {
    const container = document.getElementById("canvas-container");
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf5f5f7);

    camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 100);
    camera.position.set(0, 0.23, 0.45);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI / 2 + 0.1;
    controls.minDistance = 0.2;
    controls.maxDistance = 1.0;
    controls.target.set(0, 0.23, 0.03);

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.25);
    scene.add(ambientLight);
    const keyLight = new THREE.DirectionalLight(0xffffff, 0.85);
    keyLight.position.set(0.5, 0.5, 0.5);
    scene.add(keyLight);
    const fillLight = new THREE.DirectionalLight(0xa5b4fc, 0.4);
    fillLight.position.set(-0.5, 0.2, 0.3);
    scene.add(fillLight);
    const rimLight = new THREE.DirectionalLight(0xffffff, 0.45);
    rimLight.position.set(0, 0.8, -0.8);
    scene.add(rimLight);

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(meanPositions), 3));
    geometry.setIndex(new THREE.BufferAttribute(faceIndices, 1));

    const vertColors = getVertexColors();
    if (vertColors) {
        geometry.setAttribute('color', new THREE.BufferAttribute(vertColors, 3));
    }

    geometry.computeVertexNormals();

    const material = new THREE.MeshStandardMaterial({
        roughness: 0.4,
        metalness: 0.1,
        flatShading: false,
        side: THREE.DoubleSide,
        vertexColors: !!vertColors,
    });

    faceMesh = new THREE.Mesh(geometry, material);
    scene.add(faceMesh);

    currentEmotionCoeffs = new Float32Array(metadata.expression_dim);
    updateCurrentEmotion();
    requestAnimationFrame(renderLoop);
}

const currentPos = new Float32Array(17821 * 3);

function deformMesh(expressionCoeffs) {
    if (!faceMesh || !metadata) return 0;
    currentPos.set(meanPositions);
    const vCount = metadata.num_vertices;

    const activeIdIndices = [], activeIdWeights = [];
    for (let i = 0; i < metadata.identity_dim; i++) {
        const w = currentIdentityCoeffs[i];
        if (Math.abs(w) > 1e-4) { activeIdIndices.push(i); activeIdWeights.push(w); }
    }
    for (let i = 0; i < activeIdIndices.length; i++) {
        const idIdx = activeIdIndices[i], weight = activeIdWeights[i];
        const offset = idIdx * vCount * 3;
        for (let j = 0; j < vCount * 3; j++) currentPos[j] += identityBasis[offset + j] * weight;
    }

    const activeExprIndices = [], activeExprWeights = [];
    for (let i = 0; i < metadata.expression_dim; i++) {
        const w = expressionCoeffs[i];
        if (Math.abs(w) > 1e-4) { activeExprIndices.push(i); activeExprWeights.push(w); }
    }
    for (let i = 0; i < activeExprIndices.length; i++) {
        const exprIdx = activeExprIndices[i], weight = activeExprWeights[i];
        const offset = exprIdx * vCount * 3;
        for (let j = 0; j < vCount * 3; j++) currentPos[j] += expressionBasis[offset + j] * weight;
    }

    const joints = [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()];
    for (let j = 0; j < 4; j++) {
        let jx = 0, jy = 0, jz = 0;
        const regOffset = j * vCount;
        for (let v = 0; v < vCount; v++) {
            const w = jointRegressor[regOffset + v];
            if (Math.abs(w) > 1e-6) {
                jx += w * currentPos[v * 3]; jy += w * currentPos[v * 3 + 1]; jz += w * currentPos[v * 3 + 2];
            }
        }
        joints[j].set(jx, jy, jz);
    }

    const gazeAutoCheckbox = document.getElementById("gaze-auto");
    const activeRotations = [neckRotation, headRotation, [...leftEyeRotation], [...rightEyeRotation]];

    if (gazeAutoCheckbox && gazeAutoCheckbox.checked && camera) {
        const target = camera.position;
        const headYaw = headRotation[1] + neckRotation[1];
        const headPitch = headRotation[0] + neckRotation[0];
        for (let j = 2; j <= 3; j++) {
            const dir = new THREE.Vector3().copy(target).sub(joints[j]).normalize();
            const yaw = Math.atan2(dir.x, dir.z);
            const pitch = Math.asin(-dir.y);
            let localYaw = yaw - headYaw;
            let localPitch = pitch - headPitch;
            localYaw = Math.max(-25 * Math.PI / 180.0, Math.min(25 * Math.PI / 180.0, localYaw));
            localPitch = Math.max(-15 * Math.PI / 180.0, Math.min(15 * Math.PI / 180.0, localPitch));
            activeRotations[j] = [localPitch, localYaw, 0];
        }
    }

    function axisAngleToRotationMatrix(r) {
        const rx = r[0], ry = r[1], rz = r[2];
        const angle = Math.sqrt(rx * rx + ry * ry + rz * rz);
        const m = new THREE.Matrix4();
        if (angle > 1e-6) {
            const axis = new THREE.Vector3(rx / angle, ry / angle, rz / angle);
            m.makeRotationAxis(axis, angle);
        }
        return m;
    }

    const T_local_0 = new THREE.Matrix4().makeTranslation(joints[0].x, joints[0].y, joints[0].z).multiply(axisAngleToRotationMatrix(activeRotations[0]));
    const T_world_0 = T_local_0.clone();

    const T_local_1 = new THREE.Matrix4().makeTranslation(joints[1].x - joints[0].x, joints[1].y - joints[0].y, joints[1].z - joints[0].z).multiply(axisAngleToRotationMatrix(activeRotations[1]));
    const T_world_1 = T_world_0.clone().multiply(T_local_1);

    const T_local_2 = new THREE.Matrix4().makeTranslation(joints[2].x - joints[1].x, joints[2].y - joints[1].y, joints[2].z - joints[1].z).multiply(axisAngleToRotationMatrix(activeRotations[2]));
    const T_world_2 = T_world_1.clone().multiply(T_local_2);

    const T_local_3 = new THREE.Matrix4().makeTranslation(joints[3].x - joints[1].x, joints[3].y - joints[1].y, joints[3].z - joints[1].z).multiply(axisAngleToRotationMatrix(activeRotations[3]));
    const T_world_3 = T_world_1.clone().multiply(T_local_3);

    const T_world = [T_world_0, T_world_1, T_world_2, T_world_3];
    const skinningMatrices = [];
    for (let j = 0; j < 4; j++) {
        const invBind = new THREE.Matrix4().makeTranslation(-joints[j].x, -joints[j].y, -joints[j].z);
        skinningMatrices.push(T_world[j].clone().multiply(invBind));
    }

    for (let i = 0; i < vCount; i++) {
        const x = currentPos[i * 3], y = currentPos[i * 3 + 1], z = currentPos[i * 3 + 2];
        let sx = 0, sy = 0, sz = 0;
        for (let j = 0; j < 4; j++) {
            const w = skinningWeights[j * vCount + i];
            if (w > 1e-4) {
                const m = skinningMatrices[j].elements;
                sx += w * (m[0] * x + m[4] * y + m[8] * z + m[12]);
                sy += w * (m[1] * x + m[5] * y + m[9] * z + m[13]);
                sz += w * (m[2] * x + m[6] * y + m[10] * z + m[14]);
            }
        }
        currentPos[i * 3] = sx; currentPos[i * 3 + 1] = sy; currentPos[i * 3 + 2] = sz;
    }

    const posAttr = faceMesh.geometry.attributes.position;
    posAttr.array.set(currentPos);
    posAttr.needsUpdate = true;
    faceMesh.geometry.computeVertexNormals();
    return activeExprIndices.length + activeIdIndices.length;
}

let lastTime = performance.now();
let frameCount = 0;
let fpsVal = document.getElementById("fps-val");
let msVal = document.getElementById("ms-val");
let activeVal = document.getElementById("active-val");

let lastBlinkTime = 0;
let nextBlinkDelay = 2000 + Math.random() * 3000;
let blinkStartTime = 0;
let isBlinking = false;

function scheduleNextBlink() {
    const baseInterval = 2500;
    const jitter = (Math.random() * 2.0 - 1.0) * 1500;
    let delay = baseInterval + jitter;
    delay = Math.max(1200, Math.min(7000, delay));
    return delay;
}

function renderLoop(time) {
    requestAnimationFrame(renderLoop);
    const t0 = performance.now();

    const now = performance.now();
    let currentBlinkWeight = 0;
    if (!isBlinking && now - lastBlinkTime > nextBlinkDelay) {
        isBlinking = true;
        blinkStartTime = now;
    }
    if (isBlinking) {
        const elapsed = (now - blinkStartTime) / 120;
        if (elapsed >= 1.0) {
            isBlinking = false;
            lastBlinkTime = now;
            nextBlinkDelay = scheduleNextBlink();
        } else {
            currentBlinkWeight = Math.sin(elapsed * Math.PI) * 1.5;
        }
    }

    let activeShapesCount = 0;
    let blended = null;

    if (activeUtterancePlaying && visemeTimeline) {
        const timeS = audioSync.getCurrentTime();
        document.getElementById("audio-time").innerText = `${timeS.toFixed(2)}s / ${audioSync.getDuration().toFixed(2)}s`;
        const speechCoeffs = animController.getSpeechCoefficients(timeS, visemeTimeline);
        blended = animController.blend(speechCoeffs, currentEmotionCoeffs);
    } else {
        const speechCoeffs = new Array(182).fill(0.0);
        blended = animController.blend(speechCoeffs, currentEmotionCoeffs);
    }

    const finalCoeffs = new Float32Array(383);
    for (let j = 0; j < 182; j++) finalCoeffs[200 + j] = blended[j];
    if (blinkCoefficients && currentBlinkWeight > 0.01) {
        for (let j = 0; j < 200; j++) finalCoeffs[j] += blinkCoefficients[j] * currentBlinkWeight;
    }

    activeShapesCount = deformMesh(finalCoeffs);

    controls.update();
    renderer.render(scene, camera);

    const t1 = performance.now();
    frameCount++;
    if (t1 - lastTime >= 1000) {
        fpsVal.innerText = frameCount;
        frameCount = 0;
        lastTime = t1;
    }
    msVal.innerText = (t1 - t0).toFixed(1);
    activeVal.innerText = activeShapesCount;
}

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
    speakBtn.disabled = true;
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
        console.error("Failed to load emotion coefficients", e);
        speakBtn.disabled = false;
    });
}

speakBtn.addEventListener("click", async () => {
    const text = textInput.value.trim();
    if (!text) return;
    speakBtn.disabled = true;
    speakBtn.innerText = "Synthesizing...";
    playPauseBtn.disabled = true;
    try {
        const styleSelect = document.getElementById("style-select");
        const response = await fetch("/api/speak", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                text: text,
                emotion: emotionSelect.value,
                intensity: parseFloat(intensityRange.value),
                style_id: parseInt(styleSelect.value || "0")
            })
        });
        const data = await response.json();
        await audioSync.loadAudioFromBase64(data.audio_base64);
        visemeTimeline = data.visemes;
        playPauseBtn.disabled = false;
        playPauseBtn.innerText = "Pause";
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

playPauseBtn.addEventListener("click", () => {
    if (audioSync.isPlaying) {
        audioSync.pause();
        playPauseBtn.innerText = "Play Audio";
    } else {
        audioSync.play();
        playPauseBtn.innerText = "Pause";
    }
});

const yawSlider = document.getElementById("pose-yaw");
const pitchSlider = document.getElementById("pose-pitch");
const gazeYawSlider = document.getElementById("gaze-yaw");
const gazePitchSlider = document.getElementById("gaze-pitch");
const yawVal = document.getElementById("yaw-val");
const pitchVal = document.getElementById("pitch-val");
const gazeYVal = document.getElementById("gaze-y-val");
const gazePVal = document.getElementById("gaze-p-val");

yawSlider.addEventListener("input", (e) => {
    const val = parseInt(e.target.value);
    yawVal.innerText = `${val}°`;
    neckRotation[1] = val * Math.PI / 180.0;
});
pitchSlider.addEventListener("input", (e) => {
    const val = parseInt(e.target.value);
    pitchVal.innerText = `${val}°`;
    neckRotation[0] = val * Math.PI / 180.0;
});
gazeYawSlider.addEventListener("input", (e) => {
    const val = parseInt(e.target.value);
    gazeYVal.innerText = `${val}°`;
    const rad = val * Math.PI / 180.0;
    leftEyeRotation[1] = rad;
    rightEyeRotation[1] = rad;
});
gazePitchSlider.addEventListener("input", (e) => {
    const val = parseInt(e.target.value);
    gazePVal.innerText = `${val}°`;
    const rad = val * Math.PI / 180.0;
    leftEyeRotation[0] = rad;
    rightEyeRotation[0] = rad;
});

const randomIdBtn = document.getElementById("random-id-btn");
const idGender = document.getElementById("id-gender");
const idEthnicity = document.getElementById("id-ethnicity");

randomIdBtn.addEventListener("click", () => {
    const gender = parseInt(idGender.value);
    const ethnicity = parseInt(idEthnicity.value);
    randomIdBtn.disabled = true;
    randomIdBtn.innerText = "Generating...";
    fetch("/api/identity", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ gender: gender, ethnicity: ethnicity })
    })
    .then(r => r.json())
    .then(data => {
        currentIdentityCoeffs.set(data.coefficients);
        console.log("New GNM identity applied.");
    })
    .catch(e => console.error("Identity generation failed", e))
    .finally(() => {
        randomIdBtn.disabled = false;
        randomIdBtn.innerText = "Generate Identity";
    });
});

let blinkCoefficients = null;

async function loadBlinkCoefficients() {
    try {
        const res = await fetch("/api/blink", {
            method: "POST", headers: { "Content-Type": "application/json" }, body: "{}"
        });
        const data = await res.json();
        blinkCoefficients = new Float32Array(data.coefficients);
        console.log("Blink coefficients pre-cached.");
    } catch(e) {
        console.warn("Failed to load blink coefficients", e);
    }
}

window.addEventListener('resize', () => {
    const container = document.getElementById("canvas-container");
    if (!camera || !renderer) return;
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
});

window.onload = loadGnmBuffers;
