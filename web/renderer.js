let scene, camera, renderer, controls;
let faceMesh;
let metadata = null;
let meanPositions = null;
let identityBasis = null;
let expressionBasis = null;
let jointIdentityBasis = null;
let faceIndices = null;
let skinningWeights = null;
let jointRegressor = null;
let vertexBodyParts = null;
let mirrorIndices = null;

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

const BODY_PART_SKIN = 0;

function decodeFloat16(uint16) {
    const s = (uint16 & 0x8000) >> 15;
    const e = (uint16 & 0x7C00) >> 10;
    const f = uint16 & 0x03FF;
    if (e === 0) return f === 0 ? (s ? -0.0 : 0.0) : (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    if (e === 31) return f ? NaN : (s ? -Infinity : Infinity);
    return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

function float16BufferToFloat32Array(buf) {
    const u16 = new Uint16Array(buf);
    const f32 = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) f32[i] = decodeFloat16(u16[i]);
    return f32;
}

function fetchWithStatus(url) {
    return fetch(url).then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}: ${url} not found — run 'python tools/export_basis.py'`);
        return r;
    });
}

function axisAngleToRotationMatrix(r) {
    const rx = r[0], ry = r[1], rz = r[2];
    const angle = Math.sqrt(rx * rx + ry * ry + rz * rz);
    const m = new THREE.Matrix4();
    if (angle > 1e-8) {
        m.makeRotationAxis(new THREE.Vector3(rx / angle, ry / angle, rz / angle), angle);
    }
    return m;
}

function buildJointTransforms(joints, rotations, translation, parents) {
    const nj = joints.length;
    const localMats = [];
    const rootTrans = new THREE.Vector3().copy(joints[0]).add(translation);
    for (let j = 0; j < nj; j++) {
        const R = axisAngleToRotationMatrix(rotations[j]);
        const T = new THREE.Matrix4();
        if (j === 0) {
            T.makeTranslation(rootTrans.x, rootTrans.y, rootTrans.z);
        } else {
            T.makeTranslation(
                joints[j].x - joints[parents[j]].x,
                joints[j].y - joints[parents[j]].y,
                joints[j].z - joints[parents[j]].z
            );
        }
        T.multiply(R);
        localMats.push(T);
    }
    const worldMats = [localMats[0].clone()];
    for (let j = 1; j < nj; j++) {
        const pIdx = parents[j] === -1 ? 0 : parents[j];
        worldMats.push(new THREE.Matrix4().copy(worldMats[pIdx]).multiply(localMats[j]));
    }
    return worldMats;
}

async function loadGnmBuffers() {
    const overlay = document.getElementById("loading-overlay");
    const statusEl = overlay.querySelector("p");
    try {
        const metaRes = await fetchWithStatus("/data/web/metadata.json");
        metadata = await metaRes.json();

        const meanRes = await fetchWithStatus("/data/web/mean_positions.bin");
        meanPositions = new Float32Array(await meanRes.arrayBuffer());

        const faceRes = await fetchWithStatus("/data/web/face_indices.bin");
        faceIndices = new Uint32Array(await faceRes.arrayBuffer());

        const idRes = await fetchWithStatus("/data/web/identity_basis.bin");
        identityBasis = float16BufferToFloat32Array(await idRes.arrayBuffer());

        const jidRes = await fetchWithStatus("/data/web/joint_identity_basis.bin");
        jointIdentityBasis = float16BufferToFloat32Array(await jidRes.arrayBuffer());

        const exprRes = await fetchWithStatus("/data/web/expression_basis.bin");
        expressionBasis = float16BufferToFloat32Array(await exprRes.arrayBuffer());

        const skinRes = await fetchWithStatus("/data/web/skinning_weights.bin");
        skinningWeights = new Float32Array(await skinRes.arrayBuffer());

        const regRes = await fetchWithStatus("/data/web/joint_regressor.bin");
        jointRegressor = new Float32Array(await regRes.arrayBuffer());

        try {
            const vpRes = await fetch("/data/web/vertex_body_parts.bin");
            vertexBodyParts = new Int32Array(await vpRes.arrayBuffer());
        } catch (e) { vertexBodyParts = null; }

        try {
            const mirRes = await fetch("/data/web/mirror_indices.bin");
            mirrorIndices = new Int32Array(await mirRes.arrayBuffer());
        } catch (e) { mirrorIndices = null; }

        overlay.style.opacity = 0;
        setTimeout(() => overlay.style.display = "none", 500);

        await animController.loadVisemeTable();
        await loadBlinkCoefficients();
        initScene();
    } catch (e) {
        console.error("Failed to load GNM buffers", e);
        statusEl.innerText = "Loading Failed! " + e.message;
    }
}

function getVertexColors() {
    if (!vertexBodyParts || !metadata) return null;
    const n = metadata.num_vertices;
    const colors = new Float32Array(n * 3);
    const palette = {};
    const gn = metadata.group_names || [];
    gn.forEach((name, i) => {
        if (name.includes('sclera') || name.includes('iris') || name.includes('pupil')) palette[i] = [0.95, 0.95, 0.97];
        else if (name.includes('teeth') || name.includes('gum')) palette[i] = [0.92, 0.90, 0.85];
        else if (name.includes('tongue')) palette[i] = [0.80, 0.50, 0.50];
        else if (name.includes('mouth_sock')) palette[i] = [0.35, 0.20, 0.20];
        else if (name.includes('eye')) palette[i] = [0.85, 0.85, 0.90];
        else if (name.includes('ear')) palette[i] = [0.78, 0.70, 0.62];
        else palette[i] = [0.78, 0.70, 0.62];
    });
    for (let i = 0; i < n; i++) {
        const c = palette[vertexBodyParts[i]] || [0.78, 0.70, 0.62];
        colors[i*3] = c[0]; colors[i*3+1] = c[1]; colors[i*3+2] = c[2];
    }
    return colors;
}

function initScene() {
    const container = document.getElementById("canvas-container");
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf5f5f7);

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(meanPositions), 3));
    geometry.setIndex(new THREE.BufferAttribute(faceIndices, 1));

    geometry.computeBoundingBox();
    const bb = geometry.boundingBox;
    const center = new THREE.Vector3();
    bb.getCenter(center);
    const size = new THREE.Vector3();
    bb.getSize(size);

    console.log('Head BB min:', bb.min.toArray().map(v=>v.toFixed(4)));
    console.log('Head BB max:', bb.max.toArray().map(v=>v.toFixed(4)));
    console.log('Head center:', center.toArray().map(v=>v.toFixed(4)));
    console.log('Head size:', size.toArray().map(v=>v.toFixed(4)));

    const containerAspect = container.clientWidth / container.clientHeight;
    const vFov = 40 * Math.PI / 180;
    const margin = 1.5;
    const maxDim = Math.max(size.x, size.y, size.z);
    const dist = maxDim * margin / (2 * Math.tan(vFov / 2));
    console.log('MaxDim:', maxDim.toFixed(4), 'Dist:', dist.toFixed(4));

    // BUG: OrbitControls constructor calls update() internally, locking spherical
    // coords relative to default target (0,0,0). If we create controls with camera
    // already centered on head, the spherical offset includes the head center
    // offset from origin. When we then set controls.target = center, the camera
    // position doubles the offset.
    //
    // Fix: place camera at (0,0,dist) BEFORE creating controls (so spherical
    // offset is just +Z). AFTER controls creation, set target to head center —
    // the camera then becomes center + (0,0,dist) = (cx, cy, cz+dist).
    camera = new THREE.PerspectiveCamera(40, containerAspect, 0.01, dist * 10);
    camera.position.set(0, 0, dist);
    camera.lookAt(0, 0, 0);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.target.copy(center);
    controls.update();

    console.log('Camera pos:', camera.position.toArray().map(v=>v.toFixed(4)));
    console.log('Camera target:', controls.target.toArray().map(v=>v.toFixed(4)));
    console.log('Distance:', camera.position.distanceTo(controls.target).toFixed(4));

    const amb = new THREE.AmbientLight(0xffffff, 0.25);
    scene.add(amb);
    const key = new THREE.DirectionalLight(0xffffff, 0.85);
    key.position.set(0.5, 0.5, 0.5);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0xa5b4fc, 0.4);
    fill.position.set(-0.5, 0.2, 0.3);
    scene.add(fill);
    const rim = new THREE.DirectionalLight(0xffffff, 0.45);
    rim.position.set(0, 0.8, -0.8);
    scene.add(rim);

    const vertColors = getVertexColors();
    if (vertColors) geometry.setAttribute('color', new THREE.BufferAttribute(vertColors, 3));

    const mat = new THREE.MeshStandardMaterial({
        roughness: 0.4, metalness: 0.05, flatShading: false,
        side: THREE.DoubleSide, vertexColors: !!vertColors,
    });

    faceMesh = new THREE.Mesh(geometry, mat);
    scene.add(faceMesh);

    currentEmotionCoeffs = new Float32Array(metadata.expression_dim);
    updateCurrentEmotion();
    requestAnimationFrame(renderLoop);
}

const currentPos = new Float32Array(17821 * 3);
const workVec3 = [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()];

function deformMesh(expressionCoeffs) {
    if (!faceMesh || !metadata) return 0;
    const vCount = metadata.num_vertices;
    const nJ = metadata.num_joints;

    currentPos.set(meanPositions);

    const activeId = [], activeIdW = [];
    for (let i = 0; i < metadata.identity_dim; i++) {
        const w = currentIdentityCoeffs[i];
        if (Math.abs(w) > 1e-4) { activeId.push(i); activeIdW.push(w); }
    }
    for (let a = 0; a < activeId.length; a++) {
        const off = activeId[a] * vCount * 3;
        const w = activeIdW[a];
        for (let j = 0; j < vCount * 3; j++) currentPos[j] += identityBasis[off + j] * w;
    }

    const activeEx = [], activeExW = [];
    for (let i = 0; i < metadata.expression_dim; i++) {
        const w = expressionCoeffs[i];
        if (Math.abs(w) > 1e-4) { activeEx.push(i); activeExW.push(w); }
    }
    for (let a = 0; a < activeEx.length; a++) {
        const off = activeEx[a] * vCount * 3;
        const w = activeExW[a];
        for (let j = 0; j < vCount * 3; j++) currentPos[j] += expressionBasis[off + j] * w;
    }

    const joints = [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()];
    for (let j = 0; j < nJ; j++) {
        let jx = 0, jy = 0, jz = 0;
        const regOff = j * vCount;
        for (let v = 0; v < vCount; v++) {
            const w = jointRegressor[regOff + v];
            if (Math.abs(w) > 1e-6) {
                const vi = v * 3;
                jx += w * currentPos[vi]; jy += w * currentPos[vi + 1]; jz += w * currentPos[vi + 2];
            }
        }
        joints[j].set(jx, jy, jz);
    }

    const gazeAuto = document.getElementById("gaze-auto");
    const rot = [neckRotation, headRotation, [...leftEyeRotation], [...rightEyeRotation]];
    if (gazeAuto && gazeAuto.checked && camera) {
        const headY = headRotation[1] + neckRotation[1];
        const headP = headRotation[0] + neckRotation[0];
        for (let j = 2; j <= 3; j++) {
            const dir = new THREE.Vector3().copy(camera.position).sub(joints[j]).normalize();
            let ly = Math.atan2(dir.x, dir.z) - headY;
            let lp = Math.asin(-dir.y) - headP;
            ly = Math.max(-0.436, Math.min(0.436, ly));
            lp = Math.max(-0.262, Math.min(0.262, lp));
            rot[j] = [lp, ly, 0];
        }
    }

    const templateJoints = metadata.template_joint_positions.map(p => new THREE.Vector3(p[0], p[1], p[2]));
    const jidJoints = [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()];
    for (let j = 0; j < nJ; j++) jidJoints[j].copy(templateJoints[j]);
    if (jointIdentityBasis) {
        for (let i = 0; i < metadata.identity_dim; i++) {
            const w = currentIdentityCoeffs[i];
            if (Math.abs(w) > 1e-4) {
                const off = i * nJ * 3;
                for (let j = 0; j < nJ; j++) {
                    jidJoints[j].x += jointIdentityBasis[off + j * 3] * w;
                    jidJoints[j].y += jointIdentityBasis[off + j * 3 + 1] * w;
                    jidJoints[j].z += jointIdentityBasis[off + j * 3 + 2] * w;
                }
            }
        }
    }

    const parents = metadata.joint_parent_indices;
    const worldMats = buildJointTransforms(jidJoints, rot, new THREE.Vector3(0, 0, 0), parents);

    const skinningMats = [];
    for (let j = 0; j < nJ; j++) {
        const R = new THREE.Matrix4().extractRotation(worldMats[j]);
        const Rj = new THREE.Vector3().copy(jidJoints[j]).applyMatrix4(R);
        const T = new THREE.Matrix4().makeTranslation(
            worldMats[j].elements[12] - Rj.x,
            worldMats[j].elements[13] - Rj.y,
            worldMats[j].elements[14] - Rj.z
        );
        T.multiply(R);
        const invBind = new THREE.Matrix4().makeTranslation(-jidJoints[j].x, -jidJoints[j].y, -jidJoints[j].z);
        skinningMats.push(T.clone().multiply(invBind));
    }

    for (let i = 0; i < vCount; i++) {
        const x = currentPos[i * 3], y = currentPos[i * 3 + 1], z = currentPos[i * 3 + 2];
        let sx = 0, sy = 0, sz = 0;
        for (let j = 0; j < nJ; j++) {
            const w = skinningWeights[j * vCount + i];
            if (w > 1e-4) {
                const m = skinningMats[j].elements;
                const vx = m[0]*x + m[4]*y + m[8]*z + m[12];
                const vy = m[1]*x + m[5]*y + m[9]*z + m[13];
                const vz = m[2]*x + m[6]*y + m[10]*z + m[14];
                sx += w * vx; sy += w * vy; sz += w * vz;
            }
        }
        currentPos[i*3] = sx; currentPos[i*3+1] = sy; currentPos[i*3+2] = sz;
    }

    const posAttr = faceMesh.geometry.attributes.position;
    posAttr.array.set(currentPos);
    posAttr.needsUpdate = true;
    faceMesh.geometry.computeVertexNormals();

    return activeEx.length + activeId.length;
}

let lastTime = performance.now();
let frameCount = 0;
const fpsVal = document.getElementById("fps-val");
const msVal = document.getElementById("ms-val");
const activeVal = document.getElementById("active-val");

let lastBlinkTime = 0;
let nextBlinkDelay = 2000 + Math.random() * 3000;
let blinkStartTime = 0;
let isBlinking = false;

function scheduleNextBlink() {
    return Math.max(1200, Math.min(7000, 2500 + (Math.random() * 2 - 1) * 1500));
}

function renderLoop(time) {
    requestAnimationFrame(renderLoop);
    const t0 = performance.now();
    const now = performance.now();

    let blinkW = 0;
    if (!isBlinking && now - lastBlinkTime > nextBlinkDelay) {
        isBlinking = true; blinkStartTime = now;
    }
    if (isBlinking) {
        const el = (now - blinkStartTime) / 120;
        if (el >= 1.0) { isBlinking = false; lastBlinkTime = now; nextBlinkDelay = scheduleNextBlink(); }
        else { blinkW = Math.sin(el * Math.PI) * 1.5; }
    }

    let blended;
    if (activeUtterancePlaying && visemeTimeline) {
        const ts = audioSync.getCurrentTime();
        document.getElementById("audio-time").innerText = `${ts.toFixed(2)}s / ${audioSync.getDuration().toFixed(2)}s`;
        blended = animController.blend(animController.getSpeechCoefficients(ts, visemeTimeline), currentEmotionCoeffs);
    } else {
        blended = animController.blend(new Array(182).fill(0.0), currentEmotionCoeffs);
    }

    if (blinkCoefficients && blinkW > 0.01) {
        for (let j = 0; j < 200; j++) blended[j] += blinkCoefficients[j] * blinkW;
    }

    const active = deformMesh(blended);
    controls.update();
    renderer.render(scene, camera);

    const t1 = performance.now();
    frameCount++;
    if (t1 - lastTime >= 1000) { fpsVal.innerText = frameCount; frameCount = 0; lastTime = t1; }
    msVal.innerText = (t1 - t0).toFixed(1);
    activeVal.innerText = active;
}

const speakBtn = document.getElementById("speak-btn");
const playPauseBtn = document.getElementById("play-pause-btn");
const textInput = document.getElementById("speech-text");
const emotionSelect = document.getElementById("emotion-select");
const intensityRange = document.getElementById("intensity-range");
const intensityVal = document.getElementById("intensity-val");

intensityRange.addEventListener("input", e => { intensityVal.innerText = e.target.value; updateCurrentEmotion(); });
emotionSelect.addEventListener("change", updateCurrentEmotion);

function updateCurrentEmotion() {
    if (!metadata || !animController.visemeTable) return;
    const name = emotionSelect.value;
    const intensity = parseFloat(intensityRange.value);
    speakBtn.disabled = true;
    fetch("/api/emotion", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name, intensity }) })
        .then(r => r.json()).then(d => { currentEmotionCoeffs = new Float32Array(d.coefficients); speakBtn.disabled = false; })
        .catch(e => { console.error("Emotion fetch failed", e); speakBtn.disabled = false; });
}

speakBtn.addEventListener("click", async () => {
    const text = textInput.value.trim();
    if (!text) return;
    speakBtn.disabled = true;
    speakBtn.innerText = "Synthesizing...";
    playPauseBtn.disabled = true;
    try {
        const styleSelect = document.getElementById("style-select");
        const endpoint = text.length > 80 ? "/api/speak/stream" : "/api/speak";
        const r = await fetch(endpoint, {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                text, emotion: emotionSelect.value,
                intensity: parseFloat(intensityRange.value),
                style_id: parseInt(styleSelect.value || "0")
            })
        });
        const d = await r.json();
        await audioSync.loadAudioFromBase64(d.audio_base64);
        visemeTimeline = d.visemes;
        playPauseBtn.disabled = false;
        playPauseBtn.innerText = "Pause";
        audioSync.play();
        activeUtterancePlaying = true;
        audioSync.onEnded = () => { playPauseBtn.innerText = "Play Audio"; activeUtterancePlaying = false; };
    } catch (e) { console.error("Synthesize API failed", e); }
    finally { speakBtn.disabled = false; speakBtn.innerText = "Speak"; }
});

playPauseBtn.addEventListener("click", () => {
    if (audioSync.isPlaying) { audioSync.pause(); playPauseBtn.innerText = "Play Audio"; }
    else { audioSync.play(); playPauseBtn.innerText = "Pause"; }
});

const yawSlider = document.getElementById("pose-yaw");
const pitchSlider = document.getElementById("pose-pitch");
const gazeYawSlider = document.getElementById("gaze-yaw");
const gazePitchSlider = document.getElementById("gaze-pitch");
document.getElementById("yaw-val");
document.getElementById("pitch-val");
document.getElementById("gaze-y-val");
document.getElementById("gaze-p-val");

yawSlider.addEventListener("input", e => { const v = parseInt(e.target.value); document.getElementById("yaw-val").innerText = `${v}°`; neckRotation[1] = v * Math.PI / 180; });
pitchSlider.addEventListener("input", e => { const v = parseInt(e.target.value); document.getElementById("pitch-val").innerText = `${v}°`; neckRotation[0] = v * Math.PI / 180; });
gazeYawSlider.addEventListener("input", e => { const v = parseInt(e.target.value); document.getElementById("gaze-y-val").innerText = `${v}°`; const r = v * Math.PI / 180; leftEyeRotation[1] = r; rightEyeRotation[1] = r; });
gazePitchSlider.addEventListener("input", e => { const v = parseInt(e.target.value); document.getElementById("gaze-p-val").innerText = `${v}°`; const r = v * Math.PI / 180; leftEyeRotation[0] = r; rightEyeRotation[0] = r; });

const randomIdBtn = document.getElementById("random-id-btn");
randomIdBtn.addEventListener("click", () => {
    const gender = parseInt(document.getElementById("id-gender").value);
    const ethnicity = parseInt(document.getElementById("id-ethnicity").value);
    randomIdBtn.disabled = true; randomIdBtn.innerText = "Generating...";
    fetch("/api/identity", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ gender, ethnicity }) })
        .then(r => r.json()).then(d => { currentIdentityCoeffs.set(d.coefficients); })
        .catch(e => console.error("Identity generation failed", e))
        .finally(() => { randomIdBtn.disabled = false; randomIdBtn.innerText = "Generate Identity"; });
});

let blinkCoefficients = null;
async function loadBlinkCoefficients() {
    try {
        const r = await fetch("/api/blink", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
        blinkCoefficients = new Float32Array((await r.json()).coefficients);
    } catch(e) { console.warn("Blink coefficients failed", e); }
}

window.addEventListener('resize', () => {
    const c = document.getElementById("canvas-container");
    if (!camera || !renderer) return;
    camera.aspect = c.clientWidth / c.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(c.clientWidth, c.clientHeight);
});

let pcaSlidersInitialized = false;
const pcaSliderValues = new Float32Array(253).fill(0.0);

async function initPcaSliders() {
    if (pcaSlidersInitialized) return;
    try {
        const r = await fetch("/api/identity/info", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ n: 10 }) });
        const info = await r.json();
        const container = document.getElementById("pca-sliders-container");
        container.innerHTML = "";
        for (let i = 0; i < info.num_components; i++) {
            const div = document.createElement("div");
            div.className = "intensity-slider";
            div.style.marginBottom = "4px";
            const label = document.createElement("label");
            label.style.minWidth = "90px"; label.style.fontSize = "0.75rem";
            label.textContent = info.component_names[i];
            const slider = document.createElement("input");
            slider.type = "range"; slider.min = "-3"; slider.max = "3"; slider.step = "0.05"; slider.value = "0";
            const val = document.createElement("span");
            val.style.fontSize = "0.75rem"; val.style.minWidth = "20px"; val.textContent = "0";
            slider.addEventListener("input", (idx => {
                const v = parseFloat(slider.value);
                pcaSliderValues[idx] = v;
                currentIdentityCoeffs[idx] = v;
                val.textContent = v.toFixed(1);
            }).bind(null, i));
            div.appendChild(label); div.appendChild(slider); div.appendChild(val);
            container.appendChild(div);
        }
        pcaSlidersInitialized = true;
    } catch (e) { console.warn("PCA sliders failed", e); }
}

document.addEventListener("DOMContentLoaded", () => {
    const details = document.querySelector("#identity-pca-sliders details");
    if (details) details.addEventListener("toggle", () => { if (details.open) initPcaSliders(); });
});

window.onload = loadGnmBuffers;
