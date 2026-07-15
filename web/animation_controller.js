class AnimationController {
    constructor() {
        this.visemeTable = null;
        this.rampDuration = 0.04; // 40ms transition duration
    }

    async loadVisemeTable() {
        try {
            const response = await fetch('/data/viseme_table.json');
            this.visemeTable = await response.json();
            console.log("Viseme table loaded successfully.");
        } catch (e) {
            console.warn("Failed to load viseme table from server, using fallback defaults.", e);
            this.visemeTable = this.getDefaultVisemeTable();
        }
    }

    getDefaultVisemeTable() {
        const table = {};
        const visemes = ["IDLE", "PP", "FF", "TH", "DD", "CH", "kk", "SS", "RR", "aa", "EE", "OO", "schwa"];
        for (const v of visemes) {
            table[v] = new Array(182).fill(0.0);
        }
        
        // Plausible PCA shape approximations
        table["aa"][0] = 1.2;
        table["aa"][1] = -0.5;
        table["OO"][0] = 0.5;
        table["OO"][1] = 1.5;
        table["EE"][2] = 1.5;
        table["PP"][0] = -0.5;
        table["PP"][3] = 1.0;
        table["TH"][0] = 0.3;
        table["TH"][150] = 1.0;
        return table;
    }

    getSpeechCoefficients(timeS, timeline) {
        if (!this.visemeTable || !timeline || timeline.length === 0) {
            return new Array(182).fill(0.0);
        }

        // 1. Edge case: Before first event
        if (timeS < timeline[0].start_time) {
            return this.visemeTable["IDLE"];
        }

        // 2. Edge case: After last event
        if (timeS > timeline[timeline.length - 1].end_time) {
            return this.visemeTable["IDLE"];
        }

        // 3. Find active event index
        let activeIdx = -1;
        for (let i = 0; i < timeline.length; i++) {
            if (timeline[i].start_time <= timeS && timeS <= timeline[i].end_time) {
                activeIdx = i;
                break;
            }
        }

        // 4. Handle gap case (interpolate between events)
        if (activeIdx === -1) {
            for (let i = 0; i < timeline.length - 1; i++) {
                if (timeline[i].end_time < timeS && timeS < timeline[i+1].start_time) {
                    const tStart = timeline[i].end_time;
                    const tEnd = timeline[i+1].start_time;
                    let factor = (timeS - tStart) / Math.max(tEnd - tStart, 1e-5);
                    factor = Math.max(0.0, Math.min(1.0, factor));
                    
                    const cPrev = this.visemeTable[timeline[i].name] || this.visemeTable["IDLE"];
                    const cNext = this.visemeTable[timeline[i+1].name] || this.visemeTable["IDLE"];
                    
                    const blended = new Array(182);
                    for (let j = 0; j < 182; j++) {
                        blended[j] = cPrev[j] + factor * (cNext[j] - cPrev[j]);
                    }
                    return blended;
                }
            }
            return this.visemeTable["IDLE"];
        }

        // 5. Normal case inside an event
        const currentEvent = timeline[activeIdx];
        const currentCoeffs = this.visemeTable[currentEvent.name] || this.visemeTable["IDLE"];

        // Ramping/transition check
        if (activeIdx < timeline.length - 1) {
            const nextEvent = timeline[activeIdx + 1];
            const timeToEnd = currentEvent.end_time - timeS;
            if (timeToEnd < this.rampDuration) {
                let factor = (this.rampDuration - timeToEnd) / this.rampDuration;
                factor = Math.max(0.0, Math.min(1.0, factor));
                
                const nextCoeffs = this.visemeTable[nextEvent.name] || this.visemeTable["IDLE"];
                const blended = new Array(182);
                for (let j = 0; j < 182; j++) {
                    blended[j] = currentCoeffs[j] + factor * (nextCoeffs[j] - currentCoeffs[j]);
                }
                return blended;
            }
        }

        return currentCoeffs;
    }

    blend(speechCoeffs, emotionCoeffs) {
        // speechCoeffs is length 182, emotionCoeffs is length 383
        const blended = new Float32Array(383);
        
        // 1. Upper face & Eyes (0 to 199) -> driven by emotion
        for (let i = 0; i < 200; i++) {
            blended[i] = emotionCoeffs[i];
        }

        // 2. Lower face (200 to 349) -> speech + 0.3 * emotion
        for (let i = 0; i < 150; i++) {
            blended[200 + i] = speechCoeffs[i] + 0.3 * emotionCoeffs[200 + i];
        }

        // 3. Tongue (350 to 381) -> speech
        for (let i = 0; i < 32; i++) {
            blended[350 + i] = speechCoeffs[150 + i];
        }

        // 4. Pupils (382) -> emotion
        blended[382] = emotionCoeffs[382];

        return blended;
    }
}
