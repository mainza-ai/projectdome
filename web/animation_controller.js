class AnimationController {
    constructor() {
        this.visemeTable = null;
        this.rampDuration = 0.04;
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
        const T = 150;
        for (const v of visemes) {
            table[v] = new Array(182).fill(0.0);
        }
        // Lower-face coefficients (indices 0-149) scaled 10x for visible speech
        // mouth opening. GNM PCA basis produces only 0.008 max displacement
        // per unit coefficient — speech needs 10-20x emotion scale.
        // Tongue coefficients (T=150) left at original scale.
        table["aa"][1] = 25.0;  table["aa"][0] = -5.0; table["aa"][2] = 5.0;
        table["EE"][2] = 20.0;  table["EE"][1] = 10.0; table["EE"][3] = 3.0;
        table["OO"][1] = -10.0; table["OO"][2] = -8.0; table["OO"][T] = -0.3;
        table["PP"][1] = -15.0; table["PP"][0] = -5.0; table["PP"][3] = 5.0;
        table["FF"][1] = -3.0;  table["FF"][3] = 5.0;  table["FF"][T + 2] = 0.3;
        table["TH"][1] = 3.0;   table["TH"][T] = 1.0;  table["TH"][T + 1] = 0.8; table["TH"][T + 2] = 0.3;
        table["DD"][1] = 5.0;   table["DD"][T] = 0.6;  table["DD"][T + 1] = 0.7;
        table["CH"][T] = 0.5;   table["CH"][T + 2] = 0.6; table["CH"][T + 3] = 0.4;
        table["kk"][2] = -3.0;  table["kk"][T + 1] = 0.7; table["kk"][T + 2] = 0.5;
        table["SS"][T] = 0.3;   table["SS"][T + 1] = 0.5; table["SS"][T + 2] = 0.3;
        table["RR"][T] = 0.6;   table["RR"][T + 1] = 0.4; table["RR"][T + 2] = 0.5; table["RR"][T + 3] = 0.3;
        table["schwa"][1] = 5.0; table["schwa"][T] = 0.3;
        return table;
    }

    getSpeechCoefficients(timeS, timeline, audioDuration) {
        if (!this.visemeTable) {
            return new Array(182).fill(0.0);
        }
        if (!timeline || timeline.length < 2) {
            if (audioDuration && audioDuration > 0.1) {
                const fallback = [];
                const visemeNames = ["aa", "EE", "OO", "PP", "FF", "TH", "DD", "CH", "kk", "SS", "RR", "schwa"];
                const interval = Math.max(0.05, Math.min(0.15, audioDuration / 40));
                let t = 0;
                while (t < audioDuration) {
                    const name = visemeNames[Math.floor(Math.random() * visemeNames.length)];
                    const end = Math.min(t + interval, audioDuration);
                    fallback.push({ name, start_time: t, end_time: end });
                    t = end;
                }
                return this.getSpeechCoefficients(timeS, fallback, audioDuration);
            }
            return new Array(182).fill(0.0);
        }
        if (timeS < timeline[0].start_time) {
            return this.visemeTable["IDLE"];
        }
        if (timeS > timeline[timeline.length - 1].end_time) {
            return this.visemeTable["IDLE"];
        }
        let activeIdx = -1;
        for (let i = 0; i < timeline.length; i++) {
            if (timeline[i].start_time <= timeS && timeS <= timeline[i].end_time) {
                activeIdx = i;
                break;
            }
        }
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
        const currentEvent = timeline[activeIdx];
        const currentCoeffs = this.visemeTable[currentEvent.name] || this.visemeTable["IDLE"];
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
        const blended = new Float32Array(383);
        for (let i = 0; i < 200; i++) blended[i] = emotionCoeffs[i];
        const speechEnergy = speechCoeffs.reduce((a, b) => a + Math.abs(b), 0);
        const isSpeaking = speechEnergy > 0.01;
        if (isSpeaking) {
            for (let i = 0; i < 150; i++) blended[200 + i] = speechCoeffs[i] + 0.3 * emotionCoeffs[200 + i];
        } else {
            for (let i = 0; i < 150; i++) blended[200 + i] = emotionCoeffs[200 + i];
        }
        for (let i = 0; i < 32; i++) blended[350 + i] = speechCoeffs[150 + i] + 0.3 * emotionCoeffs[350 + i];
        blended[382] = emotionCoeffs[382];
        return blended;
    }
}
