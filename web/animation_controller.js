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
        // GNM expression basis mapping (verified against model):
        // [200] = lip raise (upward), [201] = jaw open (lip center down), [202] = lip spread
        table["aa"][1] = 2.5;   table["aa"][0] = -0.5; table["aa"][2] = 0.5;
        table["EE"][2] = 2.0;   table["EE"][1] = 1.0;  table["EE"][3] = 0.3;
        table["OO"][1] = -1.0;  table["OO"][2] = -0.8; table["OO"][T] = -0.3;
        table["PP"][1] = -1.5;  table["PP"][0] = -0.5; table["PP"][3] = 0.5;
        table["FF"][1] = -0.3;  table["FF"][3] = 0.5;  table["FF"][T + 2] = 0.3;
        table["TH"][1] = 0.3;   table["TH"][T] = 1.0;  table["TH"][T + 1] = 0.8; table["TH"][T + 2] = 0.3;
        table["DD"][1] = 0.5;   table["DD"][T] = 0.6;  table["DD"][T + 1] = 0.7;
        table["CH"][T] = 0.5;   table["CH"][T + 2] = 0.6; table["CH"][T + 3] = 0.4;
        table["kk"][2] = -0.3;  table["kk"][T + 1] = 0.7; table["kk"][T + 2] = 0.5;
        table["SS"][T] = 0.3;   table["SS"][T + 1] = 0.5; table["SS"][T + 2] = 0.3;
        table["RR"][T] = 0.6;   table["RR"][T + 1] = 0.4; table["RR"][T + 2] = 0.5; table["RR"][T + 3] = 0.3;
        table["schwa"][1] = 0.5; table["schwa"][T] = 0.3;
        return table;
    }

    getSpeechCoefficients(timeS, timeline) {
        if (!this.visemeTable || !timeline || timeline.length === 0) {
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
        for (let i = 0; i < 150; i++) {
            const s = speechCoeffs[i];
            const e = emotionCoeffs[200 + i] * 0.15;
            blended[200 + i] = Math.abs(s) > Math.abs(e) ? s : e;
        }
        for (let i = 0; i < 32; i++) blended[350 + i] = speechCoeffs[150 + i];
        blended[382] = emotionCoeffs[382];
        return blended;
    }
}
