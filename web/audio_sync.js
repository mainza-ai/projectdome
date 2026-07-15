class AudioSync {
    constructor() {
        this.audioContext = null;
        this.audioBuffer = null;
        this.sourceNode = null;
        this.startTime = 0;
        this.isPlaying = false;
        this.elapsedOffset = 0; // track elapsed time when paused
        this.onEnded = null;
    }

    async init() {
        if (!this.audioContext) {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }
    }

    async loadAudioFromBase64(base64Data) {
        await this.init();
        
        // Stop current audio if playing
        this.stop();
        
        // Decode base64 string to arraybuffer
        const binaryString = window.atob(base64Data);
        const len = binaryString.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        
        // Decode audio data to buffer
        try {
            this.audioBuffer = await this.audioContext.decodeAudioData(bytes.buffer);
            this.elapsedOffset = 0;
            this.isPlaying = false;
            console.log(`Audio loaded successfully. Duration: ${this.getDuration().toFixed(2)}s`);
        } catch (e) {
            console.error("Failed to decode audio data", e);
        }
    }

    play() {
        if (!this.audioBuffer) return;
        if (this.isPlaying) return;

        this.sourceNode = this.audioContext.createBufferSource();
        this.sourceNode.buffer = this.audioBuffer;
        this.sourceNode.connect(this.audioContext.destination);
        
        this.startTime = this.audioContext.currentTime - this.elapsedOffset;
        this.sourceNode.start(0, this.elapsedOffset);
        this.isPlaying = true;

        this.sourceNode.onended = () => {
            // Check if played to the end (not stopped manually)
            if (this.isPlaying && this.getCurrentTime() >= this.getDuration() - 0.05) {
                this.isPlaying = false;
                this.elapsedOffset = 0;
                if (this.onEnded) this.onEnded();
            }
        };
    }

    pause() {
        if (!this.isPlaying) return;
        this.sourceNode.onended = null; // Prevent triggering onended event
        this.sourceNode.stop();
        this.elapsedOffset = this.audioContext.currentTime - this.startTime;
        this.isPlaying = false;
    }

    stop() {
        if (this.isPlaying && this.sourceNode) {
            this.sourceNode.onended = null;
            this.sourceNode.stop();
        }
        this.elapsedOffset = 0;
        this.isPlaying = false;
    }

    getCurrentTime() {
        if (!this.isPlaying) {
            return this.elapsedOffset;
        }
        return this.audioContext.currentTime - this.startTime;
    }

    getDuration() {
        return this.audioBuffer ? this.audioBuffer.duration : 0;
    }
}
