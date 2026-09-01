/**
 * AERIS Low-Latency AudioWorklet Processor v3.0
 * 
 * 1. INPUT (Microphone Capture):
 *    - Captures raw microphone stream
 *    - Resamples in real-time from native hardware rate (e.g., 44.1kHz / 48kHz) to target 16,000 Hz
 *    - Accumulates exact 512-sample frames (32 ms @ 16 kHz)
 *    - Posts framed PCM packets to main thread for immediate WebSocket transmission
 *
 * 2. OUTPUT (Enhanced Playback):
 *    - Receives processed ANC enhanced audio frames from backend
 *    - Stores in a bounded, circular jitter ring buffer (MAX 4 frames = 128 ms)
 *    - Discards stale frames if jitter buffer exceeds 128 ms (prevents 10-20s delay build-up)
 *    - Resamples from 16,000 Hz up to native hardware rate for seamless speaker output
 *    - Zero feedback loop (mic input is never passed directly to speaker)
 */

class AerisAudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        
        this.targetRate = 16000;
        this.frameSize = 512; // 32 ms @ 16 kHz
        
        // --- Capture (Input) State ---
        this.inputAccumulator = new Float32Array(this.frameSize * 4);
        this.inputAccumCount = 0;
        this.inputResamplePos = 0.0;
        this.frameId = 0;
        
        // --- Playback (Output) Ring Buffer State ---
        // Max capacity: 4 frames @ 16 kHz = 2048 samples (128 ms)
        this.maxPlaybackCapacity = 2048;
        this.playbackRingBuffer = new Float32Array(this.maxPlaybackCapacity * 2);
        this.playbackReadPos = 0;
        this.playbackWritePos = 0;
        this.playbackAvailable = 0;
        this.outputResamplePos = 0.0;
        
        this.isMuted = false;
        this.staleFramesDropped = 0;
        
        this.port.onmessage = (event) => {
            const data = event.data;
            if (!data) return;
            
            if (data.type === 'enhanced_frame') {
                this.handleIncomingEnhancedAudio(data.samples, data.frame_id);
            } else if (data.type === 'flush') {
                this.flushPlaybackBuffer();
            } else if (data.type === 'set_mute') {
                this.isMuted = !!data.isMuted;
            }
        };
    }
    
    flushPlaybackBuffer() {
        this.playbackReadPos = 0;
        this.playbackWritePos = 0;
        this.playbackAvailable = 0;
        this.outputResamplePos = 0.0;
    }
    
    handleIncomingEnhancedAudio(samples, frameId) {
        if (!samples || samples.length === 0 || this.isMuted) return;
        
        const len = samples.length;
        
        // Bounded Jitter Buffer Management:
        // If buffer already has > 3 frames (1536 samples / 96 ms), drop oldest frame to maintain low latency!
        if (this.playbackAvailable + len > this.maxPlaybackCapacity) {
            const dropCount = (this.playbackAvailable + len) - this.maxPlaybackCapacity;
            this.playbackReadPos = (this.playbackReadPos + dropCount) % this.playbackRingBuffer.length;
            this.playbackAvailable -= dropCount;
            this.staleFramesDropped++;
        }
        
        // Write new samples into circular ring buffer
        for (let i = 0; i < len; i++) {
            this.playbackRingBuffer[this.playbackWritePos] = samples[i];
            this.playbackWritePos = (this.playbackWritePos + 1) % this.playbackRingBuffer.length;
        }
        this.playbackAvailable += len;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        const output = outputs[0];
        
        const inChannel = (input && input.length > 0) ? input[0] : null;
        const outChannel = (output && output.length > 0) ? output[0] : null;
        const blockSize = outChannel ? outChannel.length : 128;
        
        const hardwareRate = sampleRate; // Global in AudioWorkletGlobalScope
        const downsampleRatio = hardwareRate / this.targetRate;
        const upsampleRatio = this.targetRate / hardwareRate;
        
        // ----------------------------------------------------
        // 1. CAPTURE & DOWN-SAMPLE (Hardware Rate -> 16,000 Hz)
        // ----------------------------------------------------
        if (inChannel && inChannel.length > 0) {
            let inIdx = this.inputResamplePos;
            while (inIdx < inChannel.length) {
                const i0 = Math.floor(inIdx);
                const i1 = Math.min(i0 + 1, inChannel.length - 1);
                const frac = inIdx - i0;
                const sample = inChannel[i0] * (1 - frac) + inChannel[i1] * frac;
                
                this.inputAccumulator[this.inputAccumCount++] = sample;
                inIdx += downsampleRatio;
                
                // When full 512-sample frame @ 16 kHz is ready, dispatch immediately
                if (this.inputAccumCount >= this.frameSize) {
                    const frameSlice = Array.from(this.inputAccumulator.subarray(0, this.frameSize));
                    
                    this.port.postMessage({
                        type: 'audio_frame',
                        frame_id: ++this.frameId,
                        timestamp: currentTime,
                        sample_rate: 16000,
                        samples: frameSlice
                    });
                    
                    // Shift leftover accumulator samples (if any)
                    const remaining = this.inputAccumCount - this.frameSize;
                    if (remaining > 0) {
                        this.inputAccumulator.copyWithin(0, this.frameSize, this.inputAccumCount);
                    }
                    this.inputAccumCount = remaining;
                }
            }
            this.inputResamplePos = inIdx - inChannel.length;
        }
        
        // ----------------------------------------------------
        // 2. PLAYBACK & UP-SAMPLE (16,000 Hz -> Hardware Rate)
        // ----------------------------------------------------
        if (outChannel) {
            if (this.isMuted || this.playbackAvailable <= 0) {
                outChannel.fill(0);
            } else {
                for (let k = 0; k < blockSize; k++) {
                    if (this.playbackAvailable <= 0) {
                        outChannel[k] = 0;
                        continue;
                    }
                    
                    // Read sample from 16kHz ring buffer with linear interpolation
                    const r0 = this.playbackReadPos;
                    const r1 = (r0 + 1) % this.playbackRingBuffer.length;
                    const frac = this.outputResamplePos;
                    
                    const s0 = this.playbackRingBuffer[r0];
                    const s1 = this.playbackRingBuffer[r1];
                    const outSample = s0 * (1 - frac) + s1 * frac;
                    
                    // Soft clip output for safety
                    outChannel[k] = Math.max(-1.0, Math.min(1.0, outSample));
                    
                    this.outputResamplePos += upsampleRatio;
                    if (this.outputResamplePos >= 1.0) {
                        const step = Math.floor(this.outputResamplePos);
                        this.outputResamplePos -= step;
                        this.playbackReadPos = (this.playbackReadPos + step) % this.playbackRingBuffer.length;
                        this.playbackAvailable = Math.max(0, this.playbackAvailable - step);
                    }
                }
            }
        }
        
        return true;
    }
}

registerProcessor('aeris-processor', AerisAudioProcessor);
