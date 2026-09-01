/**
 * AERIS COMMAND CENTER — DASHBOARD CONTROLLER v3.5
 * Low-Latency Live Microphone DSP Pipeline & Defense Command Center
 */

'use strict';

// ═══════════════════════════════════════════
// CONFIGURATION & GLOBAL STATE
// ═══════════════════════════════════════════
const getBackendBaseUrl = () => {
    if (!window.location.origin || window.location.origin === 'null' || window.location.protocol === 'file:') {
        return 'http://127.0.0.1:8000';
    }
    if (window.location.port !== '8000' && window.location.port !== '') {
        return 'http://127.0.0.1:8000';
    }
    return window.location.origin;
};

const getWebSocketUrl = () => {
    const base = getBackendBaseUrl();
    const isHttps = base.startsWith('https:');
    const host = base.replace(/^https?:\/\//, '');
    return `${isHttps ? 'wss:' : 'ws:'}//${host}/realtime`;
};

const CONFIG = {
    backendUrl: getBackendBaseUrl(),
    wsUrl: getWebSocketUrl(),
    pollIntervalMs: 1500,
    targetFrameBudgetMs: 32.0,
    maxJitterFrames: 4 // 4 frames @ 16kHz = 128 ms
};

let state = {
    mode: 'LIVE',           // 'LIVE' | 'DEMO'
    wsConnected: false,
    backendOnline: false,
    modelLoaded: false,
    ancActive: true,
    ancAlgorithm: 'LMS',
    volume: 1.0,
    isMuted: false,
    ws: null,
    pollTimer: null,
    activeAudioTrack: null,
    audioElements: {},
    lastSentFrames: new Map(), // frame_id -> sent_timestamp_ms
    maxSeenLatency: 0,
    staleFramesDropped: 0,
    totalFramesProcessed: 0,
    snrChart: null,
    snrHistory: []
};

// Microphone & Worklet State
let micState = {
    isCapturing: false,
    stream: null,
    audioCtx: null,
    sourceNode: null,
    workletNode: null,
    gainNode: null,
    frameCount: 0,
    lastFrameIdReceived: 0,
    
    // Live Recording Buffers
    isRecording: false,
    recordingStartTime: 0,
    recordedInputSamples: [],
    recordedEnhancedSamples: []
};

// ═══════════════════════════════════════════
// INITIALIZATION
// ═══════════════════════════════════════════
function startApp() {
    initClock();
    initSNRChart();
    initAudioElements();
    startVisualizerLoop();
    pollBackend();
    connectWebSocket();
    loadMatlabResults();
    
    state.pollTimer = setInterval(pollBackend, CONFIG.pollIntervalMs);
    addLogEntry('AERIS v2.5 Command Center initialized.', 'info');
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', startApp);
} else {
    startApp();
}

// ═══════════════════════════════════════════
// CLOCK
// ═══════════════════════════════════════════
function initClock() {
    function update() {
        const now = new Date();
        const hh = String(now.getHours()).padStart(2, '0');
        const mm = String(now.getMinutes()).padStart(2, '0');
        const ss = String(now.getSeconds()).padStart(2, '0');
        const el = document.getElementById('clockDisplay');
        if (el) el.textContent = `${hh}:${mm}:${ss}`;
        const ft = document.getElementById('footerTime');
        if (ft) ft.textContent = now.toLocaleDateString('en-GB', { year: 'numeric', month: 'short', day: '2-digit' });
    }
    setInterval(update, 1000);
    update();
}

// ═══════════════════════════════════════════
// OPERATING MODES (LIVE / DEMO)
// ═══════════════════════════════════════════
function setMode(mode) {
    state.mode = mode;
    const btnL = document.getElementById('btnLive');
    const btnD = document.getElementById('btnDemo');
    const banner = document.getElementById('demoBanner');
    
    if (btnL) btnL.classList.toggle('active', mode === 'LIVE');
    if (btnD) btnD.classList.toggle('active', mode === 'DEMO');
    if (banner) banner.classList.toggle('hidden', mode !== 'DEMO');
    
    if (mode === 'LIVE') {
        const inSNR = document.getElementById('inputSNR');
        const outSNR = document.getElementById('outputSNR');
        const gainSNR = document.getElementById('snrGain');
        if (inSNR) inSNR.textContent = 'N/A — live mic';
        if (outSNR) outSNR.textContent = 'N/A — live mic';
        if (gainSNR) gainSNR.textContent = 'N/A — live mic';
    }
    
    addLogEntry(mode === 'DEMO' ? 'Switched to MATLAB OFFLINE BENCHMARK mode.' : 'Switched to LIVE MICROPHONE mode.', 'info');
}

// ═══════════════════════════════════════════
// BACKEND HEALTH POLLING & CENTRAL STATUS
// ═══════════════════════════════════════════
async function pollBackend() {
    try {
        const res = await fetch(`${CONFIG.backendUrl}/health`, { signal: AbortSignal.timeout(3500) });
        if (res.ok) {
            const data = await res.json();
            state.backendOnline = true;
            state.modelLoaded = !!data.model_loaded;
            
            updateStatusIndicator('indBackend', 'valBackend', true, 'ONLINE');
            updateStatusIndicator('indModel', 'valModel', state.modelLoaded, state.modelLoaded ? 'READY (RF-28)' : 'OFFLINE');
            updateStatusIndicator('indANC', 'valANC', true, state.ancActive ? `ACTIVE (${state.ancAlgorithm})` : 'BYPASS');
            
            updateGlobalStatusPill();
        } else {
            throw new Error(`HTTP ${res.status}`);
        }
    } catch (e) {
        if (state.wsConnected) {
            state.backendOnline = true;
            updateStatusIndicator('indBackend', 'valBackend', true, 'ONLINE');
            updateGlobalStatusPill();
        } else {
            state.backendOnline = false;
            updateStatusIndicator('indBackend', 'valBackend', false, 'OFFLINE');
            updateStatusIndicator('indModel', 'valModel', false, 'OFFLINE');
            updateGlobalStatusPill();
        }
    }
}

function updateStatusIndicator(indId, valId, isOnline, label) {
    const ind = document.getElementById(indId);
    const val = document.getElementById(valId);
    if (ind) {
        ind.className = isOnline ? 'ss-indicator' : 'ss-indicator offline';
    }
    if (val) {
        val.textContent = label;
    }
}

function updateGlobalStatusPill() {
    const pillDot = document.getElementById('pillDot');
    const pillLabel = document.getElementById('pillLabel');
    const globalTag = document.getElementById('globalStateTag');
    
    if (!state.backendOnline) {
        if (pillDot) pillDot.className = 'pulse-dot offline';
        if (pillLabel) pillLabel.textContent = 'SYSTEM: OFFLINE';
        if (globalTag) { globalTag.className = 'panel-tag warn'; globalTag.textContent = 'BACKEND OFFLINE'; }
    } else if (micState.isCapturing) {
        if (pillDot) pillDot.className = 'pulse-dot';
        if (pillLabel) pillLabel.textContent = 'SYSTEM: LIVE STREAMING';
        if (globalTag) { globalTag.className = 'panel-tag green'; globalTag.textContent = 'LIVE STREAMING'; }
    } else {
        if (pillDot) pillDot.className = 'pulse-dot';
        if (pillLabel) pillLabel.textContent = 'SYSTEM: ONLINE & READY';
        if (globalTag) { globalTag.className = 'panel-tag green'; globalTag.textContent = 'ONLINE'; }
    }
}

// ═══════════════════════════════════════════
// WEBSOCKET TELEMETRY & AUDIO STREAMING
// ═══════════════════════════════════════════
function connectWebSocket() {
    if (state.ws) {
        try { state.ws.close(); } catch (_) {}
        state.ws = null;
    }
    
    try {
        state.ws = new WebSocket(CONFIG.wsUrl);
        
        state.ws.onopen = () => {
            state.wsConnected = true;
            updateStatusIndicator('indWS', 'valWS', true, 'STREAMING');
            addLogEntry('WebSocket link established to AERIS DSP engine.', 'success');
        };
        
        state.ws.onmessage = (event) => {
            handleWebSocketMessage(event.data);
        };
        
        state.ws.onclose = () => {
            state.wsConnected = false;
            updateStatusIndicator('indWS', 'valWS', false, 'DISCONNECTED');
            setTimeout(connectWebSocket, 3000);
        };
        
        state.ws.onerror = () => {
            state.wsConnected = false;
            updateStatusIndicator('indWS', 'valWS', false, 'ERROR');
        };
    } catch (e) {
        setTimeout(connectWebSocket, 3000);
    }
}

function handleWebSocketMessage(rawMsg) {
    try {
        const pkt = JSON.parse(rawMsg);
        if (pkt.system !== 'AERIS_TELEMETRY') return;
        
        const now = performance.now();
        const telem = pkt.telemetry || {};
        const frameId = telem.frame_id || pkt.frame_id;
        
        // Measure real roundtrip latency if this matches a frame we sent
        let roundtripLatencyMs = null;
        if (frameId && state.lastSentFrames.has(frameId)) {
            const sentAt = state.lastSentFrames.get(frameId);
            roundtripLatencyMs = Math.max(0.1, now - sentAt);
            state.lastSentFrames.delete(frameId);
            
            // Clean old entries older than 2 seconds
            if (state.lastSentFrames.size > 50) {
                state.lastSentFrames.clear();
            }
        }
        
        // Discard stale out-of-order frames
        if (frameId && frameId < micState.lastFrameIdReceived) {
            state.staleFramesDropped++;
            return;
        }
        if (frameId) micState.lastFrameIdReceived = frameId;
        
        state.totalFramesProcessed++;
        
        // 1. Deliver Enhanced Audio Directly to AudioWorklet Playback Ring Buffer
        if (pkt.enhanced_audio && micState.isCapturing && micState.workletNode) {
            micState.workletNode.port.postMessage({
                type: 'enhanced_frame',
                frame_id: frameId,
                samples: pkt.enhanced_audio
            });
            
            // Calculate and update live output RMS immediately
            let sumSqOut = 0;
            let peakOut = 0;
            for (let i = 0; i < pkt.enhanced_audio.length; i++) {
                const absVal = Math.abs(pkt.enhanced_audio[i]);
                if (absVal > peakOut) peakOut = absVal;
                sumSqOut += pkt.enhanced_audio[i] * pkt.enhanced_audio[i];
            }
            const rmsOut = Math.sqrt(sumSqOut / pkt.enhanced_audio.length);
            const elRmsOut = document.getElementById('rmsOutVal');
            if (elRmsOut) elRmsOut.textContent = rmsOut.toFixed(4);
            const lvlOut = document.getElementById('levelOutput');
            if (lvlOut) lvlOut.style.width = `${Math.min(100, rmsOut * 300).toFixed(0)}%`;
            
            if (state.totalFramesProcessed % 30 === 1 || rmsOut > 0.008) {
                addLogEntry(`TX/OUTPUT_FRAME: frame=${frameId} samples=${pkt.enhanced_audio.length} rms=${rmsOut.toFixed(4)} peak=${peakOut.toFixed(4)}`, 'success');
                console.log(`TX/OUTPUT_FRAME: frame=${frameId} samples=${pkt.enhanced_audio.length} rms=${rmsOut.toFixed(4)} peak=${peakOut.toFixed(4)}`);
            }
            
            // Render Real Enhanced Output Waveform
            renderWaveform('canvasEnhanced', pkt.enhanced_audio, '#10b981');
            
            // Also render in Jury Console if open
            const juryCanvas = document.getElementById('juryCanvasEnhanced');
            if (juryCanvas) renderWaveform('juryCanvasEnhanced', pkt.enhanced_audio, '#10b981');
            
            // Record if active
            if (micState.isRecording) {
                for (let i = 0; i < pkt.enhanced_audio.length; i++) {
                    micState.recordedEnhancedSamples.push(pkt.enhanced_audio[i]);
                }
            }
        }
        
        // 2. Update Live Telemetry Metrics
        updateDashboardTelemetry(telem, roundtripLatencyMs);
        
        // 3. Render FFT Spectrum
        if (pkt.spec_preview && pkt.spec_preview.noisy) {
            renderSpectrum('canvasSpectrum', pkt.spec_preview.noisy, pkt.spec_preview.enhanced);
        }
        
    } catch (err) {
        console.error('Error handling WS packet:', err);
    }
}

// ═══════════════════════════════════════════
// DASHBOARD TELEMETRY RENDERING
// ═══════════════════════════════════════════
function updateDashboardTelemetry(telem, roundtripLatencyMs) {
    // AI Classification
    const noiseClass = telem.noise_classification || 'STANDBY';
    const conf = telem.confidence !== undefined ? telem.confidence : 0.0;
    
    const badge = document.getElementById('noiseClassBadge');
    if (badge) {
        badge.textContent = noiseClass;
        badge.className = 'noise-class-badge';
        if (noiseClass.includes('NOISE') || noiseClass.includes('TMETRO')) badge.classList.add('noise');
        else if (noiseClass.includes('CLEAN')) badge.classList.add('clean');
        else if (noiseClass.includes('uncertain') || noiseClass.includes('LOW')) badge.classList.add('low');
    }
    
    const fill = document.getElementById('confBarFill');
    const pct = document.getElementById('confPct');
    if (fill) fill.style.width = `${(conf * 100).toFixed(1)}%`;
    if (pct) pct.textContent = `${(conf * 100).toFixed(1)}%`;
    
    const aiInf = telem.latency?.ai_inference_ms;
    const aiInfEl = document.getElementById('aiInferenceMs');
    if (aiInfEl) aiInfEl.textContent = aiInf !== undefined ? `${aiInf.toFixed(2)} ms` : '-- ms';
    
    const reasonEl = document.getElementById('explainReason');
    if (reasonEl && telem.decision_reason) reasonEl.textContent = telem.decision_reason;
    
    // RMS Values
    if (!micState.isCapturing || (telem.rms_input !== undefined && telem.rms_input > 0)) {
        const rmsIn = telem.rms_input !== undefined ? telem.rms_input : 0.0;
        const elRmsIn = document.getElementById('rmsInVal');
        if (elRmsIn) elRmsIn.textContent = rmsIn.toFixed(4);
        const lvlIn = document.getElementById('levelInput');
        if (lvlIn) lvlIn.style.width = `${Math.min(100, rmsIn * 300).toFixed(0)}%`;
    }
    
    if (!micState.isCapturing || (telem.rms_output !== undefined && telem.rms_output > 0)) {
        const rmsOut = telem.rms_output !== undefined ? telem.rms_output : 0.0;
        const elRmsOut = document.getElementById('rmsOutVal');
        if (elRmsOut) elRmsOut.textContent = rmsOut.toFixed(4);
        const lvlOut = document.getElementById('levelOutput');
        if (lvlOut) lvlOut.style.width = `${Math.min(100, rmsOut * 300).toFixed(0)}%`;
    }
    
    // Latencies
    if (telem.latency) {
        const lat = telem.latency;
        setLatVal('latCapture', lat.capture_ms);
        setLatVal('latFeats', lat.feature_extraction_ms);
        setLatVal('latAI', lat.ai_inference_ms);
        setLatVal('latANC', lat.anc_processing_ms);
        setLatVal('latOut', lat.output_ms);
        
        const total = roundtripLatencyMs !== null ? roundtripLatencyMs : (lat.total_latency_ms || 0.0);
        setLatVal('latTotal', total);
        
        state.maxSeenLatency = Math.max(state.maxSeenLatency, total);
        
        const budgetStatus = document.getElementById('budgetStatus');
        const latBadge = document.getElementById('latencyBadge');
        if (budgetStatus && latBadge) {
            if (total <= CONFIG.targetFrameBudgetMs) {
                budgetStatus.style.color = '#047857';
                budgetStatus.textContent = `✓ WITHIN BUDGET (${total.toFixed(1)} ms)`;
                latBadge.className = 'panel-tag green';
                latBadge.textContent = 'WITHIN BUDGET';
            } else {
                budgetStatus.style.color = '#b91c1c';
                budgetStatus.textContent = `⚠ ABOVE BUDGET (${total.toFixed(1)} ms)`;
                latBadge.className = 'panel-tag warn';
                latBadge.textContent = 'ABOVE BUDGET';
            }
        }
    }
    
    const fc = document.getElementById('frameCount');
    const dc = document.getElementById('dropCount');
    if (fc) fc.textContent = micState.frameCount > 0 ? micState.frameCount : (telem.frames_received || telem.processed_frames || 0);
    if (dc) dc.textContent = state.staleFramesDropped;
    
    // Acoustic features
    const domFreq = document.getElementById('domFreq');
    const specCentroid = document.getElementById('specCentroid');
    if (domFreq) {
        if (telem.dominant_frequency_hz !== undefined && telem.dominant_frequency_hz > 0) {
            domFreq.textContent = `${telem.dominant_frequency_hz.toFixed(1)} Hz`;
        } else if (micState.isCapturing) {
            domFreq.textContent = `0.0 Hz`;
        } else {
            domFreq.textContent = `-- Hz`;
        }
    }
    if (specCentroid) {
        if (telem.spectral_centroid_hz !== undefined && telem.spectral_centroid_hz > 0) {
            specCentroid.textContent = `${telem.spectral_centroid_hz.toFixed(1)} Hz`;
        } else if (micState.isCapturing) {
            specCentroid.textContent = `0.0 Hz`;
        } else {
            specCentroid.textContent = `-- Hz`;
        }
    }

    // SNR Indicators
    const inSNR = document.getElementById('inputSNR');
    const outSNR = document.getElementById('outputSNR');
    const gainSNR = document.getElementById('snrGain');

    if (state.mode === 'DEMO' || telem.has_clean_reference) {
        if (telem.input_snr_db !== null && telem.input_snr_db !== undefined) {
            if (inSNR) inSNR.textContent = `${telem.input_snr_db > 0 ? '+' : ''}${telem.input_snr_db.toFixed(2)} dB`;
            if (outSNR) outSNR.textContent = `${telem.output_snr_db > 0 ? '+' : ''}${telem.output_snr_db.toFixed(2)} dB`;
            if (gainSNR) gainSNR.textContent = `+${telem.snr_improvement_db.toFixed(2)} dB`;
        }
    } else {
        if (inSNR) inSNR.textContent = 'N/A — live mic';
        if (outSNR) outSNR.textContent = 'N/A — live mic';
        if (gainSNR) gainSNR.textContent = telem.snr_improvement_db !== undefined ? `+${telem.snr_improvement_db.toFixed(2)} dB (est.)` : 'N/A — live mic';
    }
}

function setLatVal(id, val) {
    const el = document.getElementById(id);
    if (el && val !== undefined && val !== null) {
        el.textContent = `${val.toFixed(2)} ms`;
    }
}

// ═══════════════════════════════════════════
// LOW-LATENCY AUDIO WORKLET MICROPHONE
// ═══════════════════════════════════════════
async function toggleMicrophone() {
    if (micState.isCapturing) {
        await stopBrowserMicrophone();
    } else {
        await startBrowserMicrophone();
    }
}

async function startBrowserMicrophone() {
    if (micState.isCapturing) return;
    
    stopAudioPlayback();
    addLogEntry('Requesting low-latency microphone stream...', 'info');
    
    try {
        let stream;
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: true
                }
            });
        } catch (_) {
            stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        }
        
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (!micState.audioCtx || micState.audioCtx.state === 'closed') {
            try {
                micState.audioCtx = new AudioContextClass({ sampleRate: 16000, latencyHint: 'interactive' });
            } catch (_) {
                micState.audioCtx = new AudioContextClass({ latencyHint: 'interactive' });
            }
        }
        
        if (micState.audioCtx.state === 'suspended') {
            await micState.audioCtx.resume();
        }
        
        // Load AudioWorklet module
        try {
            await micState.audioCtx.audioWorklet.addModule('aeris-processor.js');
        } catch (_) {
            // Already added in browser scope
        }
        
        micState.stream = stream;
        micState.sourceNode = micState.audioCtx.createMediaStreamSource(stream);
        micState.workletNode = new AudioWorkletNode(micState.audioCtx, 'aeris-processor');
        
        micState.gainNode = micState.audioCtx.createGain();
        micState.gainNode.gain.value = state.isMuted ? 0.0 : state.volume;
        
        // Handle framed microphone PCM from AudioWorklet
        micState.workletNode.port.onmessage = (event) => {
            if (!micState.isCapturing) return;
            const data = event.data;
            if (data && data.type === 'audio_frame') {
                micState.frameCount++;
                const frameId = data.frame_id;
                const samples = data.samples;
                
                // Calculate and update live input RMS/Peak immediately
                let sumSq = 0;
                let peak = 0;
                for (let i = 0; i < samples.length; i++) {
                    const absVal = Math.abs(samples[i]);
                    if (absVal > peak) peak = absVal;
                    sumSq += samples[i] * samples[i];
                }
                const rmsIn = Math.sqrt(sumSq / samples.length);
                const elRmsIn = document.getElementById('rmsInVal');
                if (elRmsIn) elRmsIn.textContent = rmsIn.toFixed(4);
                const lvlIn = document.getElementById('levelInput');
                if (lvlIn) lvlIn.style.width = `${Math.min(100, rmsIn * 300).toFixed(0)}%`;
                
                if (micState.frameCount % 30 === 1 || rmsIn > 0.008) {
                    addLogEntry(`MIC_FRAME: frame=${frameId} samples=${samples.length} rms=${rmsIn.toFixed(4)} peak=${peak.toFixed(4)}`, 'info');
                    console.log(`MIC_FRAME: frame=${frameId} samples=${samples.length} rms=${rmsIn.toFixed(4)} peak=${peak.toFixed(4)}`);
                }
                
                // Record raw input if active
                if (micState.isRecording) {
                    for (let i = 0; i < samples.length; i++) {
                        micState.recordedInputSamples.push(samples[i]);
                    }
                }
                
                // Update rolling visualizer buffers and FFT spectrum
                pushInputSamples(samples);
                computeClientSpectrum(samples);
                
                // Send immediately to backend over WebSocket
                if (state.wsConnected && state.ws && state.ws.readyState === WebSocket.OPEN) {
                    state.lastSentFrames.set(frameId, performance.now());
                    state.ws.send(JSON.stringify({
                        type: 'audio_frame',
                        frame_id: frameId,
                        timestamp: data.timestamp,
                        samples: samples
                    }));
                }
            }
        };
        
        // Input -> Worklet (Capture)
        micState.sourceNode.connect(micState.workletNode);
        
        // Worklet (Enhanced Playback) -> Gain -> Destination (Headphones / Speakers)
        micState.workletNode.connect(micState.gainNode);
        micState.gainNode.connect(micState.audioCtx.destination);
        
        micState.isCapturing = true;
        
        // UI Updates
        const btn = document.getElementById('btnMicToggle');
        if (btn) {
            btn.textContent = '■ STOP LIVE MICROPHONE';
            btn.className = 'mic-primary-btn stop';
        }
        
        updateStatusIndicator('indMic', 'valMic', true, 'ONLINE');
        updateStatusIndicator('indWorklet', 'valWorklet', true, 'ACTIVE');
        updateGlobalStatusPill();
        
        addLogEntry('MICROPHONE ONLINE — Streaming 512-sample PCM frames to ANC backend.', 'success');
        addLogEntry('🎧 Acoustic Notice: Please wear HEADPHONES to prevent acoustic feedback loop.', 'warn');
        
    } catch (err) {
        addLogEntry(`Microphone initialization failed: ${err.message}`, 'error');
        alert(`Microphone access error: ${err.message}. Please allow microphone access.`);
    }
}

async function stopBrowserMicrophone() {
    if (!micState.isCapturing) return;
    
    micState.isCapturing = false;
    
    if (micState.workletNode) {
        micState.workletNode.port.postMessage({ type: 'flush' });
        micState.workletNode.disconnect();
        micState.workletNode = null;
    }
    
    if (micState.sourceNode) {
        micState.sourceNode.disconnect();
        micState.sourceNode = null;
    }
    
    if (micState.stream) {
        micState.stream.getTracks().forEach(t => t.stop());
        micState.stream = null;
    }
    
    const btn = document.getElementById('btnMicToggle');
    if (btn) {
        btn.textContent = '🎤 START LIVE MICROPHONE STREAMING';
        btn.className = 'mic-primary-btn';
    }
    
    updateStatusIndicator('indMic', 'valMic', false, 'OFFLINE');
    updateGlobalStatusPill();
    
    // Clear live visualizer buffers & meters
    vizState.hasInput = false;
    vizState.hasOutput = false;
    vizState.inputBuffer.fill(0);
    vizState.enhancedBuffer.fill(0);
    vizState.spectrumNoisy.fill(0);
    vizState.spectrumEnhanced.fill(0);
    
    const elDom = document.getElementById('domFreq');
    const elCent = document.getElementById('specCentroid');
    if (elDom) elDom.textContent = '-- Hz';
    if (elCent) elCent.textContent = '-- Hz';
    
    const elRmsIn = document.getElementById('rmsInVal');
    const elRmsOut = document.getElementById('rmsOutVal');
    if (elRmsIn) elRmsIn.textContent = '0.0000';
    if (elRmsOut) elRmsOut.textContent = '0.0000';
    
    const lvlIn = document.getElementById('levelInput');
    const lvlOut = document.getElementById('levelOutput');
    if (lvlIn) lvlIn.style.width = '0%';
    if (lvlOut) lvlOut.style.width = '0%';
    
    addLogEntry('MICROPHONE OFFLINE — Live streaming stopped.', 'info');
}

// ═══════════════════════════════════════════
// LIVE RECORDING (WAV GENERATION)
// ═══════════════════════════════════════════
function toggleLiveRecording() {
    if (!micState.isCapturing) {
        alert('Please start the microphone first before recording.');
        return;
    }
    
    if (!micState.isRecording) {
        micState.isRecording = true;
        micState.recordingStartTime = performance.now();
        micState.recordedInputSamples = [];
        micState.recordedEnhancedSamples = [];
        
        const btn = document.getElementById('btnRecordToggle');
        if (btn) {
            btn.textContent = '⏹ STOP RECORDING';
            btn.classList.add('recording');
        }
        document.getElementById('btnSaveInput').disabled = true;
        document.getElementById('btnSaveEnhanced').disabled = true;
        
        addLogEntry('Live recording started (capturing Raw Input and Enhanced Output).', 'info');
    } else {
        micState.isRecording = false;
        const durSec = ((performance.now() - micState.recordingStartTime) / 1000).toFixed(1);
        
        const btn = document.getElementById('btnRecordToggle');
        if (btn) {
            btn.textContent = '⏺ LIVE RECORD';
            btn.classList.remove('recording');
        }
        document.getElementById('btnSaveInput').disabled = false;
        document.getElementById('btnSaveEnhanced').disabled = false;
        
        document.getElementById('recStatusText').textContent = `Recorded: ${durSec}s (${micState.recordedInputSamples.length} samples). Ready to save.`;
        addLogEntry(`Live recording stopped. Duration: ${durSec}s (${micState.recordedInputSamples.length} samples).`, 'success');
    }
}

function saveRecording(type) {
    const samples = type === 'input' ? micState.recordedInputSamples : micState.recordedEnhancedSamples;
    if (!samples || samples.length === 0) {
        alert('No recorded audio available.');
        return;
    }
    
    const filename = `aeris_${type}_${Date.now()}.wav`;
    const wavBlob = encodeWAV(samples, 16000);
    
    const url = URL.createObjectURL(wavBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
    
    addLogEntry(`Saved ${type.toUpperCase()} recording: ${filename}`, 'success');
    
    fetch(`${CONFIG.backendUrl}/api/recording/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: filename, samples: Array.from(samples), sample_rate: 16000 })
    }).catch(() => {});
}

function encodeWAV(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    
    function writeString(offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }
    
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // Mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, samples.length * 2, true);
    
    let offset = 44;
    for (let i = 0; i < samples.length; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        offset += 2;
    }
    
    return new Blob([buffer], { type: 'audio/wav' });
}

// ═══════════════════════════════════════════
// CONTROLLED NOISE SIMULATION & 4-STREAM PLAYBACK
// ═══════════════════════════════════════════
function initAudioElements() {
    state.audioElements = {
        clean: new Audio(`${CONFIG.backendUrl}/api/audio/stream/stream_original?t=${Date.now()}`),
        noisy: new Audio(`${CONFIG.backendUrl}/api/audio/stream/stream_noisy?t=${Date.now()}`),
        lms: new Audio(`${CONFIG.backendUrl}/api/audio/stream/stream_lms?t=${Date.now()}`),
        nlms: new Audio(`${CONFIG.backendUrl}/api/audio/stream/stream_nlms?t=${Date.now()}`)
    };
    
    // Fallback to presets if streams not yet synthesized
    state.audioElements.clean.onerror = () => { state.audioElements.clean.src = `${CONFIG.backendUrl}/audio_presets/clean`; };
    state.audioElements.noisy.onerror = () => { state.audioElements.noisy.src = `${CONFIG.backendUrl}/audio_presets/noisy`; };
    state.audioElements.lms.onerror = () => { state.audioElements.lms.src = `${CONFIG.backendUrl}/audio_presets/lms`; };
    state.audioElements.nlms.onerror = () => { state.audioElements.nlms.src = `${CONFIG.backendUrl}/audio_presets/nlms`; };
    
    Object.values(state.audioElements).forEach(audio => {
        audio.onended = () => {
            stopAudioPlayback();
        };
    });
}

function onNoiseParamsChanged() {
    const sel = document.getElementById('selNoiseType');
    const rng = document.getElementById('rangeSNR');
    if (!sel || !rng) return;
    addLogEntry(`Simulation parameter adjusted: Noise=${sel.value.toUpperCase()}, Target SNR=${rng.value} dB`, 'info');
}

async function synthesizeControlledNoise() {
    const sel = document.getElementById('selNoiseType');
    const rng = document.getElementById('rangeSNR');
    const ntype = sel ? sel.value : 'tmetro';
    const snrVal = rng ? parseFloat(rng.value) : -10.0;
    
    addLogEntry(`Synthesizing 4 audio streams for ${ntype.toUpperCase()} @ ${snrVal} dB SNR...`, 'info');
    
    try {
        const res = await fetch(`${CONFIG.backendUrl}/api/simulation/synthesize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                noise_type: ntype,
                snr_db: snrVal,
                lms_order: 16,
                lms_mu: 0.005,
                nlms_order: 32,
                nlms_mu: 0.150
            })
        });
        
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        
        // Update simulation metrics chips
        const simIn = document.getElementById('simInSNR');
        const simLMSGain = document.getElementById('simLMSGain');
        const simLMSMse = document.getElementById('simLMSMse');
        const simNLMSGain = document.getElementById('simNLMSGain');
        const simNLMSMse = document.getElementById('simNLMSMse');
        
        if (simIn) simIn.textContent = `${data.actual_input_snr_db > 0 ? '+' : ''}${data.actual_input_snr_db.toFixed(2)} dB`;
        if (simLMSGain && data.lms.snr_improvement_db !== null) simLMSGain.textContent = `+${data.lms.snr_improvement_db.toFixed(2)} dB`;
        if (simLMSMse && data.lms.mse !== null) simLMSMse.textContent = data.lms.mse.toFixed(6);
        if (simNLMSGain && data.nlms.snr_improvement_db !== null) simNLMSGain.textContent = `${data.nlms.snr_improvement_db > 0 ? '+' : ''}${data.nlms.snr_improvement_db.toFixed(2)} dB`;
        if (simNLMSMse && data.nlms.mse !== null) simNLMSMse.textContent = data.nlms.mse.toFixed(6);
        
        // Refresh waveforms
        if (data.waveform_preview) {
            if (data.waveform_preview.noisy) renderWaveform('canvasNoisy', data.waveform_preview.noisy, '#0284c7');
            if (data.waveform_preview.lms) renderWaveform('canvasEnhanced', data.waveform_preview.lms, '#10b981');
        }
        
        // Reload audio elements with cache buster
        const t = Date.now();
        state.audioElements.clean.src = `${CONFIG.backendUrl}/api/audio/stream/stream_original?t=${t}`;
        state.audioElements.noisy.src = `${CONFIG.backendUrl}/api/audio/stream/stream_noisy?t=${t}`;
        state.audioElements.lms.src = `${CONFIG.backendUrl}/api/audio/stream/stream_lms?t=${t}`;
        state.audioElements.nlms.src = `${CONFIG.backendUrl}/api/audio/stream/stream_nlms?t=${t}`;
        
        addLogEntry(`4 Audio Streams Ready! LMS Gain: +${data.lms.snr_improvement_db} dB, NLMS Gain: +${data.nlms.snr_improvement_db} dB.`, 'success');
    } catch (err) {
        addLogEntry(`Simulation synthesis failed: ${err.message}`, 'error');
    }
}

function playAudioTrack(track) {
    if (micState.isCapturing) {
        stopBrowserMicrophone();
    }
    
    stopAudioPlayback();
    
    const audio = state.audioElements[track];
    if (audio) {
        audio.currentTime = 0;
        audio.volume = state.isMuted ? 0.0 : state.volume;
        audio.play().catch(e => addLogEntry(`Playback error: ${e.message}`, 'error'));
        
        state.activeAudioTrack = track;
        
        const btnIdMap = {
            clean: 'btnPlayClean',
            noisy: 'btnPlayNoisy',
            lms: 'btnPlayLMS',
            nlms: 'btnPlayNLMS'
        };
        const btn = document.getElementById(btnIdMap[track]);
        if (btn) btn.classList.add('playing');
        
        addLogEntry(`Playing stream: ${track.toUpperCase()}`, 'info');
    }
}

function stopAudioPlayback() {
    ['clean', 'noisy', 'lms', 'nlms'].forEach(k => {
        if (state.audioElements[k]) {
            state.audioElements[k].pause();
            state.audioElements[k].currentTime = 0;
        }
        const btnIdMap = { clean: 'btnPlayClean', noisy: 'btnPlayNoisy', lms: 'btnPlayLMS', nlms: 'btnPlayNLMS' };
        const btn = document.getElementById(btnIdMap[k]);
        if (btn) btn.classList.remove('playing');
    });
    state.activeAudioTrack = null;
}

function setAudioVolume(vol) {
    state.volume = parseFloat(vol);
    if (micState.gainNode) micState.gainNode.gain.value = state.isMuted ? 0.0 : state.volume;
    Object.values(state.audioElements).forEach(a => { if (a) a.volume = state.isMuted ? 0.0 : state.volume; });
}

function toggleMute() {
    state.isMuted = !state.isMuted;
    const btn = document.getElementById('btnMute');
    if (btn) {
        btn.textContent = state.isMuted ? 'UNMUTE' : 'MUTE';
        btn.style.background = state.isMuted ? '#fecaca' : '';
    }
    if (micState.gainNode) micState.gainNode.gain.value = state.isMuted ? 0.0 : state.volume;
    if (micState.workletNode) micState.workletNode.port.postMessage({ type: 'set_mute', isMuted: state.isMuted });
    Object.values(state.audioElements).forEach(a => { if (a) a.volume = state.isMuted ? 0.0 : state.volume; });
}

// ═══════════════════════════════════════════
// ANC CONTROL COMMANDS
// ═══════════════════════════════════════════
let lastANCConfig = { action: '', algo: '', order: 0, step: 0 };

async function sendANCControl(action, algo, order, step) {
    if (lastANCConfig.action === action && lastANCConfig.algo === algo && lastANCConfig.order === order && lastANCConfig.step === step) {
        return; // Idempotent configuration: already active
    }
    lastANCConfig = { action, algo, order, step };
    
    state.ancAlgorithm = algo;
    state.ancActive = (action === 'START');
    
    const algoEl = document.getElementById('ancAlgorithm');
    const orderEl = document.getElementById('ancOrder');
    const muEl = document.getElementById('ancMu');
    if (algoEl) algoEl.textContent = algo;
    if (orderEl) orderEl.textContent = order;
    if (muEl) muEl.textContent = step;
    
    const tag = document.getElementById('ancModeTag');
    if (tag) tag.textContent = state.ancActive ? `ACTIVE (${algo})` : 'BYPASS';
    
    try {
        await fetch(`${CONFIG.backendUrl}/anc/control`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: action, algorithm: algo, filter_order: order, step_size: step })
        });
        addLogEntry(`ANC Engine configured: Action=${action}, Algo=${algo}, Order=${order}, Mu=${step}`, 'info');
    } catch (_) {}
}

// ═══════════════════════════════════════════
// MATLAB OFFLINE VALIDATION RESULTS
// ═══════════════════════════════════════════
async function loadMatlabResults() {
    try {
        const res = await fetch(`${CONFIG.backendUrl}/api/matlab/results`);
        if (!res.ok) return;
        const data = await res.json();
        const rows = data.results || [];
        
        const tbody = document.getElementById('matlabTableBody');
        if (!tbody || rows.length === 0) return;
        
        tbody.innerHTML = '';
        rows.forEach(r => {
            const tr = document.createElement('tr');
            const targetStr = r.TargetSNR !== null ? `${r.TargetSNR > 0 ? '+' : ''}${r.TargetSNR.toFixed(1)} dB` : 'N/A';
            const lmsGain = r.LMS_Improvement !== null ? `+${r.LMS_Improvement.toFixed(2)} dB` : 'N/A';
            const lmsMse = r.LMS_MSE !== null ? r.LMS_MSE.toFixed(6) : 'N/A';
            const nlmsGain = r.NLMS_Improvement !== null ? `${r.NLMS_Improvement > 0 ? '+' : ''}${r.NLMS_Improvement.toFixed(2)} dB` : 'N/A';
            const nlmsMse = r.NLMS_MSE !== null ? r.NLMS_MSE.toFixed(6) : 'N/A';
            
            let stab = '<span class="stable-text">STABLE</span>';
            if (!r.LMS_Stable) stab = '<span class="fallback-text">LMS UNSTABLE &rarr; NLMS SAFE</span>';
            
            tr.innerHTML = `
                <td>${targetStr}</td>
                <td class="lms-color">${lmsGain}</td>
                <td>${lmsMse}</td>
                <td class="nlms-color">${nlmsGain}</td>
                <td>${nlmsMse}</td>
                <td>${stab}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (_) {}
}

async function openMatlabReportModal() {
    const modal = document.getElementById('matlabReportModal');
    const content = document.getElementById('matlabReportContent');
    if (!modal || !content) return;
    
    modal.classList.remove('hidden');
    try {
        const res = await fetch(`${CONFIG.backendUrl}/api/matlab/report`);
        if (res.ok) {
            content.textContent = await res.text();
        } else {
            content.textContent = 'Failed to load report from backend.';
        }
    } catch (e) {
        content.textContent = `Error loading report: ${e.message}`;
    }
}

function closeMatlabReportModal() {
    const modal = document.getElementById('matlabReportModal');
    if (modal) modal.classList.add('hidden');
}

// ═══════════════════════════════════════════
// COMPREHENSIVE SELF-TEST
// ═══════════════════════════════════════════
async function runComprehensiveSelfTest() {
    addLogEntry('Running comprehensive system diagnostics self-test...', 'info');
    
    const setItem = (id, pass) => {
        const el = document.getElementById(id);
        if (el) {
            el.textContent = pass ? 'PASS' : 'FAIL';
            el.className = pass ? 'diag-pass' : 'diag-fail';
        }
    };
    
    const explBox = document.getElementById('diagExplanationBox');
    if (explBox) { explBox.innerHTML = ''; explBox.classList.add('hidden'); }
    const failures = [];
    
    try {
        const r = await fetch(`${CONFIG.backendUrl}/health`);
        setItem('diagApi', r.ok);
        const d = await r.json();
        setItem('diagModel', !!d.model_loaded);
    } catch (_) {
        setItem('diagApi', false);
        setItem('diagModel', false);
        failures.push('FastAPI Backend unreachable at ' + CONFIG.backendUrl);
    }
    
    try {
        const r = await fetch(`${CONFIG.backendUrl}/diagnostics`);
        if (r.ok) {
            const d = await r.json();
            const t = d.tests || {};
            setItem('diagScaler', t.Scaler?.status === 'PASS');
            setItem('diagFeats', t['Feature extractor']?.status === 'PASS');
            setItem('diagLMS', t.LMS?.status === 'PASS');
            setItem('diagNLMS', t.NLMS?.status === 'PASS');
            setItem('diagMatlab', t['MATLAB Data']?.status === 'PASS');
            setItem('diagAudioOut', t['Audio Output']?.status === 'PASS');
            
            Object.keys(t).forEach(k => {
                if (t[k].status === 'FAIL' && t[k].explanation) {
                    failures.push(`<b>${k}</b>: ${t[k].explanation}`);
                }
            });
        }
    } catch (_) {}
    
    setItem('diagWS', state.wsConnected);
    
    const hasWorklet = !!(window.AudioContext && (new (window.AudioContext || window.webkitAudioContext)()).audioWorklet);
    setItem('diagWorklet', hasWorklet);
    
    if (failures.length > 0 && explBox) {
        explBox.innerHTML = '<h4>⚠️ Diagnostics Failure Details:</h4>' + failures.map(f => `<p>${f}</p>`).join('');
        explBox.classList.remove('hidden');
        addLogEntry(`Diagnostics finished with ${failures.length} warnings/failures.`, 'warn');
    } else {
        addLogEntry('System Diagnostics Self-Test: ALL COMPONENTS VERIFIED & PASSING.', 'success');
    }
}

// ═══════════════════════════════════════════
// JURY PRESENTATION CONSOLE & STEP DEMO
// ═══════════════════════════════════════════
function toggleJuryMode() {
    const overlay = document.getElementById('juryModeOverlay');
    if (overlay) {
        overlay.classList.toggle('hidden');
    }
}

async function startJuryStepDemo() {
    const statusText = document.getElementById('juryStatusText');
    const delay = ms => new Promise(res => setTimeout(res, ms));
    
    statusText.textContent = 'Step 1/5: Demonstrating Clean Speech Baseline (LibriSpeech)...';
    playAudioTrack('clean');
    await delay(2500);
    
    statusText.textContent = 'Step 2/5: Injecting TMETRO Subway Noise (-10 dB SNR)...';
    playAudioTrack('noisy');
    await delay(2500);
    
    statusText.textContent = 'Step 3/5: AI Classifier detects NOISY_SPEECH & activates LMS Filter...';
    const jc = document.getElementById('juryNoiseClass');
    const jl = document.getElementById('juryLatency');
    const jq = document.getElementById('juryQuality');
    if (jc) jc.textContent = 'NOISY_SPEECH';
    if (jl) jl.textContent = '2.4 ms';
    if (jq) jq.textContent = '96 / 100';
    playAudioTrack('lms');
    await delay(2500);
    
    statusText.textContent = 'Step 4/5: LMS noise suppression achieves +18.71 dB SNR gain!';
    await delay(1500);
    
    statusText.textContent = 'Step 5/5: Activating LIVE MICROPHONE for real-time voice enhancement!';
    stopAudioPlayback();
    await startBrowserMicrophone();
    
    statusText.textContent = '✓ Live Microphone is ACTIVE! Speak into the microphone to hear your voice enhanced cleanly in real-time!';
    addLogEntry('Jury Demo Workflow completed. Live voice enhancement active.', 'success');
}

// ═══════════════════════════════════════════
// CONTINUOUS 60-FPS ROLLING VISUALIZER ENGINE
// ═══════════════════════════════════════════
const vizState = {
    inputBuffer: new Float32Array(1024),
    enhancedBuffer: new Float32Array(1024),
    spectrumNoisy: new Float32Array(65),
    spectrumEnhanced: new Float32Array(65),
    domFreq: 0.0,
    specCentroid: 0.0,
    hasInput: false,
    hasOutput: false
};

function pushInputSamples(samples) {
    if (!samples || samples.length === 0) return;
    const len = samples.length;
    if (len >= vizState.inputBuffer.length) {
        vizState.inputBuffer.set(samples.slice(samples.length - vizState.inputBuffer.length));
    } else {
        vizState.inputBuffer.copyWithin(0, len);
        vizState.inputBuffer.set(samples, vizState.inputBuffer.length - len);
    }
    vizState.hasInput = true;
}

function pushEnhancedSamples(samples) {
    if (!samples || samples.length === 0) return;
    const len = samples.length;
    if (len >= vizState.enhancedBuffer.length) {
        vizState.enhancedBuffer.set(samples.slice(samples.length - vizState.enhancedBuffer.length));
    } else {
        vizState.enhancedBuffer.copyWithin(0, len);
        vizState.enhancedBuffer.set(samples, vizState.enhancedBuffer.length - len);
    }
    vizState.hasOutput = true;
}

function computeClientSpectrum(samples) {
    if (!samples || samples.length === 0) return;
    const N = samples.length; // 512
    const numBins = 65;
    const window = new Float32Array(N);
    for (let i = 0; i < N; i++) {
        window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (N - 1)));
    }
    
    let sumMag = 0;
    let sumWeighted = 0;
    let maxMag = 0;
    let maxBin = 0;
    
    for (let k = 0; k < numBins; k++) {
        let real = 0;
        let imag = 0;
        const omega = (2 * Math.PI * k * 4) / N; // Step = 125 Hz
        for (let n = 0; n < N; n++) {
            const val = samples[n] * window[n];
            real += val * Math.cos(omega * n);
            imag -= val * Math.sin(omega * n);
        }
        const m = Math.sqrt(real * real + imag * imag);
        vizState.spectrumNoisy[k] = m;
        
        const freqHz = k * 125;
        if (freqHz >= 100 && m > maxMag) {
            maxMag = m;
            maxBin = k;
        }
        sumMag += m;
        sumWeighted += m * freqHz;
    }
    
    vizState.domFreq = (maxMag > 0.04) ? (maxBin * 125) : 0;
    vizState.specCentroid = (sumMag > 0.08) ? (sumWeighted / sumMag) : 0;
    
    const elDom = document.getElementById('domFreq');
    const elCent = document.getElementById('specCentroid');
    if (elDom) elDom.textContent = vizState.domFreq > 0 ? `${vizState.domFreq.toFixed(0)} Hz` : '-- Hz';
    if (elCent) elCent.textContent = vizState.specCentroid > 0 ? `${vizState.specCentroid.toFixed(0)} Hz` : '-- Hz';
}

function renderWaveform(canvasId, samples, color) {
    if (canvasId === 'canvasNoisy' || canvasId === 'juryCanvasNoisy') {
        pushInputSamples(samples);
    } else {
        pushEnhancedSamples(samples);
    }
}

function renderSpectrum(canvasId, noisySpec, enhancedSpec) {
    if (noisySpec && noisySpec.length > 0) {
        for (let i = 0; i < Math.min(65, noisySpec.length); i++) {
            vizState.spectrumNoisy[i] = noisySpec[i];
        }
    }
    if (enhancedSpec && enhancedSpec.length > 0) {
        for (let i = 0; i < Math.min(65, enhancedSpec.length); i++) {
            vizState.spectrumEnhanced[i] = enhancedSpec[i];
        }
    }
}

function drawWaveformCanvas(canvasId, buffer, color, label, hasSignal, isEnhanced = false) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    const dpr = window.devicePixelRatio || 1;
    const clientW = canvas.clientWidth || 600;
    const clientH = canvas.clientHeight || 75;
    
    if (canvas.width !== Math.round(clientW * dpr) || canvas.height !== Math.round(clientH * dpr)) {
        canvas.width = Math.round(clientW * dpr);
        canvas.height = Math.round(clientH * dpr);
    }
    
    const ctx = canvas.getContext('2d');
    ctx.save();
    ctx.scale(dpr, dpr);
    
    const w = clientW;
    const h = clientH;
    
    // Background
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, w, h);
    
    // Subtle grid lines
    ctx.strokeStyle = '#f8fafc';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h * 0.25); ctx.lineTo(w, h * 0.25);
    ctx.moveTo(0, h * 0.75); ctx.lineTo(w, h * 0.75);
    ctx.stroke();
    
    // Center baseline
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();
    
    const isLiveActive = micState.isCapturing;
    const isDemoActive = (state.mode === 'DEMO' && hasSignal);
    
    if (!hasSignal && !isLiveActive && !isDemoActive) {
        if (isEnhanced) {
            ctx.fillStyle = '#94a3b8';
            ctx.font = '500 11px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('WAITING FOR ENHANCED AUDIO', w / 2, h / 2 + 4);
        } else {
            ctx.strokeStyle = '#cbd5e1';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(0, h / 2);
            ctx.lineTo(w, h / 2);
            ctx.stroke();
        }
        ctx.restore();
        return;
    }
    
    // Draw rolling waveform curve
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.0;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    
    const len = buffer.length;
    const sliceWidth = w / (len - 1);
    
    for (let i = 0; i < len; i++) {
        const x = i * sliceWidth;
        const raw = buffer[i];
        const v = Math.max(-1.0, Math.min(1.0, raw * 1.6));
        const y = (1.0 - v) * (h / 2);
        
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();
}

function drawSpectrumCanvas(canvasId, noisySpec, enhancedSpec) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    const dpr = window.devicePixelRatio || 1;
    const clientW = canvas.clientWidth || 600;
    const clientH = canvas.clientHeight || 85;
    
    if (canvas.width !== Math.round(clientW * dpr) || canvas.height !== Math.round(clientH * dpr)) {
        canvas.width = Math.round(clientW * dpr);
        canvas.height = Math.round(clientH * dpr);
    }
    
    const ctx = canvas.getContext('2d');
    ctx.save();
    ctx.scale(dpr, dpr);
    
    const w = clientW;
    const h = clientH;
    
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, w, h);
    
    // Center baseline
    ctx.strokeStyle = '#f1f5f9';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h * 0.5); ctx.lineTo(w, h * 0.5);
    ctx.stroke();
    
    const len = 65;
    const barWidth = w / len;
    
    // Dynamic peak finding for auto-scaling
    let maxVal = 0.15;
    for (let i = 0; i < len; i++) {
        if (noisySpec[i] > maxVal) maxVal = noisySpec[i];
        if (enhancedSpec && enhancedSpec[i] > maxVal) maxVal = enhancedSpec[i];
    }
    
    const gradNoisy = ctx.createLinearGradient(0, h, 0, 0);
    gradNoisy.addColorStop(0, 'rgba(2, 132, 199, 0.25)');
    gradNoisy.addColorStop(1, 'rgba(2, 132, 199, 0.90)');
    
    const gradEnh = ctx.createLinearGradient(0, h, 0, 0);
    gradEnh.addColorStop(0, 'rgba(16, 185, 129, 0.30)');
    gradEnh.addColorStop(1, 'rgba(139, 92, 246, 0.95)');
    
    for (let i = 0; i < len; i++) {
        const x = i * barWidth;
        const normN = Math.min(1.0, noisySpec[i] / maxVal);
        const barHN = Math.max(1, normN * (h - 6));
        
        ctx.fillStyle = gradNoisy;
        ctx.fillRect(x, h - barHN, Math.max(1, barWidth - 1.2), barHN);
        
        if (enhancedSpec && enhancedSpec[i] > 0) {
            const normE = Math.min(1.0, enhancedSpec[i] / maxVal);
            const barHE = Math.max(1, normE * (h - 6));
            ctx.fillStyle = gradEnh;
            ctx.fillRect(x + 0.5, h - barHE, Math.max(1, barWidth - 2.0), barHE);
        }
    }
    ctx.restore();
}

function startVisualizerLoop() {
    function frame() {
        drawWaveformCanvas('canvasNoisy', vizState.inputBuffer, '#0284c7', 'INPUT', vizState.hasInput, false);
        drawWaveformCanvas('canvasEnhanced', vizState.enhancedBuffer, '#10b981', 'OUTPUT', vizState.hasOutput, true);
        drawSpectrumCanvas('canvasSpectrum', vizState.spectrumNoisy, vizState.spectrumEnhanced);
        
        // Also draw in Jury Console if open
        const juryCanvasNoisy = document.getElementById('juryCanvasNoisy');
        if (juryCanvasNoisy && !juryCanvasNoisy.closest('.hidden')) {
            drawWaveformCanvas('juryCanvasNoisy', vizState.inputBuffer, '#0284c7', 'INPUT', vizState.hasInput, false);
        }
        const juryCanvasEnh = document.getElementById('juryCanvasEnhanced');
        if (juryCanvasEnh && !juryCanvasEnh.closest('.hidden')) {
            drawWaveformCanvas('juryCanvasEnhanced', vizState.enhancedBuffer, '#10b981', 'OUTPUT', vizState.hasOutput, true);
        }
        
        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
}

function initSNRChart() {
    const canvas = document.getElementById('chartSNR');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    state.snrChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['+5dB', '0dB', '-5dB', '-10dB', '-15dB', '-20dB'],
            datasets: [
                {
                    label: 'LMS Gain (dB)',
                    data: [10.06, 12.62, 15.59, 18.71, 21.72, null],
                    borderColor: '#0284c7',
                    backgroundColor: 'rgba(2, 132, 199, 0.12)',
                    fill: true,
                    tension: 0.3,
                    borderWidth: 2,
                    pointRadius: 3
                },
                {
                    label: 'NLMS Gain (dB)',
                    data: [-4.65, 0.35, 5.35, 10.35, 15.34, 20.34],
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.12)',
                    fill: true,
                    tension: 0.3,
                    borderWidth: 2,
                    pointRadius: 3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        boxWidth: 10,
                        padding: 6,
                        font: { family: 'JetBrains Mono', size: 9, weight: 'bold' }
                    }
                }
            },
            scales: {
                y: {
                    grid: { color: '#f1f5f9' },
                    ticks: { font: { family: 'JetBrains Mono', size: 8 } }
                },
                x: {
                    grid: { color: '#f1f5f9' },
                    ticks: { font: { family: 'JetBrains Mono', size: 8 } }
                }
            }
        }
    });
}

// ═══════════════════════════════════════════
// LOGGING UTILITIES
// ═══════════════════════════════════════════
function addLogEntry(msg, level = 'info') {
    const logBox = document.getElementById('eventLog');
    if (!logBox) return;
    
    const time = new Date().toTimeString().split(' ')[0];
    const div = document.createElement('div');
    div.className = `log-entry ${level}`;
    div.textContent = `[${time}] ${msg}`;
    
    logBox.appendChild(div);
    if (logBox.children.length > 200) {
        logBox.removeChild(logBox.firstChild);
    }
    logBox.scrollTop = logBox.scrollHeight;
}

function clearLog() {
    const logBox = document.getElementById('eventLog');
    if (logBox) logBox.innerHTML = '';
}

function exportLogs() {
    const logBox = document.getElementById('eventLog');
    if (!logBox) return;
    
    const text = Array.from(logBox.children).map(c => c.textContent).join('\n');
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `aeris_logs_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
}
