// RAGinGoa Frontend Application Logic
const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ? 'http://localhost:8000'
  : 'http://localhost:8000';

// DOM Elements
const apiStatusEl = document.getElementById('api-status');
const tabVoiceBtn = document.getElementById('tab-voice');
const tabTextBtn = document.getElementById('tab-text');
const panelVoice = document.getElementById('panel-voice');
const panelText = document.getElementById('panel-text');

const recordBtn = document.getElementById('record-btn');
const recordStateText = document.getElementById('record-state-text');
const recordTimer = document.getElementById('record-timer');
const micErrorBanner = document.getElementById('mic-error-banner');
const waveformCanvas = document.getElementById('waveform-canvas');

const textQueryInput = document.getElementById('text-query-input');
const sendTextBtn = document.getElementById('send-text-btn');

const stepperSection = document.getElementById('stepper-section');
const pipelineStatusTag = document.getElementById('pipeline-status-tag');

const resultsContainer = document.getElementById('results-container');
const transcriptPanel = document.getElementById('transcript-panel');
const transcriptText = document.getElementById('transcript-text');

const ttsToggleBtn = document.getElementById('tts-toggle-btn');
const answerContent = document.getElementById('answer-content');
const groundingBadge = document.getElementById('grounding-badge');
const shortCircuitBanner = document.getElementById('short-circuit-banner');
const flaggedClaimsWrapper = document.getElementById('flagged-claims-wrapper');
const flaggedClaimsList = document.getElementById('flagged-claims-list');

const totalLatencyBadge = document.getElementById('total-latency-badge');
const latencyBars = document.getElementById('latency-bars');
const citationsList = document.getElementById('citations-list');
const sourcesCount = document.getElementById('sources-count');

// Recording & Audio State
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let startTime = 0;
let timerInterval = null;
let audioCtx = null;
let analyser = null;
let animFrameId = null;

// Speech Synthesis (TTS) State
let isTtsEnabled = true;
let availableVoices = [];

// Initialization
document.addEventListener('DOMContentLoaded', () => {
  checkBackendHealth();
  initTabs();
  initAudioRecorder();
  initTextInput();
  initTTS();
});

// 1. Health Check
async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (res.ok) {
      const data = await res.json();
      apiStatusEl.textContent = 'ONLINE (200 OK)';
      apiStatusEl.style.color = '#10b981';
    } else {
      apiStatusEl.textContent = 'ERROR';
      apiStatusEl.style.color = '#ef4444';
    }
  } catch (err) {
    console.warn('Backend connection failed:', err);
    apiStatusEl.textContent = 'OFFLINE';
    apiStatusEl.style.color = '#ef4444';
  }
}

// 2. Tab Navigation
function initTabs() {
  tabVoiceBtn.addEventListener('click', () => {
    tabVoiceBtn.classList.add('active');
    tabTextBtn.classList.remove('active');
    panelVoice.classList.remove('hidden');
    panelText.classList.add('hidden');
  });

  tabTextBtn.addEventListener('click', () => {
    tabTextBtn.classList.add('active');
    tabVoiceBtn.classList.remove('active');
    panelText.classList.remove('hidden');
    panelVoice.classList.add('hidden');
  });
}

// 3. Audio Recorder Logic (Cross-Browser / iOS Safari Mime Fallback)
function initAudioRecorder() {
  recordBtn.addEventListener('click', async () => {
    stopSpeech(); // Cancel any active TTS speech immediately when mic button is tapped
    if (!isRecording) {
      await startRecording();
    } else {
      stopRecording();
    }
  });
}

function getSupportedMimeType() {
  const types = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/mp4',
    'audio/aac',
    'audio/wav'
  ];
  for (const t of types) {
    if (MediaRecorder.isTypeSupported(t)) {
      return t;
    }
  }
  return '';
}

async function startRecording() {
  micErrorBanner.classList.add('hidden');
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    
    const mimeType = getSupportedMimeType();
    const options = mimeType ? { mimeType } : {};
    
    mediaRecorder = new MediaRecorder(stream, options);
    audioChunks = [];

    // Setup Audio Visualizer
    setupWaveform(stream);

    mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };

    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: mimeType || 'audio/wav' });
      await submitAudioQuery(audioBlob);
      // Stop stream tracks
      stream.getTracks().forEach((track) => track.stop());
    };

    mediaRecorder.start();
    isRecording = true;
    recordBtn.classList.add('recording');
    recordStateText.textContent = 'Recording audio... Click button to stop & submit';
    
    startTime = Date.now();
    timerInterval = setInterval(updateTimer, 1000);
  } catch (err) {
    console.error('Mic error:', err);
    micErrorBanner.classList.remove('hidden');
    recordStateText.textContent = 'Microphone permission denied or unavailable';
  }
}

function stopRecording() {
  if (mediaRecorder && isRecording) {
    mediaRecorder.stop();
    isRecording = false;
    recordBtn.classList.remove('recording');
    recordStateText.textContent = 'Processing audio upload...';
    clearInterval(timerInterval);
    if (animFrameId) cancelAnimationFrame(animFrameId);
  }
}

function updateTimer() {
  const elapsed = Math.floor((Date.now() - startTime) / 1000);
  const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
  const secs = String(elapsed % 60).padStart(2, '0');
  recordTimer.textContent = `${mins}:${secs}`;
}

function setupWaveform(stream) {
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const source = audioCtx.createMediaStreamSource(stream);
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 64;
  source.connect(analyser);

  const canvasCtx = waveformCanvas.getContext('2d');
  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);

  function draw() {
    animFrameId = requestAnimationFrame(draw);
    analyser.getByteFrequencyData(dataArray);

    canvasCtx.fillStyle = 'rgba(10, 13, 20, 0.5)';
    canvasCtx.fillRect(0, 0, waveformCanvas.width, waveformCanvas.height);

    const barWidth = (waveformCanvas.width / bufferLength) * 1.5;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const barHeight = (dataArray[i] / 255) * waveformCanvas.height;
      canvasCtx.fillStyle = '#6366f1';
      canvasCtx.fillRect(x, waveformCanvas.height - barHeight, barWidth - 2, barHeight);
      x += barWidth;
    }
  }

  draw();
}

// 4. Text Input Logic
function initTextInput() {
  sendTextBtn.addEventListener('click', async () => {
    stopSpeech(); // Cancel any active TTS speech when submitting a new text query
    const query = textQueryInput.value.trim();
    if (!query) {
      alert('Please enter a valid text query.');
      return;
    }
    await submitTextQuery(query);
  });
}

// 5. Submit Functions
async function submitAudioQuery(audioBlob) {
  stopSpeech();
  resetUIState(true);
  stepperSection.classList.remove('hidden');

  const formData = new FormData();
  formData.append('audio', audioBlob, 'user_recording.wav');

  try {
    const res = await fetch(`${API_BASE_URL}/ask`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      throw new Error(`Server returned status HTTP ${res.status}`);
    }

    const data = await res.json();
    renderPipelineResults(data, true);
  } catch (err) {
    alert('Failed to process audio query: ' + err.message);
    console.error('Error submitting audio query:', err);
    pipelineStatusTag.className = 'status-tag short-circuited';
    pipelineStatusTag.textContent = 'Request Failed';
  } finally {
    recordStateText.textContent = 'Click the microphone to start speaking';
    recordTimer.textContent = '00:00';
  }
}

async function submitTextQuery(query) {
  stopSpeech();
  resetUIState(false);
  stepperSection.classList.remove('hidden');

  try {
    const res = await fetch(`${API_BASE_URL}/ask_text`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query }),
    });

    if (!res.ok) {
      throw new Error(`Server returned status HTTP ${res.status}`);
    }

    const data = await res.json();
    renderPipelineResults(data, false);
  } catch (err) {
    alert('Failed to process text query: ' + err.message);
    console.error('Error submitting text query:', err);
    pipelineStatusTag.className = 'status-tag short-circuited';
    pipelineStatusTag.textContent = 'Request Failed';
  }
}

// 6. Text-To-Speech (TTS) Logic
function initTTS() {
  if ('speechSynthesis' in window) {
    availableVoices = window.speechSynthesis.getVoices();
    window.speechSynthesis.onvoiceschanged = () => {
      availableVoices = window.speechSynthesis.getVoices();
    };
  }

  if (ttsToggleBtn) {
    ttsToggleBtn.addEventListener('click', () => {
      if (window.speechSynthesis && window.speechSynthesis.speaking) {
        stopSpeech();
      } else {
        isTtsEnabled = !isTtsEnabled;
        updateTtsButtonUI(false);
      }
    });
  }
}

function updateTtsButtonUI(isSpeaking) {
  if (!ttsToggleBtn) return;

  if (isSpeaking) {
    ttsToggleBtn.className = 'toggle-btn-small speaking';
    ttsToggleBtn.innerHTML = '<i class="fa-solid fa-volume-high"></i> <span>Speaking... (Click to Stop)</span>';
  } else if (isTtsEnabled) {
    ttsToggleBtn.className = 'toggle-btn-small';
    ttsToggleBtn.innerHTML = '<i class="fa-solid fa-volume-high"></i> <span>Read Aloud (On)</span>';
  } else {
    ttsToggleBtn.className = 'toggle-btn-small muted';
    ttsToggleBtn.innerHTML = '<i class="fa-solid fa-volume-xmark"></i> <span>Muted</span>';
  }
}

function stopSpeech() {
  if ('speechSynthesis' in window) {
    window.speechSynthesis.cancel();
  }
  updateTtsButtonUI(false);
}

function speakAnswerText(rawText) {
  if (!isTtsEnabled || !('speechSynthesis' in window) || !rawText) return;

  stopSpeech();

  // Edge Case 1: Strip non-spoken content (citation tags [1], [1, 2, 3], markdown formatting)
  const spokenText = rawText
    .replace(/\[[\d,\s]+\]/g, '') // Strip ALL citation tags like [1], [1, 2, 3], [2, 3]
    .replace(/\*+/g, '')        // Strip markdown bold/italic asterisks
    .replace(/#+/g, '')         // Strip markdown headers
    .replace(/`+/g, '')         // Strip code backticks
    .replace(/\s+/g, ' ')       // Normalize whitespace
    .trim();


  if (!spokenText) return;

  const utterance = new SpeechSynthesisUtterance(spokenText);

  // Detect language: check for Devanagari (Hindi) characters
  const isHindi = /[\u0900-\u097F]/.test(spokenText);
  const targetLang = isHindi ? 'hi' : 'en';

  if (availableVoices.length === 0) {
    availableVoices = window.speechSynthesis.getVoices();
  }

  const voice = availableVoices.find((v) => v.lang.startsWith(targetLang) || v.lang.startsWith(isHindi ? 'hi-IN' : 'en-US'));
  if (voice) {
    utterance.voice = voice;
  }
  utterance.lang = isHindi ? 'hi-IN' : 'en-US';
  utterance.rate = 1.0;
  utterance.pitch = 1.0;

  utterance.onstart = () => {
    updateTtsButtonUI(true);
  };

  utterance.onend = () => {
    updateTtsButtonUI(false);
  };

  utterance.onerror = (e) => {
    console.warn('Speech synthesis error:', e);
    updateTtsButtonUI(false);
  };

  window.speechSynthesis.speak(utterance);
}

// 7. UI Reset & Honest Stepper Playback
function resetUIState(isAudio) {
  stopSpeech();
  resultsContainer.classList.add('hidden');
  shortCircuitBanner.classList.add('hidden');
  flaggedClaimsWrapper.classList.add('hidden');
  transcriptPanel.classList.add('hidden');
  answerContent.innerHTML = '';
  citationsList.innerHTML = '';
  latencyBars.innerHTML = '';

  const stages = ['transcribe', 'classify', 'retrieve', 'check_grounding', 'generate', 'verify'];
  stages.forEach((stage) => {
    const stepEl = document.getElementById(`step-${stage}`);
    const timeEl = document.getElementById(`time-${stage}`);
    if (stepEl) {
      stepEl.className = 'step-item';
      if (!isAudio && stage === 'transcribe') {
        stepEl.style.opacity = '0.2';
      } else {
        stepEl.style.opacity = '0.4';
      }
    }
    if (timeEl) timeEl.textContent = '-- ms';
  });

  pipelineStatusTag.className = 'status-tag running';
  pipelineStatusTag.textContent = 'Processing Pipeline...';
}

function updateStepperState(stageName, state, ms = null) {
  const stepEl = document.getElementById(`step-${stageName}`);
  const timeEl = document.getElementById(`time-${stageName}`);
  if (stepEl) {
    stepEl.className = `step-item ${state}`;
  }
  if (timeEl && ms !== null) {
    timeEl.textContent = `${ms.toFixed(1)} ms`;
  }
}

function renderPipelineResults(data, isAudio) {
  const stoppedAt = data.stopped_at || 'verify';
  const timings = data.latency_ms?.stages || {};
  const totalMs = data.latency_ms?.total || 0;

  // Option B: Honest Playback of Actual Backend Stage Timings
  const stagesOrdered = ['transcribe', 'classify', 'retrieve', 'check_grounding', 'generate', 'verify'];
  let passedStop = false;

  stagesOrdered.forEach((stg) => {
    if (!isAudio && stg === 'transcribe') return;

    if (timings[stg] !== undefined) {
      updateStepperState(stg, stg === stoppedAt && stoppedAt !== 'verify' ? 'stopped' : 'completed', timings[stg]);
    } else if (passedStop) {
      updateStepperState(stg, 'disabled');
    }
    if (stg === stoppedAt) passedStop = true;
  });

  if (stoppedAt === 'verify') {
    pipelineStatusTag.className = 'status-tag completed';
    pipelineStatusTag.textContent = 'Completed (6 Stages)';
  } else {
    pipelineStatusTag.className = 'status-tag short-circuited';
    pipelineStatusTag.textContent = `Stopped early at ${stoppedAt}`;
  }

  // Show Results Panel
  resultsContainer.classList.remove('hidden');

  // Live Transcript Display
  if (data.transcript && data.transcript.trim()) {
    transcriptPanel.classList.remove('hidden');
    transcriptText.textContent = `"${data.transcript}"`;
  }

  // Grounding & Short-Circuit Handling
  let answerToSpeak = data.answer || '';

  if (stoppedAt === 'check_grounding') {
    shortCircuitBanner.classList.remove('hidden');
    groundingBadge.className = 'status-tag short-circuited';
    groundingBadge.innerHTML = '<i class="fa-solid fa-ban"></i> Insufficient Grounding (<0.83)';
  } else {
    groundingBadge.className = 'badge-success';
    groundingBadge.innerHTML = '<i class="fa-solid fa-shield-check"></i> Grounded Answer';
  }

  answerContent.innerHTML = formatAnswerText(answerToSpeak);

  // Trigger TTS Read Aloud
  speakAnswerText(answerToSpeak);

  // Flagged Claims (Safety Audit)
  if (data.flagged_claims && data.flagged_claims.length > 0) {
    flaggedClaimsWrapper.classList.remove('hidden');
    flaggedClaimsList.innerHTML = data.flagged_claims
      .map((fc) => `<li><strong>[Claim #${fc.citation_index || '1'}]</strong> ${fc.reason || fc}</li>`)
      .join('');
  }

  // Latency Breakdown Bars
  totalLatencyBadge.innerHTML = `Total: <strong>${totalMs.toFixed(1)} ms</strong>`;
  latencyBars.innerHTML = Object.entries(timings)
    .map(([stg, ms]) => {
      const pct = Math.min(100, Math.max(5, (ms / totalMs) * 100));
      return `
      <div class="latency-row">
        <div class="latency-row-header">
          <span>${stg}</span>
          <strong>${ms.toFixed(1)} ms</strong>
        </div>
        <div class="latency-bar-track">
          <div class="latency-bar-fill" style="width: ${pct}%"></div>
        </div>
      </div>
    `;
    })
    .join('');

  // Citations & Sources List
  const citations = data.citations || [];
  sourcesCount.textContent = `${citations.length} sources`;
  if (citations.length > 0) {
    citationsList.innerHTML = citations
      .map(
        (c, idx) => `
      <div class="citation-card">
        <div class="citation-header">
          <span>SOURCE [${idx + 1}]</span>
        </div>
        <div>${c}</div>
      </div>
    `
      )
      .join('');
  } else {
    citationsList.innerHTML = '<div style="color: var(--text-dim); font-size: 0.85rem;">No citations available for this query.</div>';
  }
}

function formatAnswerText(text) {
  return text.replace(/\[([\d,\s]+)\]/g, '<span style="color: var(--accent-purple); font-weight: bold;">[$1]</span>');
}

