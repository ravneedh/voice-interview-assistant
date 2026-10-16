# ==========================================
# 1. DEPENDENCIES & OLLAMA INSTALLATION
# ==========================================
print("1/5: Cleaning up processes & installing dependencies...")
!fuser -k 8000/tcp > /dev/null 2>&1 || true

!apt-get update -qq && apt-get install -y -qq ffmpeg zstd curl > /dev/null 2>&1

!pip install -q "click==8.1.8"
!pip install -q openai-whisper soundfile gTTS pyngrok fastapi uvicorn requests torch

!curl -fsSL https://ollama.com/install.sh | sh > /dev/null 2>&1

import os
import time
import shutil
import tempfile
import asyncio
import requests
import whisper
import torch
import random
from gtts import gTTS
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn
from threading import Thread

print("2/5: Starting Ollama background server...")
!nohup ollama serve > /dev/null 2>&1 &
time.sleep(5)

print("3/5: Pulling Llama 3.2 model...")
!ollama pull llama3.2:3b

print("4/5: Loading Whisper STT model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Whisper on target device: {device}")
whisper_model = whisper.load_model("tiny", device=device)

# ==========================================
# 2. BEHAVIORAL STAR & FLUENCY BACKEND
# ==========================================
print("5/5: Setting up Behavioral STAR & Fluency FastAPI backend...")
app = FastAPI(title="Behavioral  STAR & Fluency Interviewer")

SYSTEM_PROMPT = """
You are a Senior  Interviewer conducting a Behavioral Mock Interview.

Evaluate the candidate's turn using this mandatory structure:

1. DICTION AUDIT: Call out specific filler words ('um', 'like', 'you know', 'basically', 'so').
2. STAR EVALUATION: State which letter of STAR is missing or weak.
3. FOLLOW-UP QUESTION: Ask ONE sharp follow-up question probing deeper into their actions or metrics.

Keep your response concise (under 80 words).
"""

BEHAVIORAL_QUESTIONS = [
    "Tell me about a time when you had to manage a difficult stakeholder or resolve severe team misalignment on a project.",
    "Describe a situation where a project you led failed or missed its core deadline. What went wrong and what did you learn?",
    "Tell me about a time you had to make an important product decision with incomplete or ambiguous data.",
    "Give an example of a time you advocated for the user when engineering or business leadership pushed back.",
    "Tell me about a time you had to prioritize between two critical features with limited engineering bandwidth."
]

conversation_history = []
current_turn_count = 0
latest_candidate_audio_path = ""

def analyze_speech_fluency(whisper_result):
    """Calculates WPM and builds transcript with clickable pause timestamps."""
    segments = whisper_result.get("segments", [])
    words = []
    for seg in segments:
        words.extend(seg.get("words", []))

    if not words:
        clean_text = whisper_result.get("text", "").strip()
        return {"wpm": 0, "pauses": 0, "annotated_transcript": clean_text}

    total_duration = words[-1]["end"] - words[0]["start"]
    word_count = len(words)
    wpm = round((word_count / total_duration) * 60) if total_duration > 0 else 0

    long_pauses = 0
    annotated_words = [words[0]["word"]]

    for i in range(1, len(words)):
        pause_duration = words[i]["start"] - words[i-1]["end"]
        if pause_duration >= 1.5:
            long_pauses += 1
            pause_time_sec = words[i-1]["end"]
            mins = int(pause_time_sec // 60)
            secs = int(pause_time_sec % 60)
            timestamp_str = f"{mins}:{secs:02d}"
            pause_btn = f' <button class="pause-tag" onclick="playPauseAudio(this, {pause_time_sec})"> Pause {pause_duration:.1f}s @ {timestamp_str}</button> '
            annotated_words.append(pause_btn)

        annotated_words.append(words[i]["word"])

    annotated_transcript = "".join(annotated_words)
    return {"wpm": wpm, "pauses": long_pauses, "annotated_transcript": annotated_transcript}

@app.get("/api/start-interview")
async def start_interview():
    global conversation_history, current_turn_count
    current_turn_count = 0
    initial_q = random.choice(BEHAVIORAL_QUESTIONS)

    conversation_history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": initial_q}
    ]

    tts = gTTS(text=initial_q, lang='en')
    output_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    tts.save(output_audio_path)
    return {"interviewer_text": initial_q, "audio_url": f"/audio?path={output_audio_path}"}

@app.post("/api/interview-turn")
async def process_candidate_turn(file: UploadFile = File(...)):
    global current_turn_count, latest_candidate_audio_path
    current_turn_count += 1

    candidate_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
    shutil.copyfileobj(file.file, candidate_audio_file)
    latest_candidate_audio_path = candidate_audio_file.name

    result = whisper_model.transcribe(
        latest_candidate_audio_path,
        initial_prompt="Um, uh, like, so, you know, I mean, right, basically...",
        fp16=torch.cuda.is_available(),
        word_timestamps=True
    )
    raw_transcript = result["text"].strip()

    fluency_stats = analyze_speech_fluency(result)

    conversation_history.append({"role": "user", "content": raw_transcript})

    ollama_res = requests.post("http://localhost:11434/api/chat", json={
        "model": "llama3.2:3b",
        "messages": conversation_history,
        "stream": False,
        "options": {
            "num_predict": 180,
            "temperature": 0.3
        }
    })
    interviewer_reply = ollama_res.json()["message"]["content"]
    conversation_history.append({"role": "assistant", "content": interviewer_reply})

    fluency_summary = f"[Fluency Analysis: {fluency_stats['wpm']} WPM | {fluency_stats['pauses']} awkward pause(s) detected (>1.5s)]\n\n"
    combined_response = fluency_summary + interviewer_reply

    tts = gTTS(text=interviewer_reply, lang='en')
    output_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    tts.save(output_audio_path)

    return {
        "candidate_transcript_html": fluency_stats["annotated_transcript"],
        "candidate_audio_url": f"/audio?path={latest_candidate_audio_path}",
        "interviewer_response": combined_response,
        "interviewer_audio_url": f"/audio?path={output_audio_path}"
    }

@app.post("/api/generate-model-answer")
async def generate_model_answer():
    user_transcripts = [msg["content"] for msg in conversation_history if msg["role"] == "user"]
    combined_user_text = " ".join(user_transcripts)

    prompt = f"""
    The candidate provided this raw, unpolished behavioral interview response across their turns:
    "{combined_user_text}"

    Please rewrite this into an exemplary, executive-ready  STAR interview answer.

    Structure your output as follows:
    Situation and Task: 2 crisp sentences framing the core problem and constraints.
    Action: 3 clear, action-oriented bullet points emphasizing leadership and decision-making.
    Result: 2 bullet points with realistic quantitative metrics.

    Keep it concise, impactful, and clear of any filler words.
    """

    res = requests.post("http://localhost:11434/api/chat", json={
        "model": "llama3.2:3b",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_predict": 250, "temperature": 0.4}
    })
    model_answer = res.json()["message"]["content"]

    tts = gTTS(text=model_answer, lang='en')
    output_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    tts.save(output_audio_path)

    return {
        "model_answer": model_answer,
        "audio_url": f"/audio?path={output_audio_path}"
    }

@app.get("/audio")
async def get_audio(path: str):
    return FileResponse(path, media_type="audio/mp3")

@app.get("/", response_class=HTMLResponse)
async def get_interface():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Behavioral STAR + Interactive Pause Replay</title>
        <style>
            body { font-family: system-ui, -apple-system, sans-serif; max-width: 750px; margin: 30px auto; padding: 20px; background: #f8fafc; }
            .card { background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.06); }
            .chat-box { max-height: 420px; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; margin: 20px 0; padding: 10px; border: 1px solid #e2e8f0; border-radius: 8px; background: #fafafa; }
            .msg { padding: 12px 16px; border-radius: 8px; max-width: 85%; line-height: 1.5; font-size: 15px; white-space: pre-wrap; }
            .interviewer { background: #e0f2fe; color: #0369a1; align-self: flex-start; }
            .candidate { background: #2563eb; color: white; align-self: flex-end; }
            .pause-tag { background: #ef4444; color: white; border: none; padding: 2px 6px; border-radius: 4px; font-size: 12px; font-weight: bold; cursor: pointer; margin: 0 3px; }
            .pause-tag:hover { background: #dc2626; }
            .model-answer-card { background: #f0fdf4; border: 1px solid #bbf7d0; color: #166534; padding: 15px; border-radius: 8px; margin-top: 15px; white-space: pre-wrap; }
            button { background: #2563eb; color: white; border: none; padding: 12px 18px; border-radius: 8px; font-weight: 600; cursor: pointer; margin-right: 8px; }
            button.recording { background: #dc2626; }
            button.model-btn { background: #16a34a; }
            audio { width: 100%; margin-top: 10px; }
        </style>
    </head>
    <body>
        <div class="card">
            <h2>Behavioral STAR + Pause Replay Coach</h2>
            <div class="chat-box" id="chatBox"></div>
            <audio id="audioPlayer" controls></audio>
            <div style="margin-top: 15px;">
                <button id="startBtn">Begin Behavioral Interview</button>
                <button id="recordBtn" style="display:none;">Start Answer</button>
                <button id="modelBtn" class="model-btn" style="display:none;">Draft Better Response (Model Answer)</button>
            </div>
            <div id="statusText" style="color: #64748b; font-size: 14px; margin-top: 10px;"></div>
            <div id="modelAnswerBox"></div>
        </div>
        <script>
            let mediaRecorder, audioChunks = [];
            const chatBox = document.getElementById('chatBox');
            const startBtn = document.getElementById('startBtn');
            const recordBtn = document.getElementById('recordBtn');
            const modelBtn = document.getElementById('modelBtn');
            const statusText = document.getElementById('statusText');
            const audioPlayer = document.getElementById('audioPlayer');
            const modelAnswerBox = document.getElementById('modelAnswerBox');

            function addMessage(role, htmlOrText, isHtml = false, audioUrl = "") {
                const div = document.createElement('div');
                div.className = `msg ${role}`;
                if (audioUrl) div.dataset.audioUrl = audioUrl;
                if (isHtml) {
                    div.innerHTML = htmlOrText;
                } else {
                    div.innerText = htmlOrText;
                }
                chatBox.appendChild(div);
                chatBox.scrollTop = chatBox.scrollHeight;
            }

            function playPauseAudio(button, startTime) {
                const message = button.closest('.candidate');
                const audioUrl = message ? message.dataset.audioUrl : "";
                if (!audioUrl) return;

                audioPlayer.pause();
                audioPlayer.onloadedmetadata = () => {
                    audioPlayer.onloadedmetadata = null;
                    audioPlayer.currentTime = Math.max(0, startTime - 1.0);
                    audioPlayer.play().catch(() => {
                        statusText.innerText = "Press Play to listen to the selected recording.";
                    });
                };
                audioPlayer.src = audioUrl;
                audioPlayer.load();
                statusText.innerText = `Replaying audio prior to pause at ${startTime.toFixed(1)}s...`;
            }

            startBtn.onclick = async () => {
                startBtn.style.display = "none";
                modelAnswerBox.innerHTML = "";
                statusText.innerText = "Starting interview...";
                const res = await fetch('/api/start-interview');
                const data = await res.json();
                addMessage('interviewer', data.interviewer_text);
                audioPlayer.onloadedmetadata = null;
                audioPlayer.src = data.audio_url;
                audioPlayer.play();
                recordBtn.style.display = "inline-block";
                modelBtn.style.display = "inline-block";
                statusText.innerText = "Ready. Click 'Start Answer' to respond.";
            };

            recordBtn.onclick = async () => {
                if (mediaRecorder && mediaRecorder.state === "recording") {
                    mediaRecorder.stop();
                    recordBtn.innerText = "Start Answer";
                    recordBtn.classList.remove('recording');
                    recordBtn.disabled = true;
                    statusText.innerText = "Analyzing fluency, STAR structure & pause locations...";
                } else {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];
                    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
                    mediaRecorder.onstop = sendTurn;
                    mediaRecorder.start();
                    recordBtn.innerText = "Stop & Send Answer";
                    recordBtn.classList.add('recording');
                    statusText.innerText = "Recording response...";
                }
            };

            async function sendTurn() {
                const blob = new Blob(audioChunks, { type: 'audio/webm' });
                const formData = new FormData();
                formData.append('file', blob, 'turn.webm');
                const res = await fetch('/api/interview-turn', { method: 'POST', body: formData });
                const data = await res.json();

                addMessage('candidate', data.candidate_transcript_html, true, data.candidate_audio_url);
                addMessage('interviewer', data.interviewer_response);

                audioPlayer.onloadedmetadata = null;
                audioPlayer.src = data.interviewer_audio_url;
                audioPlayer.play();

                recordBtn.disabled = false;
                statusText.innerText = "Click red pause tags in your message to replay hesitation spots.";
            }

            modelBtn.onclick = async () => {
                statusText.innerText = "Generating optimized STAR model answer and voice playback...";
                modelBtn.disabled = true;
                const res = await fetch('/api/generate-model-answer', { method: 'POST' });
                const data = await res.json();
                modelAnswerBox.innerHTML = `<div class="model-answer-card"><strong>💡 MODEL ANSWER (How you could have answered):</strong><br><br>${data.model_answer}</div>`;
                audioPlayer.onloadedmetadata = null;
                audioPlayer.src = data.audio_url;
                audioPlayer.play();
                statusText.innerText = "Playing Model Answer audio!";
                modelBtn.disabled = false;
            };
        </script>
    </body>
    </html>
    """

# ==========================================
# 3. THREADED SERVER START
# ==========================================
def start_uvicorn():
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="error")
    server = uvicorn.Server(config)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(server.serve())

Thread(target=start_uvicorn, daemon=True).start()
time.sleep(3)

from google.colab.output import eval_js
public_url = eval_js("google.colab.kernel.proxyPort(8000)")
print(f"\nBEHAVIORAL INTERVIEW COACH READY! Click the link:\n {public_url}\n")
