"""Colab voice interview benchmark,  (short + long answers).

COLAB INSTRUCTIONS
1. Select CPU/None in Runtime > Change runtime type.
2. Run the mock_interview.py app cell and wait for setup to finish.
3. Paste THIS ENTIRE FILE into the next code cell. Leave CPU settings below.
4. Run the cell. Download the printed ZIP before switching runtimes.
5. Switch to T4 GPU, rerun the app setup cell, and upload the NEW v3 CPU ZIP.
6. In this benchmark cell set COLAB_BACKEND = "t4" and
   COLAB_CPU_RESULTS_ZIP = "/content/<actual-new-v3-CPU-archive-name>.zip".
7. Run and download the T4 ZIP. Send both ZIPs for analysis.

Each runtime: 1 startup observation + 4 warm-ups (2 per workload) + 40 measured
requests (20 short, 20 long) = 45 total. Every request includes reply gTTS.
After warm-up, each of four blocks contains all ten inputs in seeded shuffled
order, so short and long requests are interleaved. Summaries separate workloads.
Long answers have five sentences and 80-120 whitespace-delimited words.

Do not interact with the app or send other Ollama requests during the test.
Keep the runtime connected. No script upload or %run is needed for pasted cells.

WHAT IS MEASURED
pipeline_ms: elapsed STT -> prompt assembly -> complete LLM feedback processing.
tts_ms: gTTS object creation through return from saving reply MP3; includes
external service/network time and local preparation/file writing.
feedback_audio_ready_ms: one continuous timer from STT start to reply MP3 saved.
Browser recording, upload, audio delivery and playback are excluded.
Input-audio synthesis is outside all request timers. All input/output MP3s saved.
The exact LLM response is sent to gTTS (no manual shortening or Markdown removal).
No reply-audio caching. Output lengths may vary; their counts are recorded.
STT/LLM work excludes external TTS from its timer

OUTPUT
attempts.csv / attempts.jsonl: all requests, workload labels, timings and text.
summary.json: per-phase and per-workload summaries; primary view is
  measured_by_workload.short / measured_by_workload.long.
summary_by_workload.csv: ready-to-read mean, sample SD, median, P95 per workload.
comparison.json (T4): ratios against the supplied CPU archive, by workload.
corpus/: ten input MP3s and manifest with text, word count, duration, checksums.
reply_audio/: generated reply MP3s. config/environment/schedule/status/source saved.

Report five distinct synthetic inputs per workload with four repetitions each.
Long inputs add semantic detail as well as duration: this is an application-
workload comparison, not a controlled experiment isolating word count alone.
TTS differences across runtimes include external-service variability and output
length, and should not be attributed to GPU acceleration of gTTS.
First request is a startup observation, not a controlled cold-start experiment.
RAM/VRAM are snapshots, not complete pipeline peak-memory measurements.

References: https://docs.ollama.com/api/chat
https://docs.ollama.com/api/ps
https://gtts.readthedocs.io/en/latest/module.html
"""

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import signal
import statistics
import subprocess
import sys
import time
import uuid
import zipfile
from datetime import datetime, timezone

COLAB_BACKEND = "cpu"
COLAB_CPU_RESULTS_ZIP = None
COLAB_DOWNLOAD_RESULTS = True

PROTOCOL = "3.0"
STT_DEADLINE_S = 300
LLM_DEADLINE_S = 600
TTS_DEADLINE_S = 180
MODEL = "llama3.2:3b"
BASE_URL = "http://127.0.0.1:11434"
SEED = 20260919
FILLER_PROMPT = "Um, uh, like, so, you know, I mean, right, basically..."
COACH_PROMPT = """
You are a Senior PM Interviewer conducting a Behavioral Mock Interview.

Evaluate the candidate's turn using this mandatory structure:

1. DICTION AUDIT: Call out specific filler words ('um', 'like', 'you know', 'basically', 'so').
2. STAR EVALUATION: State which letter of STAR is missing or weak.
3. FOLLOW-UP QUESTION: Ask ONE sharp follow-up question probing deeper into their actions or metrics.

Keep your response concise (under 80 words).
"""
PROMPTS = [
    "I led a cross-functional team to reduce storage latency by 35 percent.",
    "Um, so... like, we had a major stakeholder misalignment on product scope.",
    "Basically, I prioritized feature A over feature B due to bandwidth limits.",
    "When engineering pushed back, I brought user telemetry data to prove impact.",
    "We missed our quarter deadline because third-party APIs failed unexpectedly.",
]


LONG_PROMPTS = [
    "I led a cross-functional team to reduce storage latency by 35 percent when our analytics product began missing response targets for enterprise customers. "
    "My responsibility was to identify the most valuable improvement while keeping the release within our existing engineering capacity and delivery schedule. "
    "I reviewed customer complaints and request traces with engineering, then prioritized a caching change over a larger redesign because it addressed the most frequent delays. "
    "We introduced the change gradually, compared response times against the previous configuration, and checked that storage costs and error rates remained acceptable. "
    "The release reduced latency by 35 percent, and I documented the tradeoffs and monitoring plan for the next product review.",
    "Um, so, like, we had a major stakeholder misalignment on product scope while preparing a new reporting feature for a customer pilot. "
    "Sales wanted several custom dashboards, engineering wanted to simplify the release, and I was responsible for getting agreement on a realistic starting point. "
    "I organized a discussion around the customer problems, wrote down the competing requests, and asked each team to explain what would happen if its request waited. "
    "Basically, we agreed to deliver a smaller workflow first and revisit the remaining requests after collecting feedback from the pilot users. "
    "The teams seemed more aligned afterward, although I did not collect a clear measure of whether the decision improved customer adoption.",
    "Basically, I prioritized feature A over feature B due to bandwidth limits when our team had capacity for only one major improvement that quarter. "
    "Feature A simplified account setup for new customers, while feature B offered advanced reporting that a smaller group of existing customers had requested. "
    "I compared support tickets, onboarding dropoffs, customer interviews, and engineering estimates, then proposed choosing feature A because the evidence suggested a broader immediate benefit. "
    "To address concerns from the reporting customers, I explained the sequencing decision and arranged a smaller discovery effort for feature B. "
    "After launch, setup completion increased from 62 to 74 percent, and we used the remaining reporting feedback to refine the next roadmap discussion.",
    "When engineering pushed back, I brought user telemetry data to prove impact during planning for an improvement to our mobile checkout experience. "
    "The proposed change required work across several services, and the engineers were concerned that it would delay reliability tasks already committed for the release. "
    "I reviewed the abandonment data with the team and compared it with customer interviews to show where users were repeatedly getting stuck. "
    "We then discussed a smaller implementation, adjusted the delivery plan, and agreed to check both conversion and reliability during a limited rollout. "
    "The rollout improved checkout completion, but our team shared the analysis and execution work, and I would need to explain my individual contribution more clearly.",
    "We missed our quarter deadline because third-party APIs failed unexpectedly during the final integration tests for a partner reporting feature. "
    "As the product manager, I had promised a launch date before the team had completed a realistic review of the external dependency and its failure modes. "
    "Once the failures appeared, I worked with engineering to identify a reduced release, informed the affected customers, and changed the launch plan instead of hiding the delay. "
    "We introduced a fallback for unavailable data and added dependency checks to the next planning cycle so that similar problems would surface earlier. "
    "The feature shipped two weeks later, and I learned to validate critical dependencies before committing to a delivery date."
]
WORKLOADS = ("short", "long")
PROMPT_SPECS = [
    {"prompt_id": f"{workload}_{i:02d}", "scenario_id": f"p{i:02d}",
     "workload": workload, "source_text": text, "source_word_count": len(text.split())}
    for workload, texts in [("short", PROMPTS), ("long", LONG_PROMPTS)]
    for i, text in enumerate(texts, 1)
]


def build_plan(items):
    groups = {w: [i for i in items if i["workload"] == w] for w in WORKLOADS}
    if any(len(v) != 5 for v in groups.values()):
        raise ValueError("Expected five inputs in each workload")
    plan = [("startup_observation", 0, groups["short"][0], LLM_DEADLINE_S)]
    for w in WORKLOADS:
        plan.extend(("warmup", 0, i, LLM_DEADLINE_S) for i in groups[w][:2])
    rng = random.Random(SEED)
    for block in range(1, 5):
        shuffled = items.copy()
        rng.shuffle(shuffled)
        plan.extend(("measured", block, i, LLM_DEADLINE_S) for i in shuffled)
    return plan


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))


def command(args):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=15)
        return {"returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
    except Exception as exc:
        return {"error": str(exc)}


def api(route, payload=None):
    import requests
    if payload is None:
        response = requests.get(BASE_URL + route, timeout=(5, 30))
    else:
        response = requests.post(BASE_URL + route, json=payload, timeout=(5, 30))
    response.raise_for_status()
    result = response.json()
    if isinstance(result, dict) and result.get("error"):
        raise RuntimeError(result["error"])
    return result


class DeadlineExceeded(BaseException):
    """BaseException prevents HTTP libraries from hiding the wall-clock deadline."""


@contextlib.contextmanager
def deadline(seconds):
    # Google Colab runs this script in the main thread on Linux via %run -i.
    if not hasattr(signal, "setitimer"):
        raise RuntimeError("This benchmark requires Linux/Colab signal timers.")
    if signal.getitimer(signal.ITIMER_REAL)[0] > 0:
        raise RuntimeError("An existing process alarm is active; use a fresh runtime.")
    previous = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        raise DeadlineExceeded(f"Stage wall-clock deadline exceeded ({seconds}s)")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def stream_chat(messages, options, seconds, row):
    import requests
    payload = {"model": MODEL, "messages": messages, "stream": True,
               "options": options, "keep_alive": "30m"}
    row["messages"] = messages
    row["llm_request_sha256"] = sha(json.dumps(payload, sort_keys=True).encode())
    row["response_text"] = ""
    row["client_first_content_ms"] = None
    started = time.perf_counter()
    completed = False
    try:
        with deadline(seconds):
            with requests.post(BASE_URL + "/api/chat", json=payload, stream=True,
                               timeout=(10, seconds)) as response:
                response.raise_for_status()
                # chunk_size=1 avoids the default 512-byte buffering for TTFT.
                # All configurations use the same client; server throughput is
                # separately measured from Ollama's final response statistics.
                for line in response.iter_lines(chunk_size=1):
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if chunk.get("error"):
                        raise RuntimeError(chunk["error"])
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        if row["client_first_content_ms"] is None:
                            row["client_first_content_ms"] = (time.perf_counter() - started) * 1000
                        row["response_text"] += content
                    if chunk.get("done"):
                        row["ollama_final"] = chunk
                        completed = True
                        break
        if not completed:
            raise RuntimeError("Stream ended without a final done message")
        final = row["ollama_final"]
        count, ns = final.get("eval_count"), final.get("eval_duration")
        if not isinstance(count, int) or count <= 0 or not isinstance(ns, (int, float)) or ns <= 0:
            raise RuntimeError("Missing/invalid Ollama token count or generation duration")
        if not row["response_text"].strip():
            raise RuntimeError("Ollama returned no nonempty response text")
        row["output_tokens"] = count
        row["ollama_tokens_per_s"] = count / (ns / 1e9)
        for key in ["total_duration", "load_duration", "prompt_eval_duration", "eval_duration"]:
            row["ollama_" + key + "_ms"] = final.get(key, 0) / 1e6 if key in final else None
        row["prompt_tokens"] = final.get("prompt_eval_count")
        row["done_reason"] = final.get("done_reason")
    finally:
        row["llm_client_ms"] = (time.perf_counter() - started) * 1000


def placement(backend):
    state = api("/api/ps")
    matches = [m for m in state.get("models", [])
               if MODEL in (m.get("name"), m.get("model"))]
    if len(matches) != 1:
        raise RuntimeError("Cannot identify the requested model in Ollama /api/ps")
    model = matches[0]
    vram = model.get("size_vram")
    if not isinstance(vram, (int, float)):
        raise RuntimeError("Ollama did not report size_vram; cannot verify placement")
    if backend == "cpu" and vram != 0:
        raise RuntimeError("CPU condition invalid: Ollama reports GPU allocation")
    if backend == "t4" and vram <= 0:
        raise RuntimeError("T4 condition invalid: Ollama reports no GPU allocation")
    # size_vram > 0 establishes some GPU residency, not full layer offload.
    # Keep the original ps output for full/partial placement review.
    return {"api_ps": state, "model": model,
            "classification": "cpu" if vram == 0 else "gpu_resident_fullness_not_inferred"}


def telemetry():
    import psutil
    return {"utc": utc(), "system_ram_used_bytes": psutil.virtual_memory().used,
            "notebook_rss_bytes": psutil.Process().memory_info().rss,
            "nvidia_smi": command(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                                   "--format=csv,noheader"])}


def restore_corpus(archive, corpus):
    allowed = ["manifest.json"] + [i["prompt_id"] + ".mp3" for i in PROMPT_SPECS]
    with zipfile.ZipFile(archive) as z:
        for name in allowed:
            member = "corpus/" + name
            if z.getinfo(member).file_size > 10_000_000:
                raise ValueError("Unexpectedly large corpus member")
            (corpus / name).write_bytes(z.read(member))


def prepare_corpus(corpus, reuse):
    import whisper
    from gtts import gTTS
    corpus.mkdir()
    if reuse:
        restore_corpus(reuse, corpus)
        manifest = json.loads((corpus / "manifest.json").read_text())
        if manifest.get("protocol") != PROTOCOL or len(manifest.get("items", [])) != 10:
            raise ValueError("Use a NEW protocol 3.0 CPU ZIP with ten input recordings")
    else:
        print("Preparing ten synthetic recordings outside measurement timers...", flush=True)
        items = []
        for spec in PROMPT_SPECS:
            name = spec["prompt_id"] + ".mp3"
            with deadline(TTS_DEADLINE_S):
                gTTS(text=spec["source_text"], lang="en", tld="com", slow=False,
                     timeout=(10, 60)).save(str(corpus / name))
            items.append({**spec, "file": name, "sha256": sha((corpus / name).read_bytes())})
        manifest = {"protocol": PROTOCOL, "created_utc": utc(),
                    "source": "gTTS synthetic speech, en; not human recordings", "items": items}
    for item, spec in zip(manifest["items"], PROMPT_SPECS):
        if any(item.get(k) != v for k, v in spec.items()) or item["file"] != spec["prompt_id"] + ".mp3":
            raise ValueError("Corpus inputs differ from protocol")
        audio = corpus / item["file"]
        if sha(audio.read_bytes()) != item["sha256"]:
            raise ValueError("Audio checksum mismatch")
        samples = whisper.load_audio(str(audio))
        duration = len(samples) / whisper.audio.SAMPLE_RATE
        if duration <= 0:
            raise ValueError("Empty audio")
        item["decoded_duration_s"] = duration
        item["sample_rate"] = whisper.audio.SAMPLE_RATE
        print(f"  {item['prompt_id']}: {item['source_word_count']} words, {duration:.2f}s", flush=True)
    manifest["corpus_id"] = sha("".join(i["sha256"] for i in manifest["items"]).encode())
    write_json(corpus / "manifest.json", manifest)
    return manifest


def synthesize_reply(text, path):
    from gtts import gTTS
    with deadline(TTS_DEADLINE_S):
        gTTS(text=text, lang="en", tld="com", slow=False, timeout=(10, 60)).save(str(path))


def trial(model, item, corpus, phase, index, block, options, backend, seconds):
    import torch
    row = {"phase": phase, "index": index, "block": block, "prompt_id": item["prompt_id"],
           "workload": item["workload"], "scenario_id": item["scenario_id"],
           "source_word_count": item["source_word_count"],
           "audio_sha256": item["sha256"], "audio_duration_s": item["decoded_duration_s"],
           "source_text": item["source_text"], "started_utc": utc(), "status": "started",
           "telemetry_before": telemetry(), "llm_deadline_s": seconds}
    stage = "stt"
    start = time.perf_counter()
    try:
        if backend == "t4":
            torch.cuda.synchronize()
        st = time.perf_counter()
        try:
            with deadline(STT_DEADLINE_S):
                result = model.transcribe(str(corpus / item["file"]), initial_prompt=FILLER_PROMPT,
                                          fp16=backend == "t4", word_timestamps=True)
                if backend == "t4":
                    torch.cuda.synchronize()
        finally:
            row["stt_ms"] = (time.perf_counter() - st) * 1000
        row["transcript"] = result.get("text", "").strip()
        row["whisper_language"] = result.get("language")
        row["whisper_segments"] = result.get("segments", [])
        if not row["transcript"]:
            raise RuntimeError("Empty Whisper transcript; no fallback substituted")
        row["stt_rtf"] = row["stt_ms"] / 1000 / item["decoded_duration_s"]
        stage = "prompt"
        pt = time.perf_counter()
        messages = [{"role": "system", "content": COACH_PROMPT},
                    {"role": "user", "content": row["transcript"]}]
        row["prompt_ms"] = (time.perf_counter() - pt) * 1000
        stage = "llm"
        stream_chat(messages, options, seconds, row)
        row["pipeline_ms"] = (time.perf_counter() - start) * 1000
        row["local_stage_sum_ms"] = row["stt_ms"] + row["llm_client_ms"]
        row["stt_llm_completed"] = True
        stage = "tts"
        reply = corpus.parent / "reply_audio" / f"{index:02d}_{phase}_{item['prompt_id']}.mp3"
        row["reply_audio_file"] = str(reply.relative_to(corpus.parent))
        row["response_word_count"] = len(row["response_text"].split())
        row["response_char_count"] = len(row["response_text"])
        ts = time.perf_counter()
        synthesize_reply(row["response_text"], reply)
        audio_ready = time.perf_counter()
        row["tts_ms"] = (audio_ready - ts) * 1000
        row["feedback_audio_ready_ms"] = (audio_ready - start) * 1000
        # Validation/checksumming happen after the audio-ready timer stops.
        stage = "audio_validation"
        reply_bytes = reply.read_bytes()
        if not reply_bytes:
            raise RuntimeError("gTTS saved an empty reply file")
        row["reply_audio_bytes"] = len(reply_bytes)
        row["reply_audio_sha256"] = sha(reply_bytes)
        stage = "placement_check"
        row["placement_after"] = placement(backend)
        row["status"] = "ok"
    except (Exception, DeadlineExceeded, KeyboardInterrupt) as exc:
        row["status"] = "failed"
        row["failed_stage"] = stage
        row["error_type"] = type(exc).__name__
        row["error"] = str(exc)
        if stage == "tts":
            row["tts_attempt_elapsed_ms"] = (time.perf_counter() - ts) * 1000
        row["elapsed_to_error_ms"] = (time.perf_counter() - start) * 1000
    row["finished_utc"] = utc()
    row["telemetry_after"] = telemetry()
    return row


def percentile(values, p):
    values = sorted(values)
    k = (len(values) - 1) * p / 100
    a, b = math.floor(k), math.ceil(k)
    return values[a] + (values[b] - values[a]) * (k - a)


def describe(values):
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "population_sd": statistics.pstdev(values),
            "p95_linear": percentile(values, 95), "min": min(values), "max": max(values)}


METRICS = ["stt_ms", "stt_rtf", "client_first_content_ms", "llm_client_ms",
           "local_stage_sum_ms", "pipeline_ms", "output_tokens", "ollama_tokens_per_s",
           "ollama_load_duration_ms", "ollama_prompt_eval_duration_ms", "prompt_ms",
           "tts_ms", "feedback_audio_ready_ms", "audio_duration_s", "source_word_count",
           "response_word_count", "response_char_count"]


def summarize(rows):
    def group(subset):
        good = [r for r in subset if r["status"] == "ok"]
        return {"attempts": len(subset), "successful": len(good),
                "failed": len(subset) - len(good),
                "metrics_success_only": {m: describe([r[m] for r in good if r.get(m) is not None])
                                         for m in METRICS}}
    measured = [r for r in rows if r["phase"] == "measured"]
    return {"protocol": PROTOCOL,
            "note": "Report measured_by_workload separately. Each workload has five inputs repeated four times. "
                    "Warm-up failures abort; no attempt is retried or silently removed. "
                    "P95 with 20 attempts is unstable. Backend GPU residency need not imply full offload.",
            "phases": {p: group([r for r in rows if r["phase"] == p])
                       for p in ["startup_observation", "warmup", "measured"]},
            "measured_by_workload": {w: group([r for r in measured if r["workload"] == w])
                                     for w in WORKLOADS},
            "measured_by_prompt": {i["prompt_id"]: group([r for r in measured if r["prompt_id"] == i["prompt_id"]])
                                   for i in PROMPT_SPECS}}



def package_results(out, rows, status):
    write_json(out / "summary.json", summarize(rows))
    write_json(out / "run_status.json", status)
    columns = ["phase", "index", "block", "workload", "scenario_id", "prompt_id", "status", "failed_stage", "error_type",
               "error", "started_utc", "finished_utc", "audio_sha256", "source_text", "transcript",
               "response_text", "prompt_tokens", "done_reason", "stt_llm_completed",
               "reply_audio_file", "reply_audio_bytes", "reply_audio_sha256",
               "tts_attempt_elapsed_ms", "elapsed_to_error_ms"] + METRICS
    with (out / "attempts.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with (out / "summary_by_workload.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["workload", "metric", "n", "mean", "sample_sd", "median", "p95_linear", "min", "max"])
        writer.writeheader()
        for workload, group in summarize(rows)["measured_by_workload"].items():
            for metric, stats in group["metrics_success_only"].items():
                writer.writerow({"workload": workload, "metric": metric,
                                 **{k: v for k, v in stats.items() if k != "population_sd"}})
    archive = out.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(out))
    print(f"\nRESULT ARCHIVE: {archive}", flush=True)
    print("Download this ZIP before switching Colab runtimes. Upload both ZIPs for analysis.", flush=True)
    return archive


def main(args):
    import psutil
    import torch
    import whisper
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
    out = Path(args.output_root).resolve() / f"pm_benchmark_v3_{args.backend}_{run_id}"
    out.mkdir(parents=True)
    rows = []
    status = {"state": "initializing", "started_utc": utc(), "protocol": PROTOCOL}
    model = globals().get("whisper_model")
    options = {"num_predict": 180, "temperature": 0.3, "seed": SEED, "num_ctx": 2048}
    if args.backend == "cpu":
        options["num_gpu"] = 0
    config = {"protocol": PROTOCOL, "backend_requested": args.backend, "model": MODEL,
              "options": options, "system_prompt": COACH_PROMPT, "initial_prompt": FILLER_PROMPT,
              "warmups_required": 4, "warmups_per_workload": 2, "measured_attempts_planned": 40,
              "measured_per_workload": 20, "stt_deadline_s": STT_DEADLINE_S,
              "llm_deadline_s": LLM_DEADLINE_S, "tts_deadline_s": TTS_DEADLINE_S,
              "sampling_order_seed": SEED, "input_definitions": PROMPT_SPECS,
              "gtts": {"lang": "en", "tld": "com", "slow": False, "timeout": [10, 60],
                       "text_policy": "exact generated response; no Markdown removal", "reply_cache": False},
              "reply_tts": True, "llm_streaming": True, "fresh_chat_history_each_request": True,
              "startup_is_not_controlled_cold_start": True,
              "prompt_cache_policy": "Ollama default; no cache flush; repeated short prompts may reuse prefixes",
              "changes_from_v2": "Two input workloads; 40 measured requests; reply gTTS and audio-ready timing; "
                  "per-workload summaries; output MP3s saved; per-stage deadlines. "
                  "New protocol; do not pool with v2 requests."}
    write_json(out / "config.json", config)
    try:
        (out / "reply_audio").mkdir()
        if args.reuse_results in ("None", "none", ""):
            raise ValueError('For CPU use COLAB_CPU_RESULTS_ZIP = None without quotes; for T4 supply the actual v3 CPU ZIP path.')
        # Pasted cells have no __file__; preserve their source from IPython history.
        source = Path(main.__code__.co_filename)
        if source.is_file():
            (out / "benchmarking_script.py").write_bytes(source.read_bytes())
        else:
            try:
                from IPython import get_ipython
                cell_source = get_ipython().history_manager.input_hist_raw[-1]
                if "def stream_chat" not in cell_source or "def main" not in cell_source:
                    raise ValueError("Current cell source does not contain the benchmark")
                (out / "benchmarking_script.py").write_text(cell_source)
            except Exception as exc:
                write_json(out / "source_capture_note.json", {"source_not_captured": str(exc)})
        if args.backend == "cpu" and torch.cuda.is_available():
            raise RuntimeError("Select a Colab CPU runtime for this condition, then rerun app setup.")
        if args.backend == "t4":
            if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
                raise RuntimeError("Select a Colab T4 GPU runtime, then rerun app setup.")
            if not args.reuse_results:
                raise RuntimeError("T4 comparison requires --reuse-results pointing to the CPU ZIP.")
        packages = {}
        for name in ["torch", "openai-whisper", "requests", "numpy", "gTTS", "psutil"]:
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        env = {"utc": utc(), "platform": platform.platform(), "python": sys.version,
               "lscpu": command(["lscpu"]), "logical_cores": os.cpu_count(),
               "physical_cores_reported": psutil.cpu_count(logical=False),
               "ram_bytes": psutil.virtual_memory().total, "packages": packages,
               "torch_cuda_version": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
               "ollama_version": api("/api/version"), "ollama_tags": api("/api/tags"),
               "ollama_model": api("/api/show", {"model": MODEL}),
               "ollama_ps_before": api("/api/ps"), "ollama_ps_cli_before": command(["ollama", "ps"]),
               "nvidia_smi": command(["nvidia-smi"]), "ffmpeg": command(["ffmpeg", "-version"]),
               "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}
        write_json(out / "environment.json", env)
        if args.reuse_results:
            with zipfile.ZipFile(args.reuse_results) as prior_zip:
                prior_env = json.loads(prior_zip.read("environment.json"))
                prior_config = json.loads(prior_zip.read("config.json"))
                prior_status = json.loads(prior_zip.read("run_status.json"))
                if prior_status.get("state") != "completed" or prior_config.get("backend_requested") != "cpu":
                    raise RuntimeError("Reference ZIP must be a completed CPU run")
                if prior_config.get("protocol") != PROTOCOL:
                    raise RuntimeError("Use a NEW protocol 3.0 CPU ZIP, not a September v2 archive")
                for key in ["system_prompt", "initial_prompt", "input_definitions", "gtts", "sampling_order_seed", "reply_tts"]:
                    if prior_config.get(key) != config[key]:
                        raise RuntimeError(f"Reference configuration mismatch: {key}")
                ref_options = {k: v for k, v in prior_config["options"].items() if k != "num_gpu"}
                if ref_options != {k: v for k, v in options.items() if k != "num_gpu"}:
                    raise RuntimeError("Reference model options differ")
            def digest(environment):
                matches = [m for m in environment["ollama_tags"].get("models", [])
                           if MODEL in (m.get("name"), m.get("model"))]
                if len(matches) != 1 or not matches[0].get("digest"):
                    raise RuntimeError("Cannot determine exact Ollama model digest")
                return matches[0]["digest"]
            if digest(env) != digest(prior_env):
                raise RuntimeError("CPU/T4 model digests differ; use the same model artifact before comparison.")
            if prior_config.get("protocol") != PROTOCOL or prior_config.get("system_prompt") != COACH_PROMPT:
                raise RuntimeError("Reference ZIP uses a different protocol or coaching prompt")
            version_differences = {k: {"reference": prior_env.get("packages", {}).get(k), "current": v}
                                   for k, v in packages.items()
                                   if prior_env.get("packages", {}).get(k) != v}
            env["reference_version_differences"] = version_differences
            if version_differences:
                print("WARNING: package versions differ from the reference run; see environment.json:",
                      version_differences, flush=True)
            env["reference_ollama_version"] = prior_env.get("ollama_version")
            write_json(out / "environment.json", env)
        corpus = out / "corpus"
        manifest = prepare_corpus(corpus, args.reuse_results)
        if model is None:
            print("Loading Whisper tiny (load time recorded separately)...", flush=True)
            begin = time.perf_counter()
            model = whisper.load_model("tiny", device="cuda" if args.backend == "t4" else "cpu")
            env["whisper_load_ms"] = (time.perf_counter() - begin) * 1000
            env["whisper_reused_from_app"] = False
        else:
            env["whisper_load_ms"] = None
            env["whisper_reused_from_app"] = True
        expected = "cuda" if args.backend == "t4" else "cpu"
        if next(model.parameters()).device.type != expected:
            raise RuntimeError("Whisper device mismatch; rerun app setup in the selected runtime.")
        # Tiny's model dimensions; avoids silently benchmarking another Whisper size.
        if model.dims.n_audio_state != 384 or model.dims.n_audio_layer != 4 or model.dims.n_vocab != 51865:
            raise RuntimeError("Expected multilingual Whisper tiny from the supplied app.")
        env["whisper_device"] = str(next(model.parameters()).device)
        env["whisper_dimensions"] = vars(model.dims)
        env["torch_num_threads"] = torch.get_num_threads()
        env["torch_num_interop_threads"] = torch.get_num_interop_threads()
        env["notebook_system_prompt_matches_original"] = (
            globals().get("SYSTEM_PROMPT") == COACH_PROMPT
        )
        if "SYSTEM_PROMPT" in globals() and globals()["SYSTEM_PROMPT"] != COACH_PROMPT:
            print("NOTE: Notebook SYSTEM_PROMPT differs. Benchmark uses the original "
                  "diction/STAR/follow-up prompt saved in config.json; notebook variable "
                  "is not modified.", flush=True)
        write_json(out / "environment.json", env)
        items = manifest["items"]
        plan = build_plan(items)
        schedule = [{"phase": p, "block": b, "workload": i["workload"], "prompt_id": i["prompt_id"]}
                    for p, b, i, _ in plan]
        write_json(out / "schedule.json", schedule)
        if args.reuse_results:
            with zipfile.ZipFile(args.reuse_results) as z:
                if json.loads(z.read("schedule.json")) != schedule:
                    raise RuntimeError("Reference schedule differs")
        print(f"Starting {args.backend}: 1 startup + 4 warm-ups + 40 measured = 45 requests.", flush=True)
        print("Reply gTTS enabled. Do not use the interview UI while this runs.", flush=True)
        for index, (phase, block, item, seconds) in enumerate(plan, 1):
            print(f"[{index}/45] {phase}: {item['prompt_id']} (LLM deadline {seconds}s)", flush=True)
            row = trial(model, item, corpus, phase, index, block, options, args.backend, seconds)
            rows.append(row)
            with (out / "attempts.jsonl").open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            if row["status"] != "ok":
                raise RuntimeError(f"Stopped at {phase}/{item['prompt_id']}: {row.get('error')}. "
                                   "No retry. Restart runtime and app before another collection.")
            print(f"  STT {row['stt_ms']/1000:.3f}s | LLM {row['llm_client_ms']/1000:.3f}s | "
                  f"Pipeline {row['pipeline_ms']/1000:.3f}s | TTS {row['tts_ms']/1000:.3f}s | "
                  f"Audio ready {row['feedback_audio_ready_ms']/1000:.3f}s | "
                  f"{row['output_tokens']} actual tokens | {row['ollama_tokens_per_s']:.2f} tok/s", flush=True)
            if phase != "measured":
                write_json(out / f"ollama_ps_after_{index}.json", command(["ollama", "ps"]))
        if len(rows) != 45:
            raise RuntimeError("Incomplete request count")
        if args.reuse_results:
            with zipfile.ZipFile(args.reuse_results) as z:
                reference = json.loads(z.read("summary.json"))["measured_by_workload"]
            current = summarize(rows)["measured_by_workload"]
            comparison = {"reference_cpu_zip": str(args.reuse_results), "by_workload": {},
                          "note": "TTS/audio-ready ratios include external service variability and generated response length."}
            for w in WORKLOADS:
                comparison["by_workload"][w] = {}
                for key in ["stt_ms", "llm_client_ms", "pipeline_ms", "tts_ms", "feedback_audio_ready_ms"]:
                    cpu = reference[w]["metrics_success_only"][key]["mean"]
                    gpu = current[w]["metrics_success_only"][key]["mean"]
                    comparison["by_workload"][w][key] = {"cpu_mean_ms": cpu, "t4_mean_ms": gpu,
                        "cpu_over_t4_mean_ratio": cpu/gpu, "t4_mean_reduction_percent": 100*(1-gpu/cpu)}
            write_json(out / "comparison.json", comparison)
        status["state"] = "completed"
    except (Exception, DeadlineExceeded, KeyboardInterrupt) as exc:
        status.update(state="aborted", error_type=type(exc).__name__, error=str(exc))
        print("\nABORTED:", exc, flush=True)
    finally:
        status["finished_utc"] = utc()
        archive = package_results(out, rows, status)
        if status["state"] == "completed":
            summary = summarize(rows)["measured_by_workload"]
            print("\nMEASURED RESULTS BY WORKLOAD (mean ± sample SD; seconds)", flush=True)
            for workload, group in summary.items():
                print(f"{workload.upper()}: {group['successful']} successful measured requests", flush=True)
                for key in ["stt_ms", "llm_client_ms", "pipeline_ms", "tts_ms", "feedback_audio_ready_ms"]:
                    v = group["metrics_success_only"][key]
                    print(f"  {key}: {v['mean']/1000:.3f} ± {v['sample_sd']/1000:.3f} s; "
                          f"median {v['median']/1000:.3f}; P95 {v['p95_linear']/1000:.3f}", flush=True)
        if args.download:
            try:
                from google.colab import files
                files.download(str(archive))
            except Exception as exc:
                print("Automatic download unavailable; download the ZIP from Colab Files:", exc)
    return status["state"] == "completed"


def self_test():
    import tempfile
    from collections import Counter
    assert all(80 <= len(p.split()) <= 120 and p.count(".") == 5 for p in LONG_PROMPTS)
    plan = build_plan(PROMPT_SPECS)
    assert len(plan) == 45 and plan == build_plan(PROMPT_SPECS)
    assert Counter(p for p, _, _, _ in plan) == {"startup_observation": 1, "warmup": 4, "measured": 40}
    assert Counter(i["workload"] for p, _, i, _ in plan if p == "warmup") == {"short": 2, "long": 2}
    counts = Counter(i["prompt_id"] for p, _, i, _ in plan if p == "measured")
    assert len(counts) == 10 and set(counts.values()) == {4}
    assert abs(percentile([1, 2, 3, 4], 95) - 3.85) < 1e-10
    assert describe([]) == {"n": 0}
    assert describe([4])["sample_sd"] is None
    rows = [{"phase": "warmup", "status": "ok", "workload": "short", "prompt_id": "short_01", "local_stage_sum_ms": 999},
            {"phase": "measured", "status": "ok", "workload": "short", "prompt_id": "short_01", "local_stage_sum_ms": 100},
            {"phase": "measured", "status": "failed", "workload": "long", "prompt_id": "long_02", "local_stage_sum_ms": 1000}]
    summary = summarize(rows)
    assert summary["phases"]["measured"]["attempts"] == 2
    assert summary["phases"]["measured"]["metrics_success_only"]["local_stage_sum_ms"]["mean"] == 100
    assert summary["phases"]["measured"]["failed"] == 1
    assert summary["measured_by_prompt"]["long_02"]["successful"] == 0
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test"
        out.mkdir()
        archive = package_results(out, rows, {"state": "aborted"})
        with zipfile.ZipFile(archive) as z:
            assert "attempts.csv" in z.namelist()
            assert json.loads(z.read("run_status.json"))["state"] == "aborted"
    try:
        with deadline(0.02):
            time.sleep(0.2)
    except DeadlineExceeded:
        pass
    else:
        raise AssertionError("Deadline did not fire")
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0
    print("SELF-TEST PASSED: summary denominators, failure retention, ZIP output, deadline cleanup.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=["cpu", "t4"])
    parser.add_argument("--reuse-results", help="Prior results ZIP: reuse identical synthetic audio")
    parser.add_argument("--output-root", default="/content/pm_benchmark_results")
    parser.add_argument("--download", action="store_true", help="Trigger Colab browser download when finished")
    parser.add_argument("--self-test", action="store_true")
    explicit_cli = any(a == "--backend" or a.startswith("--backend=") or
                       a in ("--self-test", "--help", "-h") for a in sys.argv[1:])
    if "ipykernel" in sys.modules and not explicit_cli:
        # Ignore Jupyter's -f connection-file argument when pasted into a cell.
        parsed = parser.parse_args([])
        parsed.backend = COLAB_BACKEND
        parsed.reuse_results = COLAB_CPU_RESULTS_ZIP
        parsed.download = COLAB_DOWNLOAD_RESULTS
        if parsed.backend not in ("cpu", "t4"):
            parser.error('COLAB_BACKEND must be "cpu" or "t4"')
    else:
        parsed = parser.parse_args()
    if parsed.self_test:
        self_test()
    else:
        if not parsed.backend:
            parser.error("--backend cpu or --backend t4 is required")
        main(parsed)