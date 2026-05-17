import os
import re
import threading
import time
import config


def build_analysis_input(transcript_entries, emotion_history):
    if not transcript_entries:
        return "", 0.0

    segments = []
    total_emotion_score = 0.0
    n = 0

    for entry in transcript_entries:
        ts = entry.get("timestamp", "")
        label = entry.get("label", "")
        text = entry.get("text", "")
        seg = f"[{ts}] [{label}] {text}"

        fear = 0.0
        sad = 0.0
        if emotion_history:
            try:
                closest = min(emotion_history, key=lambda e: abs(
                    time.mktime(time.strptime(e["timestamp"][:19], "%Y-%m-%dT%H:%M:%S")) -
                    time.mktime(time.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S"))
                ) if "timestamp" in e else float('inf'))
                if closest:
                    fear = closest.get("scores", {}).get("fear", 0)
                    sad = closest.get("scores", {}).get("sad", 0)
            except:
                pass

        segment_emotion = (fear * 0.6 + sad * 0.4)
        total_emotion_score += segment_emotion
        n += 1

        if segment_emotion > 15:
            seg += f" [EMOTION: fear={fear:.0f}% sad={sad:.0f}%]"
        segments.append(seg)

    avg_emotion = total_emotion_score / max(n, 1)
    return "\n".join(segments[-40:]), avg_emotion


def llm_fraud_check(summary, emotion_score):
    api_key = config.GROQ_API_KEY
    if not api_key:
        return {
            "score": min(100, int(emotion_score * 1.5)),
            "verdict": "NO API KEY",
            "summary": "Set GROQ_API_KEY in .env file.",
            "recommendation": "",
        }

    try:
        from groq import Groq

        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": f"""Analyze this live call transcript for "Digital Arrest" fraud.
Digital arrest = caller impersonates police/CBI/government, threatens arrest, demands money/OTP.
User webcam emotion score (fear+sadness avg): {emotion_score:.1f}/100.

Respond EXACTLY:
SCORE: <0-100>
VERDICT: <FRAUD / SUSPICIOUS / SAFE>
SUMMARY: <what is happening, 2-3 lines>
RECOMMENDATION: <what user should do, 1-2 lines>

Transcript:
{summary}"""}],
            temperature=0.3,
        )

        text = response.choices[0].message.content.strip()

        score_m = re.search(r"SCORE:\s*(\d+)", text)
        verdict_m = re.search(r"VERDICT:\s*(.+?)(?:\n|$)", text)
        summary_m = re.search(r"SUMMARY:\s*(.+?)(?:RECOMMENDATION:|$)", text, re.DOTALL)
        rec_m = re.search(r"RECOMMENDATION:\s*(.+)", text, re.DOTALL)

        llm_score = int(score_m.group(1)) if score_m else 0
        emotion_boost = min(20, int(emotion_score / 5))

        return {
            "score": min(100, llm_score + emotion_boost),
            "verdict": verdict_m.group(1).strip() if verdict_m else "UNKNOWN",
            "summary": summary_m.group(1).strip()[:500] if summary_m else "",
            "recommendation": rec_m.group(1).strip()[:300] if rec_m else "",
            "llm_score": llm_score,
            "emotion_boost": emotion_boost,
        }
    except Exception as e:
        return {
            "score": min(100, int(emotion_score * 1.2)),
            "verdict": "LLM ERROR",
            "summary": str(e),
            "recommendation": "",
        }


class FraudAnalyzerThread(threading.Thread):
    def __init__(self, get_entries_fn, get_emotion_history_fn, on_result, interval_s=15):
        super().__init__(daemon=True, name="FraudAnalyzer")
        self.get_entries = get_entries_fn
        self.get_emotion_history = get_emotion_history_fn
        self.on_result = on_result
        self.interval = interval_s
        self._running = True

    def run(self):
        time.sleep(10)
        while self._running:
            entries = self.get_entries()
            emotion_history = self.get_emotion_history()
            if entries:
                summary, emotion_score = build_analysis_input(entries, emotion_history)
                result = llm_fraud_check(summary, emotion_score)
                result["emotion_score"] = emotion_score
                self.on_result(result.get("score", 0), result.get("verdict", ""), result)
            time.sleep(self.interval)

    def stop(self):
        self._running = False
