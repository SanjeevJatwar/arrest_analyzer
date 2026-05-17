"""
core/fusion/fraud_analyzer.py

Analyzes accumulated transcript for fraud indicators.
Two modes:
  1. Keyword scanner (always available, no API needed)
  2. LLM-based analysis via Google Gemini API (if key is set)
"""
import os
import re
import threading
import time
import config


# ── Keyword / pattern-based scanner ──────────────────────────────

_FRAUD_PATTERNS = [
    # Impersonation
    r"\b(police|cbi|cid|fbi|irs|customs|narcotics|interpol|cyber.?cell)\b",
    r"\b(officer|inspector|commissioner|judge|magistrate|constable)\b",
    r"\b(warrant|arrest|fir|case\s*number|section\s*\d+)\b",
    # Threats
    r"\b(arrest|jail|prison|custody|suspend|freeze|block|seize)\b",
    r"\b(immediately|urgent|right\s*now|within\s*\d+\s*(hour|minute))\b",
    # Money demands
    r"\b(transfer|payment|amount|fine|penalty|fee|deposit|bail)\b",
    r"\b(upi|bank\s*account|neft|rtgs|wire|western\s*union|crypto)\b",
    r"\b(otp|pin|cvv|password|aadh?ar|pan\s*card|ssn)\b",
    # Isolation tactics
    r"\b(don.?t\s*tell|keep\s*secret|confidential|classified|recorded)\b",
    r"\b(don.?t\s*hang\s*up|stay\s*on\s*(the\s*)?line|do\s*not\s*disconnect)\b",
    # Emotional manipulation
    r"\b(scared|afraid|worry|serious\s*trouble|criminal|guilty|launder)\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _FRAUD_PATTERNS]


def keyword_scan(transcript: str) -> dict:
    """
    Returns { 'score': 0-100, 'matches': [...], 'verdict': str }
    """
    if not transcript.strip():
        return {"score": 0, "matches": [], "verdict": "INSUFFICIENT DATA"}

    matches = []
    for pattern in _COMPILED:
        found = pattern.findall(transcript)
        matches.extend(found)

    unique = list(set(m.lower() for m in matches))
    score = min(100, int(len(unique) / len(_COMPILED) * 100 * 2.5))

    if score >= 70:
        verdict = "HIGH RISK — LIKELY FRAUD"
    elif score >= 40:
        verdict = "MODERATE RISK — SUSPICIOUS"
    elif score >= 15:
        verdict = "LOW RISK — SOME INDICATORS"
    else:
        verdict = "MINIMAL RISK"

    return {"score": score, "matches": unique[:20], "verdict": verdict}


# ── LLM-based analysis (Gemini) ─────────────────────────────────

def llm_analyze(transcript: str) -> dict:
    """
    Uses Google Gemini API to analyze transcript for fraud.
    Returns { 'score': 0-100, 'analysis': str, 'verdict': str }
    Falls back to keyword scan if no API key.
    """
    api_key = config.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        result = keyword_scan(transcript)
        result["method"] = "keyword"
        return result

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        prompt = f"""Analyze the following phone/video call transcript for signs of a 
"digital arrest" scam or fraud. Digital arrest scams involve callers 
impersonating law enforcement, threatening arrest, demanding money, 
or asking for sensitive information.

Respond in EXACTLY this format (no markdown):
SCORE: <number 0-100>
VERDICT: <one of: LIKELY FRAUD / SUSPICIOUS / LOW RISK / CLEAN>
ANALYSIS: <2-3 sentence explanation>

Transcript:
{transcript[-3000:]}"""

        response = model.generate_content(prompt)
        text = response.text.strip()

        score_match = re.search(r"SCORE:\s*(\d+)", text)
        verdict_match = re.search(r"VERDICT:\s*(.+?)(?:\n|$)", text)
        analysis_match = re.search(r"ANALYSIS:\s*(.+)", text, re.DOTALL)

        return {
            "score": int(score_match.group(1)) if score_match else 0,
            "verdict": verdict_match.group(1).strip() if verdict_match else "UNKNOWN",
            "analysis": analysis_match.group(1).strip()[:500] if analysis_match else text[:500],
            "method": "gemini",
        }
    except Exception as e:
        print(f"[FraudAnalyzer] LLM error: {e}, falling back to keyword scan")
        result = keyword_scan(transcript)
        result["method"] = "keyword-fallback"
        return result


# ── Background analyzer thread ───────────────────────────────────

class FraudAnalyzerThread(threading.Thread):
    """
    Periodically analyzes the accumulated transcript for fraud.
    Calls on_result(score, verdict, details) on each check.
    """

    def __init__(self, get_transcript_fn, on_result, interval_s=30):
        super().__init__(daemon=True, name="FraudAnalyzer")
        self.get_transcript = get_transcript_fn
        self.on_result      = on_result
        self.interval        = interval_s
        self._running        = True

    def run(self):
        time.sleep(10)  # wait for initial transcript to accumulate
        while self._running:
            transcript = self.get_transcript()
            if transcript.strip():
                result = llm_analyze(transcript)
                self.on_result(
                    result.get("score", 0),
                    result.get("verdict", ""),
                    result,
                )
            time.sleep(self.interval)

    def stop(self):
        self._running = False
