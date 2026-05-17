"""
core/fusion/fraud_analyzer.py

Analyzes accumulated transcript (LOCAL + REMOTE combined) for fraud indicators.
Supports both Hindi and English keywords.
Two modes:
  1. Keyword scanner (always available, no API needed)
  2. LLM-based analysis via Google Gemini API (if key is set)

Generates real-time suggestions and alert notifications.
"""
import os
import re
import threading
import time
import config


# ── Hindi + English fraud keyword patterns ───────────────────────

_FRAUD_PATTERNS = [
    # --- Impersonation (Hindi + English) ---
    r"\b(police|polic|CBI|CID|FBI|IRS|customs|narcotics|interpol|cyber\s*cell)\b",
    r"(पुलिस|सीबीआई|सीआईडी|साइबर\s*सेल|कस्टम|नारकोटिक्स|इंटरपोल)",
    r"\b(officer|inspector|commissioner|judge|magistrate|constable)\b",
    r"(अधिकारी|इंस्पेक्टर|कमिश्नर|जज|मजिस्ट्रेट|कांस्टेबल|अफसर|साहब)",
    r"\b(warrant|arrest|FIR|case\s*number|section\s*\d+)\b",
    r"(वारंट|गिरफ्तार|एफआईआर|केस\s*नंबर|धारा)",

    # --- Threats (Hindi + English) ---
    r"\b(arrest|jail|prison|custody|suspend|freeze|block|seize)\b",
    r"(गिरफ्तार|जेल|कारावास|हिरासत|सस्पेंड|फ्रीज|ब्लॉक|जब्त)",
    r"\b(immediately|urgent|right\s*now|within\s*\d+\s*(hour|minute))\b",
    r"(तुरंत|अभी|फौरन|जल्दी|अर्जेंट|तत्काल)",

    # --- Money demands ---
    r"\b(transfer|payment|amount|fine|penalty|fee|deposit|bail)\b",
    r"(ट्रांसफर|पेमेंट|रकम|जुर्माना|पेनल्टी|फीस|डिपॉजिट|जमानत|पैसे|रुपये)",
    r"\b(UPI|bank\s*account|NEFT|RTGS|wire|crypto|bitcoin)\b",
    r"(यूपीआई|बैंक\s*अकाउंट|खाता|नेफ्ट|आरटीजीएस)",
    r"\b(OTP|PIN|CVV|password|aadhaa?r|pan\s*card|SSN)\b",
    r"(ओटीपी|पिन|सीवीवी|पासवर्ड|आधार|पैन\s*कार्ड)",

    # --- Isolation tactics ---
    r"\b(don.?t\s*tell|keep\s*secret|confidential|classified|recorded)\b",
    r"(किसी\s*को\s*मत\s*बताना|गोपनीय|सीक्रेट|रिकॉर्ड|गुप्त)",
    r"\b(don.?t\s*hang\s*up|stay\s*on|do\s*not\s*disconnect)\b",
    r"(फोन\s*मत\s*काटना|लाइन\s*पर\s*रहो|डिस्कनेक्ट\s*मत\s*करो|काटना\s*मत)",

    # --- Emotional manipulation ---
    r"\b(scared|afraid|trouble|criminal|guilty|launder|money\s*launder)\b",
    r"(डर|डरो|मुसीबत|अपराधी|दोषी|मनी\s*लॉन्ड्रिंग|हवाला|काला\s*धन)",

    # --- Verification scam ---
    r"(verify|verification|Aadhaar\s*link|link\s*your|update\s*KYC)",
    r"(वेरिफिकेशन|सत्यापन|आधार\s*लिंक|केवाईसी\s*अपडेट)",
]

_COMPILED = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in _FRAUD_PATTERNS]


# ── Suggestions based on detected patterns ───────────────────────

_SUGGESTIONS = {
    "impersonation": {
        "keywords": ["police", "cbi", "cid", "officer", "inspector", "warrant",
                     "fir", "पुलिस", "सीबीआई", "अधिकारी", "इंस्पेक्टर",
                     "वारंट", "एफआईआर", "अफसर"],
        "msg_hi": "!! चेतावनी: कॉलर सरकारी अधिकारी होने का दावा कर रहा है। असली पुलिस कभी फोन पर पैसे नहीं मांगती।",
        "msg_en": "!! WARNING: Caller claims to be a government official. Real police NEVER demand money over phone.",
    },
    "money_demand": {
        "keywords": ["transfer", "payment", "upi", "bank", "otp", "pin",
                     "ट्रांसफर", "पेमेंट", "यूपीआई", "ओटीपी", "पैसे", "रुपये",
                     "खाता", "जमानत"],
        "msg_hi": "!! चेतावनी: पैसे या बैंक जानकारी मांगी जा रही है। कभी भी OTP, PIN या बैंक डिटेल्स शेयर न करें!",
        "msg_en": "!! WARNING: Money or bank details being requested. NEVER share OTP, PIN, or bank details!",
    },
    "threat": {
        "keywords": ["arrest", "jail", "custody", "immediately", "urgent",
                     "गिरफ्तार", "जेल", "हिरासत", "तुरंत", "फौरन"],
        "msg_hi": "!! चेतावनी: गिरफ्तारी की धमकी दी जा रही है। यह डिजिटल अरेस्ट स्कैम हो सकता है!",
        "msg_en": "!! WARNING: Arrest threats detected. This could be a DIGITAL ARREST SCAM!",
    },
    "isolation": {
        "keywords": ["secret", "don't tell", "don't hang up", "confidential",
                     "किसी को मत बताना", "गोपनीय", "फोन मत काटना"],
        "msg_hi": "!! चेतावनी: आपको अलग रखने की कोशिश हो रही है। तुरंत किसी परिवार वाले को बताएं!",
        "msg_en": "!! WARNING: Caller trying to isolate you. IMMEDIATELY inform a family member!",
    },
}


def keyword_scan(transcript: str) -> dict:
    """
    Returns { 'score': 0-100, 'matches': [...], 'verdict': str,
              'suggestions': [...] }
    """
    if not transcript.strip():
        return {"score": 0, "matches": [], "verdict": "INSUFFICIENT DATA",
                "suggestions": []}

    matches = []
    for pattern in _COMPILED:
        found = pattern.findall(transcript)
        matches.extend(found)

    unique = list(set(m.lower() if isinstance(m, str) else m for m in matches))
    score = min(100, int(len(unique) / max(len(_COMPILED), 1) * 100 * 2.5))

    # Generate suggestions
    suggestions = []
    unique_lower = [u.lower() if isinstance(u, str) else u for u in unique]
    for cat, info in _SUGGESTIONS.items():
        for kw in info["keywords"]:
            if any(kw.lower() in u for u in unique_lower):
                suggestions.append({
                    "category": cat,
                    "message_hi": info["msg_hi"],
                    "message_en": info["msg_en"],
                })
                break

    if score >= 70:
        verdict = "HIGH RISK — LIKELY FRAUD / उच्च जोखिम — संभावित धोखाधड़ी"
    elif score >= 40:
        verdict = "MODERATE RISK — SUSPICIOUS / मध्यम जोखिम — संदिग्ध"
    elif score >= 15:
        verdict = "LOW RISK / कम जोखिम"
    else:
        verdict = "MINIMAL RISK / न्यूनतम जोखिम"

    return {"score": score, "matches": unique[:20], "verdict": verdict,
            "suggestions": suggestions}


# ── LLM-based analysis (Gemini) ─────────────────────────────────

def llm_analyze(transcript: str) -> dict:
    """
    Uses Google Gemini API to analyze transcript for fraud.
    Falls back to keyword scan if no API key.
    """
    api_key = config.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")

    # Always run keyword scan first
    kw_result = keyword_scan(transcript)

    if not api_key:
        kw_result["method"] = "keyword"
        return kw_result

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        prompt = f"""You are a fraud detection system analyzing a live phone/video call 
transcript. The conversation may be in Hindi, English, or Hinglish (mixed).

This is about "Digital Arrest" scams where callers impersonate police/CBI/government 
officers, threaten arrest, and demand money or personal details.

Analyze the following transcript from BOTH the local user and remote caller.
Give your response in this EXACT format (no markdown):

SCORE: <number 0-100, where 100 = definite fraud>
VERDICT: <LIKELY FRAUD / SUSPICIOUS / LOW RISK / CLEAN>
SUGGESTION_HI: <1-2 line actionable advice in Hindi for the user>
SUGGESTION_EN: <1-2 line actionable advice in English for the user>
ANALYSIS: <2-3 sentence explanation of why>

Transcript (LOCAL = user, REMOTE = caller):
{transcript[-4000:]}"""

        response = model.generate_content(prompt)
        text = response.text.strip()

        score_m   = re.search(r"SCORE:\s*(\d+)", text)
        verdict_m = re.search(r"VERDICT:\s*(.+?)(?:\n|$)", text)
        sug_hi_m  = re.search(r"SUGGESTION_HI:\s*(.+?)(?:\n|$)", text)
        sug_en_m  = re.search(r"SUGGESTION_EN:\s*(.+?)(?:\n|$)", text)
        analysis_m = re.search(r"ANALYSIS:\s*(.+)", text, re.DOTALL)

        llm_score = int(score_m.group(1)) if score_m else kw_result["score"]
        final_score = max(llm_score, kw_result["score"])

        suggestions = list(kw_result.get("suggestions", []))
        if sug_hi_m or sug_en_m:
            suggestions.insert(0, {
                "category": "llm",
                "message_hi": sug_hi_m.group(1).strip() if sug_hi_m else "",
                "message_en": sug_en_m.group(1).strip() if sug_en_m else "",
            })

        return {
            "score": final_score,
            "verdict": verdict_m.group(1).strip() if verdict_m else kw_result["verdict"],
            "analysis": analysis_m.group(1).strip()[:500] if analysis_m else "",
            "suggestions": suggestions,
            "matches": kw_result["matches"],
            "method": "gemini+keyword",
        }
    except Exception as e:
        print(f"[FraudAnalyzer] LLM error: {e}, using keyword scan")
        kw_result["method"] = "keyword-fallback"
        return kw_result


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
        self.interval       = interval_s
        self._running       = True
        self._last_score    = 0

    def run(self):
        time.sleep(8)  # wait for initial transcript to accumulate
        while self._running:
            transcript = self.get_transcript()
            if transcript.strip():
                result = llm_analyze(transcript)
                self._last_score = result.get("score", 0)
                self.on_result(
                    result.get("score", 0),
                    result.get("verdict", ""),
                    result,
                )
            time.sleep(self.interval)

    def stop(self):
        self._running = False
