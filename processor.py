"""
processor.py — Autonomous Review Intelligence Worker  v4.3  (Cloud-Optimized)
──────────────────────────────────────────────────────────────────────────────
  • Primary model  : deepseek-ai/deepseek-v4-flash  (5x faster, stable)
  • Fallback       : meta/llama-3.1-8b-instruct
  • Base URL       : https://integrate.api.nvidia.com/v1
  • Concurrency    : asyncio + Semaphore(3)
  • Rate limit     : 25 RPM governor
  • 429 backoff    : immediate 60 s sleep then step to fallback
  • API timeout    : 90 s
  • Retry backoff  : 5 s → 15 s → 45 s
  • Run limit      : 500 reviews then clean exit (GitHub Actions safe)
  • Env vars       : os.environ.get() — compatible with GitHub Secrets
  • Logging        : Dual output → console + pipeline.log
  • Think blocks   : <think>...</think> stripped before JSON parse
  • Intelligence   : intelligence_core.txt as absolute system prompt
"""

import os
import re
import json
import asyncio
import logging
import sys
import time
from datetime import datetime, timezone
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import AsyncOpenAI, APITimeoutError, APIConnectionError, RateLimitError

# ── Dual Logging ──────────────────────────────────────────────────────────────
log = logging.getLogger("pipeline")
log.setLevel(logging.DEBUG)
_fmt = logging.Formatter("%(asctime)s  [%(levelname)-8s]  %(message)s",
                         datefmt="%Y-%m-%d %H:%M:%S")
for handler in [
    logging.FileHandler("pipeline.log", encoding="utf-8", mode="a"),
    logging.StreamHandler(sys.stdout),
]:
    handler.setFormatter(_fmt)
    log.addHandler(handler)

# ── Config ────────────────────────────────────────────────────────────────────
# Model chain: flash as primary (stable + fast), micro as emergency fallback
MODEL_PRIMARY    = "deepseek-ai/deepseek-v4-flash"   # 5x faster, high stability
MODEL_MICRO      = "meta/llama-3.1-8b-instruct"      # near-zero 429 risk
MODEL_CHAIN      = [MODEL_PRIMARY, MODEL_MICRO]
BASE_URL         = "https://integrate.api.nvidia.com/v1"

CONCURRENCY      = 3           # Semaphore — avoids TPM burst
RPM_LIMIT        = 25          # Requests per minute — safe free-tier buffer
RATE_429_SLEEP   = 60          # Sleep duration on any 429 before stepping down
API_TIMEOUT_SECS = 90          # Flash model responds much faster than Pro
IDLE_SLEEP_SECS  = 60
FETCH_PAGE_SIZE  = 50          # Smaller page = more frequent progress logs
CHECKPOINT_EVERY = 50
RETRY_DELAYS     = [5, 15, 45]
TOTAL_EST        = 15636
RUN_LIMIT        = 500         # Exit cleanly after N reviews (GitHub Actions cap)

VALID_CATEGORIES = {
    "Price & Value",
    "Taste & Food Quality",
    "Wait Time & Speed",
    "Staff Attitude & Service",
    "Positive Reinforcement",
}

# ── Environment ── (load_dotenv for local; os.environ for GitHub Secrets) ─────
load_dotenv()   # no-op when vars are already set via GitHub Secrets / system env
SUPABASE_URL   = os.environ.get("SUPABASE_URL")
SUPABASE_KEY   = os.environ.get("SUPABASE_KEY")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")

for name, val in [("SUPABASE_URL", SUPABASE_URL),
                  ("SUPABASE_KEY", SUPABASE_KEY),
                  ("NVIDIA_API_KEY", NVIDIA_API_KEY)]:
    if not val:
        log.critical(f"Missing required env var: {name}. Halting.")
        sys.exit(1)

# ── Clients ───────────────────────────────────────────────────────────────────
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

nvidia_client = AsyncOpenAI(
    base_url=BASE_URL,
    api_key=NVIDIA_API_KEY,
    timeout=API_TIMEOUT_SECS,
)

# ── Rate Limiter (Token Bucket, 38 RPM) ───────────────────────────────────────
class RateLimiter:
    """
    Token bucket that enforces a maximum of `rpm` requests per 60 seconds.
    Callers await acquire() before each API request.
    """
    def __init__(self, rpm: int):
        self._interval = 60.0 / rpm   # seconds between tokens
        self._lock     = asyncio.Lock()
        self._next_ok  = 0.0           # earliest time next call is allowed

    async def acquire(self):
        async with self._lock:
            now  = asyncio.get_event_loop().time()
            wait = max(0.0, self._next_ok - now)
            self._next_ok = max(now, self._next_ok) + self._interval
        if wait > 0:
            await asyncio.sleep(wait)

rate_limiter = RateLimiter(RPM_LIMIT)

# ── Load Intelligence Core ────────────────────────────────────────────────────
_intel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "intelligence_core.txt")
try:
    with open(_intel_path, "r", encoding="utf-8") as _f:
        INTELLIGENCE_CORE = _f.read()
    log.info(f"Intelligence Core loaded — {len(INTELLIGENCE_CORE):,} characters.")
except FileNotFoundError:
    log.critical(f"intelligence_core.txt not found at: {_intel_path}. Halting.")
    sys.exit(1)

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are an elite Restaurant Operations Consultant applying Enlightened Hospitality and TvA Costing frameworks. Your sole source of truth is the Intelligence Core below.

Focus specifically on identifying:
- Price & Value: Scan for signals like "not worth it". Diagnose issues using terms like "Supplier Price Drift" or "Violating the Plowhorse Rule".
- Taste & Food Quality: Identify "FOH-BOH Disconnects" for cold food and suggest "Appointing an Expeditor (Expo)" as the fix.
- Wait Time & Speed: Detect "Omnichannel Overload" and suggest fixes like "RevPASH Optimization" or "Dynamic Throttling".
- Staff Attitude & Service: Identify signals of rude service and diagnose a lack of "51 percenters".

=== INTELLIGENCE CORE ===
{INTELLIGENCE_CORE}
=========================

You will receive ONE restaurant review. Return ONLY a single raw JSON object. No markdown. No explanation. No preamble. No ```json fences.

Required keys (all must be present):

  "complaint_category"     : Exactly one of: "Price & Value" | "Taste & Food Quality" | "Wait Time & Speed" | "Staff Attitude & Service" | "Positive Reinforcement"
  "operational_diagnosis"  : The Operational Why from the Intelligence Core (e.g. "FOH-BOH Disconnect", "Violating the Plowhorse Rule", "RevPASH Bottleneck — Omnichannel Overload", "Lack of 51-percenters", "Supplier Price Drift", "Positive Reinforcement")
  "management_fix"         : The exact Management Fix from the Intelligence Core (e.g. "Implement TvA Costing & Menu Engineering", "Appoint Strong Expeditor (Expo)", "Optimize RevPASH & Dynamic Throttling", "Prioritize Employee Experience & Hire 51-percenters"). Use "N/A" for positive reviews.
  "recovery_reply"         : A personalized, non-defensive owner response using the "5 A's of Mistake Recovery" (Awareness, Acknowledgement, Apology, Action, Additional Generosity). Prioritize writing a "great last chapter". For positive reviews, write a loyalty-reinforcing thank-you.
  "strategic_tags"         : JSON array of strings using exact framework terms from the Intelligence Core e.g. ["Plowhorse Rule", "TvA Costing", "RevPASH", "FOH-BOH Disconnect", "51-Percenter", "Supplier Price Drift", "Enlightened Hospitality", "Menu Engineering", "5 A's"]
  "sentiment_metrics"      : JSON object — {{"food": <int 1-10>, "service": <int 1-10>, "value": <int 1-10>, "vibe": <int 1-10>}}
  "urgency_score"          : Integer 1-10. 10 = Local Guide + critical operational failure. 1 = minor positive.

OUTPUT RULES — STRICTLY ENFORCED:
1. Raw JSON object ONLY. Nothing before or after the opening and closing braces.
2. All string values must be properly JSON-escaped.
3. Output must be parseable by Python's json.loads() with no preprocessing.
"""

# ── Output Sanitizer ──────────────────────────────────────────────────────────

def clean_output(text: str) -> str:
    """Strip <think> blocks, markdown fences, and leading/trailing whitespace."""
    # Remove DeepSeek internal reasoning blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def validate_result(raw: dict) -> dict:
    """Sanitize all fields in-place with safe defaults."""
    if raw.get("complaint_category") not in VALID_CATEGORIES:
        raw["complaint_category"] = "Taste & Food Quality"

    raw.setdefault("operational_diagnosis", "Requires Manual Review")
    raw.setdefault("management_fix", "N/A")
    raw.setdefault("recovery_reply", "")

    # strategic_tags must be a list of strings
    tags = raw.get("strategic_tags", [])
    raw["strategic_tags"] = [str(t) for t in tags] if isinstance(tags, list) else []

    # sentiment_metrics must be a dict with int scores 1-10
    sm = raw.get("sentiment_metrics", {})
    if not isinstance(sm, dict):
        sm = {}
    for key in ("food", "service", "value", "vibe"):
        try:
            sm[key] = max(1, min(10, int(sm.get(key, 5))))
        except (TypeError, ValueError):
            sm[key] = 5
    raw["sentiment_metrics"] = sm

    # urgency_score must be int 1-10
    try:
        raw["urgency_score"] = max(1, min(10, int(raw.get("urgency_score", 5))))
    except (TypeError, ValueError):
        raw["urgency_score"] = 5

    return raw


# ── Single-Review API Call ────────────────────────────────────────────────────

async def analyze_one(review: dict, semaphore: asyncio.Semaphore) -> dict | None:
    """
    Analyze one review walking MODEL_CHAIN (pro → flash → micro).
    Any 429 RateLimitError triggers an immediate RATE_429_SLEEP before
    stepping down to the next lighter model in the chain.
    Returns validated result dict or None if entire chain is exhausted.
    """
    rid = review.get("id", "?")
    payload = (
        f"Review ID     : {rid}\n"
        f"Restaurant    : {review.get('restaurant_name', '')}\n"
        f"Reviewer      : {review.get('reviewer_name', '')} "
        f"({'Local Guide' if review.get('is_local_guide') else 'Regular'})\n"
        f"Rating        : {review.get('rating', 'N/A')} / 5 stars\n"
        f"Review Text   : {(review.get('review_text') or '').strip()}"
    )

    async with semaphore:
        for model_idx, model in enumerate(MODEL_CHAIN):
            model_label = model.split("/")[-1]   # short name for logs
            for attempt, delay in enumerate(RETRY_DELAYS, start=1):
                await rate_limiter.acquire()
                try:
                    log.debug(f"  [{rid}] {model_label} attempt {attempt}...")
                    resp = await nvidia_client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user",   "content": payload},
                        ],
                        temperature=0.1,
                        max_tokens=1200,
                    )
                    raw_text = resp.choices[0].message.content
                    clean    = clean_output(raw_text)
                    parsed   = json.loads(clean)

                    if not isinstance(parsed, dict):
                        raise ValueError(f"Expected dict, got {type(parsed).__name__}")

                    log.debug(f"  [{rid}] Success via {model_label}.")
                    return validate_result(parsed)

                except RateLimitError as e:
                    # 429 — sleep 60s immediately, then step to lighter model
                    log.warning(
                        f"  [{rid}] 429 on {model_label} (attempt {attempt}). "
                        f"Sleeping {RATE_429_SLEEP}s then stepping down model..."
                    )
                    await asyncio.sleep(RATE_429_SLEEP)
                    break   # exit inner retry loop → next model in chain

                except (APITimeoutError, APIConnectionError) as e:
                    log.warning(f"  [{rid}] {model_label} attempt {attempt} timeout/conn: {e}")

                except json.JSONDecodeError as e:
                    snippet = clean[:200] if "clean" in dir() else "<no output>"
                    log.error(f"  [{rid}] {model_label} attempt {attempt} JSON error: {e} | {snippet}")

                except Exception as e:
                    log.error(f"  [{rid}] {model_label} attempt {attempt} {type(e).__name__}: {e}")

                # Backoff before next retry on the same model
                if attempt < len(RETRY_DELAYS):
                    log.info(f"  [{rid}] Backing off {delay}s...")
                    await asyncio.sleep(delay)

            # All retries on this model used up (non-429 failures)
            if model_idx < len(MODEL_CHAIN) - 1:
                next_model = MODEL_CHAIN[model_idx + 1].split("/")[-1]
                log.warning(f"  [{rid}] {model_label} exhausted. Stepping to {next_model}...")

    log.error(f"  [{rid}] Entire model chain exhausted. Skipping row.")
    return None


# ── Database Helpers ──────────────────────────────────────────────────────────

def fetch_unprocessed(limit: int = FETCH_PAGE_SIZE) -> list[dict]:
    try:
        resp = (
            supabase.table("restaurant_reviews")
            .select("id, restaurant_name, reviewer_name, review_text, rating, is_local_guide")
            .is_("operational_diagnosis", "null")
            .limit(limit)
            .execute()
        )
        return resp.data or []
    except Exception as e:
        log.error(f"Supabase fetch error: {e}")
        return []


def write_to_db(review_id, result: dict) -> bool:
    try:
        supabase.table("restaurant_reviews").upsert({
            "id":                    review_id,
            "complaint_category":    result["complaint_category"],
            "operational_diagnosis": result["operational_diagnosis"],
            "management_fix":        result["management_fix"],
            "recovery_reply":        result["recovery_reply"],
            "strategic_tags":        result["strategic_tags"],
            "sentiment_metrics":     result["sentiment_metrics"],
            "urgency_score":         result["urgency_score"],
        }).execute()
        return True
    except Exception as e:
        log.error(f"  DB write error for ID {review_id}: {e}")
        return False


def count_done() -> int:
    try:
        resp = (
            supabase.table("restaurant_reviews")
            .select("id", count="exact")
            .not_.is_("operational_diagnosis", "null")
            .execute()
        )
        return resp.count or 0
    except Exception:
        return -1


# ── Checkpoint ────────────────────────────────────────────────────────────────

def checkpoint(session_ok: int, session_fail: int, start_ts: float):
    done    = count_done()
    elapsed = time.time() - start_ts
    rate    = session_ok / elapsed * 3600 if elapsed > 0 else 0
    pct     = done / TOTAL_EST * 100 if done > 0 else 0
    eta_hrs = (TOTAL_EST - done) / rate if rate > 0 else float("inf")

    log.info("=" * 65)
    log.info(f"  CHECKPOINT  —  {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
    log.info(f"  Elapsed       : {elapsed/60:.1f} min")
    log.info(f"  Session OK    : {session_ok:,}  |  Skipped: {session_fail}")
    log.info(f"  DB total done : {done:,} / ~{TOTAL_EST:,}  ({pct:.1f}%)")
    log.info(f"  Throughput    : ~{rate:,.0f} reviews / hour")
    log.info(f"  ETA           : ~{eta_hrs:.1f} hours remaining")
    log.info("=" * 65)


# ── Concurrent Page Processor ─────────────────────────────────────────────────

async def process_page(reviews: list[dict], semaphore: asyncio.Semaphore) -> tuple[int, int]:
    tasks   = [analyze_one(r, semaphore) for r in reviews]
    results = await asyncio.gather(*tasks)

    ok = fail = 0
    for review, result in zip(reviews, results):
        if result and write_to_db(review["id"], result):
            ok += 1
        else:
            fail += 1

    return ok, fail


# ── Autonomous Main Loop ──────────────────────────────────────────────────────

async def run():
    semaphore    = asyncio.Semaphore(CONCURRENCY)
    session_ok   = 0
    session_fail = 0
    start_ts     = time.time()
    next_chk     = CHECKPOINT_EVERY

    log.info("=" * 65)
    log.info("  AUTONOMOUS INTELLIGENCE WORKER  v4.3  (Cloud-Optimized)  —  STARTED")
    log.info(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    log.info(f"  Model chain    : {' → '.join(m.split('/')[-1] for m in MODEL_CHAIN)}")
    log.info(f"  Base URL       : {BASE_URL}")
    log.info(f"  Concurrency    : Semaphore({CONCURRENCY})  |  Rate cap: {RPM_LIMIT} RPM")
    log.info(f"  429 backoff    : {RATE_429_SLEEP}s sleep then step down model")
    log.info(f"  Timeout        : {API_TIMEOUT_SECS}s  |  Backoff: {RETRY_DELAYS}s")
    log.info(f"  Run limit      : {RUN_LIMIT} reviews then clean exit")
    log.info(f"  Fetch size     : {FETCH_PAGE_SIZE} reviews / iteration")
    log.info("=" * 65)

    while session_ok < RUN_LIMIT:
        # Clamp the next fetch so we never overshoot RUN_LIMIT
        remaining    = RUN_LIMIT - session_ok
        fetch_size   = min(FETCH_PAGE_SIZE, remaining)
        reviews      = fetch_unprocessed(limit=fetch_size)

        if not reviews:
            log.info("Queue empty — no NULL reviews remain. Exiting cleanly.")
            break

        log.info(
            f"Fetched {len(reviews)} reviews "
            f"({session_ok}/{RUN_LIMIT} done this run). "
            f"Dispatching {CONCURRENCY} workers @ {RPM_LIMIT} RPM..."
        )

        ok, fail   = await process_page(reviews, semaphore)
        session_ok   += ok
        session_fail += fail

        elapsed = time.time() - start_ts
        rate    = session_ok / elapsed * 3600 if elapsed > 0 else 0
        db_pct  = session_ok / TOTAL_EST * 100

        log.info(
            f"Page done — OK: {ok} | Skip: {fail} | "
            f"Run total: {session_ok}/{RUN_LIMIT} | "
            f"DB est: {db_pct:.1f}% | ~{rate:,.0f}/hr"
        )

        if session_ok >= next_chk:
            checkpoint(session_ok, session_fail, start_ts)
            next_chk += CHECKPOINT_EVERY

    # ── Clean exit summary ────────────────────────────────────────────────
    elapsed = time.time() - start_ts
    log.info("=" * 65)
    log.info(f"  RUN COMPLETE — processed {session_ok} reviews in {elapsed/60:.1f} min")
    log.info(f"  Skipped / failed : {session_fail}")
    log.info(f"  Exit reason      : {'run limit reached' if session_ok >= RUN_LIMIT else 'queue empty'}")
    log.info("=" * 65)


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        log.info("Worker stopped by user (Ctrl+C).")
        sys.exit(0)
    except Exception as e:
        log.critical(f"FATAL: {e}", exc_info=True)
        sys.exit(1)
