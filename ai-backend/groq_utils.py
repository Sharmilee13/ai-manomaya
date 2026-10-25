import os
import re
import time

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

EMOTION_KEYS = frozenset(
    {"trust", "attachment", "awkwardness", "resentment", "comfort"}
)


def coerce_emotion_value(value):
    """Normalize LLM emotion values to int 0-100."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return max(0, min(100, int(round(value))))
    if isinstance(value, str):
        cleaned = value.strip().rstrip("%")
        try:
            return max(0, min(100, int(round(float(cleaned)))))
        except ValueError:
            return None
    if isinstance(value, dict):
        for key in ("new", "value", "score", "level"):
            if key in value:
                return coerce_emotion_value(value[key])
    return None


def compute_emotion_deltas(current_state, updated_emotions):
    deltas = {}
    if not isinstance(updated_emotions, dict):
        return deltas

    for key, raw in updated_emotions.items():
        if key not in EMOTION_KEYS or key not in current_state:
            continue
        new_val = coerce_emotion_value(raw)
        curr_val = coerce_emotion_value(current_state.get(key))
        if new_val is not None and curr_val is not None:
            deltas[key] = new_val - curr_val
    return deltas


def _parse_retry_seconds(error_message):
    match = re.search(r"try again in ([\d.]+)s", str(error_message), re.IGNORECASE)
    if match:
        return float(match.group(1)) + 0.25
    return None


def _is_rate_limit_error(exc):
    msg = str(exc).lower()
    return "429" in msg or "rate_limit" in msg or "rate limit" in msg


def groq_chat_completion(client, *, max_retries=4, **kwargs):
    """Call Groq chat completions with automatic retry on rate limits and model defaults."""
    # Ensure current model is used
    model = kwargs.get("model")
    if not model or model in ("llama-3.1-8b-instant", "llama-3.3-70b-versatile"):
        kwargs["model"] = DEFAULT_MODEL

    # Adapt max_tokens to max_completion_tokens for openai/* and reasoning models
    current_model = kwargs.get("model", "")
    if "openai/" in current_model or "oss" in current_model:
        if "max_tokens" in kwargs and "max_completion_tokens" not in kwargs:
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")

    last_error = None
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            last_error = exc
            err_str = str(exc).lower()

            # Handle parameter error where API expects max_completion_tokens instead of max_tokens
            if "max_tokens" in err_str and "max_completion_tokens" in err_str and "max_tokens" in kwargs:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                try:
                    return client.chat.completions.create(**kwargs)
                except Exception as inner_exc:
                    last_error = inner_exc
                    err_str = str(inner_exc).lower()

            # If json_object mode fails on model, remove response_format and retry
            if "response_format" in kwargs and ("not supported" in err_str or "unsupported" in err_str or "400" in err_str):
                kwargs.pop("response_format", None)
                try:
                    return client.chat.completions.create(**kwargs)
                except Exception as inner_exc:
                    last_error = inner_exc
                    err_str = str(inner_exc).lower()

            if not _is_rate_limit_error(exc) or attempt >= max_retries - 1:
                raise
            wait = _parse_retry_seconds(exc) or (2.5 * (attempt + 1))
            print(
                f"Groq rate limit hit, retrying in {wait:.1f}s "
                f"(attempt {attempt + 1}/{max_retries})..."
            )
            time.sleep(wait)
    raise last_error


def truncate_for_analysis(text, max_chars=1200):
    if not text or len(text) <= max_chars:
        return text or ""
    return text[:max_chars] + "\n... [truncated for analysis]"
