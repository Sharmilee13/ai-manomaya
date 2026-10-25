from groq import Groq
import os
from dotenv import load_dotenv
import json
from groq_utils import groq_chat_completion, DEFAULT_MODEL

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

story_bibles = {}

DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "story_bibles.json")

def load_story_bibles():
    global story_bibles
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                story_bibles = json.load(f)
        except Exception as e:
            print(f"Error loading story bibles: {e}")
            story_bibles = {}
    else:
        story_bibles = {}
    return story_bibles

def save_story_bibles():
    try:
        os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(story_bibles, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving story bibles: {e}")


_PRIORITY_KEYS = (
    "event", "title", "summary", "thread", "description", "text",
    "outcome", "impact", "status", "emotion", "tension", "change",
    "arc", "stakes", "resolution", "importance", "characters", "character",
    "detail", "note", "relationship",
)


def _normalize_bible_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                return _normalize_bible_text(json.loads(stripped))
            except Exception:
                pass
        return stripped
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [_normalize_bible_text(item) for item in value]
        return " · ".join(p for p in parts if p)
    if isinstance(value, dict):
        parts = []
        seen = set()
        for key in _PRIORITY_KEYS:
            if key in value and value[key] is not None and value[key] != "":
                if isinstance(value[key], dict):
                    nested = _normalize_bible_text(value[key])
                    if nested:
                        parts.append(nested)
                else:
                    parts.append(str(value[key]).strip())
                seen.add(key)
        for key, val in value.items():
            if key in seen or key in ("id", "type", "index"):
                continue
            if val is None or val == "":
                continue
            if isinstance(val, dict):
                nested = _normalize_bible_text(val)
                if nested:
                    parts.append(nested)
            else:
                parts.append(str(val).strip())
        return " — ".join(p for p in parts if p)
    return str(value).strip()


def _normalize_bible_list(items):
    seen = set()
    out = []
    for item in items or []:
        text = _normalize_bible_text(item)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def create_story_bible(project_id):
    load_story_bibles()
    if project_id not in story_bibles:
        story_bibles[project_id] = {
            "important_events": [],
            "active_story_threads": [],
            "character_summaries": {},
            "relationship_summaries": {},
            "world_summary": "",
        }
        save_story_bibles()
    return story_bibles[project_id]


def _sanitize_story_bible(bible):
    bible["important_events"] = _normalize_bible_list(bible.get("important_events"))
    bible["active_story_threads"] = _normalize_bible_list(bible.get("active_story_threads"))

    char = {}
    for name, summary in (bible.get("character_summaries") or {}).items():
        key = _normalize_bible_text(name) or str(name)
        text = _normalize_bible_text(summary)
        if key and text:
            char[key] = text
    bible["character_summaries"] = char

    rel = {}
    for pair, summary in (bible.get("relationship_summaries") or {}).items():
        key = _normalize_bible_text(pair) or str(pair)
        text = _normalize_bible_text(summary)
        if key and text:
            rel[key] = text
    bible["relationship_summaries"] = rel

    bible["world_summary"] = _normalize_bible_text(bible.get("world_summary"))
    return bible


def get_story_bible(project_id):
    load_story_bibles()
    bible = create_story_bible(project_id)
    sanitized = _sanitize_story_bible(bible)
    story_bibles[project_id] = sanitized
    save_story_bibles()
    return sanitized


def update_story_bible(project_id, bible_update):
    bible = create_story_bible(project_id)

    for key, value in bible_update.items():
        if key in bible:
            bible[key] = value

    save_story_bibles()
    return bible


def merge_story_bible_analysis(project_id, analysis):
    """Merge scene analysis into the bible without dropping prior events."""
    bible = create_story_bible(project_id)

    for event in _normalize_bible_list(analysis.get("important_events")):
        if event not in bible["important_events"]:
            bible["important_events"].append(event)

    threads = _normalize_bible_list(analysis.get("active_story_threads"))
    if threads:
        bible["active_story_threads"] = threads

    char_summaries = analysis.get("character_summaries") or {}
    if isinstance(char_summaries, dict):
        for name, summary in char_summaries.items():
            key = _normalize_bible_text(name) or str(name)
            text = _normalize_bible_text(summary)
            if key and text:
                bible["character_summaries"][key] = text

    rel_summaries = analysis.get("relationship_summaries") or {}
    if isinstance(rel_summaries, dict):
        for pair, summary in rel_summaries.items():
            key = _normalize_bible_text(pair) or str(pair)
            text = _normalize_bible_text(summary)
            if key and text:
                bible["relationship_summaries"][key] = text

    world = _normalize_bible_text(analysis.get("world_summary"))
    if world:
        bible["world_summary"] = world

    save_story_bibles()
    return bible


def format_story_bible_for_prompt(project_id):
    bible = get_story_bible(project_id)

    events = bible.get("important_events") or []
    threads = bible.get("active_story_threads") or []
    char_summaries = bible.get("character_summaries") or {}
    rel_summaries = bible.get("relationship_summaries") or {}
    world = bible.get("world_summary") or ""

    if not any([events, threads, char_summaries, rel_summaries, world]):
        return ""

    events_str = "\n".join(f"- {_normalize_bible_text(e)}" for e in events) if events else "- None yet"
    threads_str = "\n".join(f"- {_normalize_bible_text(t)}" for t in threads) if threads else "- None yet"
    char_str = "\n".join(
        f"- {_normalize_bible_text(name)}: {_normalize_bible_text(summary)}"
        for name, summary in char_summaries.items()
    ) if char_summaries else "- None yet"
    rel_str = "\n".join(
        f"- {_normalize_bible_text(pair)}: {_normalize_bible_text(summary)}"
        for pair, summary in rel_summaries.items()
    ) if rel_summaries else "- None yet"

    return f"""
========================
STORY BIBLE (NARRATIVE DIRECTION)
========================
What matters in this story — not raw memory logs.

Important Events:
{events_str}

Active Story Threads:
{threads_str}

Character Evolution:
{char_str}

Relationship Summaries:
{rel_str}

World Summary:
{world or "Not established yet."}
"""


def _parse_json_response(response_text):
    cleaned = response_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def analyze_story_impact(scene_text):
    analysis_prompt = f"""
You are a narrative analyst.

Analyze this scene and determine:

1. Important Events
2. Active Story Threads
3. Character Changes
4. Relationship Changes
5. Long-term Story Impact

Return ONLY JSON.

Use plain strings in arrays (not nested objects). Example event:
"Ryan confessed to Aisha — romantic tension is unresolved"

Format:

{{
    "important_events": ["string", "string"],
    "active_story_threads": ["string", "string"],
    "character_summaries": {{ "Name": "one sentence evolution" }},
    "relationship_summaries": {{ "NameA->NameB": "one sentence dynamic" }},
    "world_summary": "one paragraph string"
}}

Scene:

{scene_text}
"""

    completion = groq_chat_completion(
        client,
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": analysis_prompt}],
        response_format={"type": "json_object"},
        max_tokens=500,
    )

    response_text = _parse_json_response(completion.choices[0].message.content)

    try:
        return json.loads(response_text)
    except Exception as e:
        print("Error decoding story bible JSON:", e)
        return {
            "important_events": [],
            "active_story_threads": [],
            "character_summaries": {},
            "relationship_summaries": {},
            "world_summary": "",
        }
