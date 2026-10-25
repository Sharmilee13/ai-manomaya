from groq import Groq
import os
import json
from groq_utils import groq_chat_completion, DEFAULT_MODEL

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def _parse_json_response(response_text):
    cleaned = response_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()

def analyze_arc_progress(scene_text, character):
    char_name = character.get("name", "Unnamed")
    
    prompt = f"""
You are a character development analyst.
Analyze how this scene affects the character's personal growth and character arc.

Character Profile:
- Name: {char_name}
- Age: {character.get("age")}
- Core Traits: {character.get("core_traits")}
- Fears: {character.get("fears")}
- Goals: {character.get("goals")}

Scene screen-play:
{scene_text}

Determine the character's progression:
1. Current State: How they behave or feel emotionally in this scene.
2. Growth Direction: What they are learning or moving towards.
3. Internal Conflict: What internal struggle they are facing.
4. Arc Stage: Select from "beginning", "middle", "climax", "resolution".
5. Progress: A percentage integer (0-100) representing their progress along their character arc (starting at starting state towards growth direction).

Return ONLY a valid JSON object matching the keys exactly:
{{
  "current_state": "one sentence state description",
  "growth_direction": "one sentence growth label",
  "current_conflict": "one sentence internal struggle description",
  "arc_stage": "beginning/middle/climax/resolution",
  "arc_progress": 42
}}
"""

    try:
        completion = groq_chat_completion(
            client,
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=300
        )
        
        response_text = completion.choices[0].message.content.strip()
        cleaned = _parse_json_response(response_text)
        return json.loads(cleaned)
    except Exception as e:
        print("Error parsing arc analysis:", e)
        return {
            "current_state": "Emotionally guarded",
            "growth_direction": "Becoming emotionally open",
            "current_conflict": "Fear of abandonment",
            "arc_stage": "beginning",
            "arc_progress": 10
        }
