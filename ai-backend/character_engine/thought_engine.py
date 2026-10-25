import os
import json
from groq import Groq
from groq_utils import groq_chat_completion, DEFAULT_MODEL

def generate_hidden_thought_prompt(
    character_brain,
    emotional_state,
    memories,
    scene
):
    return f"""
You are the internal mind of a character.

Character Brain:
{character_brain}

Current Emotional State:
{emotional_state}

Relevant Memories:
{memories}

Current Scene:
{scene}

Generate:

1. Hidden Thoughts
2. Emotional Motivation
3. What the character wants
4. What the character is hiding

Return only JSON.

Format:

{{
  "hidden_thoughts":"",
  "motivation":"",
  "secret_feeling":""
}}
"""

def get_character_hidden_thoughts(client: Groq, character_name: str, character_brain: str, emotional_state: str, memories: str, scene: str) -> dict:
    prompt = generate_hidden_thought_prompt(character_brain, emotional_state, memories, scene)
    
    try:
        completion = groq_chat_completion(
            client,
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=400,
        )
        response_text = completion.choices[0].message.content
        cleaned_text = response_text.strip()
        
        # Strip markdown if present
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()
        
        return json.loads(cleaned_text)
    except Exception as e:
        print(f"Error generating hidden thoughts for {character_name}: {e}")
        return {
            "hidden_thoughts": "Guarded and keeping feelings close.",
            "motivation": "Navigate the situation without exposing vulnerability.",
            "secret_feeling": "Reserved"
        }
