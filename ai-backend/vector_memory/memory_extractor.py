from groq import Groq
import os
from dotenv import load_dotenv
import json
from groq_utils import groq_chat_completion, DEFAULT_MODEL

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def extract_memory(scene_text):

    extraction_prompt = f"""
    You are an emotional memory extraction engine.

    Extract ONLY emotionally important memories from this scene.

    Ignore:
    - filler dialogue
    - environment descriptions
    - unimportant actions

    Focus ONLY on:
    - confessions
    - emotional shifts
    - betrayals
    - tension
    - promises
    - arguments
    - emotional vulnerability
    - relationship changes

    Return ONLY valid JSON.

    JSON format:

    {{
      "memory_type": "",
      "importance": 1-10,
      "emotion": "",
      "summary": "",
      "characters": []
    }}

    Scene:
    {scene_text}
    """

    completion = groq_chat_completion(
        client,
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": extraction_prompt}],
        response_format={"type": "json_object"},
        max_tokens=400,
    )

    response_text = completion.choices[0].message.content.strip()

    if response_text.startswith("```json"):
        response_text = response_text[7:]
    elif response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    response_text = response_text.strip()

    try:
        return json.loads(response_text)
    except Exception as e:
        print("Error decoding memory JSON:", e)
        return {
            "memory_type": "event",
            "importance": 5,
            "emotion": "neutral",
            "summary": "Characters interacted in the scene.",
            "characters": []
        }