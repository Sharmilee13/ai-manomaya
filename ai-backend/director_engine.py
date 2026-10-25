from groq import Groq
from dotenv import load_dotenv
import os
import json
from groq_utils import groq_chat_completion, DEFAULT_MODEL

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def generate_direction(scene_text):

    direction_prompt = f"""
    You are an AI cinematic director.

    Analyze the scene and generate cinematic direction metadata.

    Focus on:
    - emotional mood
    - camera framing
    - lighting
    - weather
    - visual tone
    - scene intensity

    Return ONLY valid JSON.

    JSON format:

    {{
      "camera": "",
      "lighting": "",
      "weather": "",
      "mood": "",
      "color_palette": "",
      "scene_intensity": 0
    }}

    Scene:
    {scene_text}
    """

    completion = groq_chat_completion(
        client,
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": direction_prompt}],
        response_format={"type": "json_object"},
        max_tokens=300,
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
        print("Error decoding director JSON:", e)
        return {
            "camera": "medium shot",
            "lighting": "neutral lighting",
            "weather": "clear",
            "mood": "neutral",
            "color_palette": "natural tones",
            "scene_intensity": 50
        }