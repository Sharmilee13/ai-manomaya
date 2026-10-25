import os
import json
from groq import Groq
from dotenv import load_dotenv
from groq_utils import groq_chat_completion, DEFAULT_MODEL
from storyboard.prompts import STORYBOARD_SYSTEM_PROMPT, STORYBOARD_USER_PROMPT_TEMPLATE
from storyboard.schemas import StoryboardResponse

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_storyboard(scene: str) -> StoryboardResponse:
    """
    Generate storyboard shots from scene screenplay text using Groq.
    """
    model_name = DEFAULT_MODEL
    user_content = STORYBOARD_USER_PROMPT_TEMPLATE.format(scene=scene)
    
    response_text = ""
    try:
        completion = groq_chat_completion(
            client,
            model=model_name,
            messages=[
                {"role": "system", "content": STORYBOARD_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"},
            max_tokens=2000,
            temperature=0.7
        )
        response_text = completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling {model_name}: {e}")
        response_text = ""

    if not response_text:
        # Build a safe default structure if both API calls fail
        return StoryboardResponse(shots=[
            {
                "shot_number": 1,
                "shot_type": "establishing",
                "duration": 4,
                "camera_angle": "wide",
                "emotion": "neutral",
                "description": f"Establishing scene view: {scene[:200]}"
            }
        ])

    # Clean code blocks wrapper if model returned them
    cleaned = response_text
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        # Validate structure with Pydantic
        return StoryboardResponse(**data)
    except Exception as e:
        print(f"Error parsing storyboard response JSON: {e}")
        try:
            # Try parsing from substring matching {...} if parsing raw failed
            start = cleaned.find('{')
            end = cleaned.rfind('}')
            if start != -1 and end != -1:
                data = json.loads(cleaned[start:end+1])
                return StoryboardResponse(**data)
        except Exception as nested_err:
            print(f"Sub-parsing also failed: {nested_err}")

        # Return a safe default structure
        return StoryboardResponse(shots=[
            {
                "shot_number": 1,
                "shot_type": "establishing",
                "duration": 4,
                "camera_angle": "wide",
                "emotion": "neutral",
                "description": f"Establishing scene view: {scene[:200]}"
            }
        ])
