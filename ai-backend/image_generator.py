import requests
import os
import urllib.parse
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import uuid

load_dotenv()

DEFAULT_HF_IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"

def get_api_url():
    model = os.getenv("HF_IMAGE_MODEL", DEFAULT_HF_IMAGE_MODEL)
    return f"https://router.huggingface.co/hf-inference/models/{model}"

def _generate_via_pollinations(prompt):
    """Generate high-quality FLUX image via Pollinations (no API key or credits required)."""
    encoded_prompt = urllib.parse.quote(prompt[:500])
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?model=flux&width=1024&height=768&nologo=true"
    response = requests.get(url, timeout=45)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    raise Exception(f"Pollinations FLUX error: Status {response.status_code}")

def _generate_via_huggingface(prompt):
    """Attempt HuggingFace image generation."""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise Exception("HF_TOKEN not configured")
        
    headers = {
        "Authorization": f"Bearer {hf_token}"
    }

    api_url = get_api_url()
    response = requests.post(
        api_url,
        headers=headers,
        json={"inputs": prompt},
        timeout=30
    )

    if response.status_code != 200:
        raise Exception(f"HuggingFace API error: {response.status_code} - {response.text[:200]}")

    return Image.open(BytesIO(response.content))

def generate_scene_image(prompt):
    provider = os.getenv("IMAGE_PROVIDER", "auto").lower()
    image = None

    if provider == "pollinations":
        try:
            image = _generate_via_pollinations(prompt)
        except Exception as e:
            print(f"Pollinations error: {e}")
            raise e
    else:
        # Default / Auto: Attempt HuggingFace first, seamlessly fallback to Pollinations FLUX
        try:
            image = _generate_via_huggingface(prompt)
        except Exception as hf_err:
            print(f"HuggingFace inference unavailable ({hf_err}). Seamlessly falling back to FLUX via Pollinations...")
            try:
                image = _generate_via_pollinations(prompt)
            except Exception as poll_err:
                print(f"Fallback generation also failed: {poll_err}")
                raise hf_err

    filename = f"{uuid.uuid4()}.png"
    os.makedirs("generated_images", exist_ok=True)
    save_path = f"generated_images/{filename}"
    image.save(save_path)
    return filename