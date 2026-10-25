import os
from groq import Groq
from groq_utils import groq_chat_completion, DEFAULT_MODEL

def enhance_prompt_with_llm(base_prompt):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return base_prompt
    try:
        client = Groq(api_key=api_key)
        system_prompt = (
            "You are a professional cinematographer and master AI prompt engineer. "
            "Your task is to rewrite the input visual description into a highly detailed, realistic, and cinematic prompt for FLUX. "
            "Enhance the prompt by describing: "
            "1. The environment: lighting (e.g. volumetric shafts, anamorphic lens flares, soft golden hour), atmosphere (e.g. humid air, floating dust motes), and realistic textures. "
            "2. The characters: precise postures, subtle facial expressions (e.g. jaw clenched, eyes scanning), detailed skin textures, and clothing folds. "
            "3. Composition: cinematic camera setup (e.g. 35mm lens, shallow depth of field, sharp focus, natural color grading). "
            "Keep the output as a single, highly descriptive paragraph (under 150 words) that flows naturally. "
            "Ensure the scene is extremely realistic and matches a premium film or high-end drama. "
            "Do NOT use generic buzzwords like 'photorealistic', 'hyperrealistic', '4k', or '8k'. "
            "Output ONLY the enhanced prompt paragraph, with no introductory text or markdown formatting."
        )
        
        completion = groq_chat_completion(
            client,
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Scene details to enhance:\n{base_prompt}"}
            ],
            max_tokens=250,
            temperature=0.7
        )
        enhanced = completion.choices[0].message.content.strip()
        if enhanced:
            return enhanced
    except Exception as e:
        print(f"Error in prompt enhancement: {e}")
    return base_prompt

def build_visual_prompt(
    scene,
    direction,
    characters,
    frame_index=0
):

    character_descriptions = []

    for char in characters:
        appearance = char.get('appearance') or {}
        if not isinstance(appearance, dict):
            appearance = {}

        hair = appearance.get('hair') or char.get('hair') or ""
        clothing = appearance.get('clothing') or char.get('clothing') or ""
        eyes = appearance.get('eyes') or char.get('eyes') or ""
        skin_tone = appearance.get('skinTone') or char.get('skinTone') or ""
        height = appearance.get('height') or char.get('height') or ""
        age = char.get('age') or ""
        gender = char.get('gender') or ""

        # Check for threeDConfig customized color values
        three_d = char.get('threeDConfig') or {}
        color_details = []
        if three_d.get('hairColor'):
            color_details.append(f"hair hex: {three_d.get('hairColor')}")
        if three_d.get('clothingColor'):
            color_details.append(f"clothing hex: {three_d.get('clothingColor')}")
        if three_d.get('skinColor'):
            color_details.append(f"skin tone hex: {three_d.get('skinColor')}")
        if three_d.get('faceShape'):
            color_details.append(f"face shape: {three_d.get('faceShape')}")
        if three_d.get('hairStyle'):
            color_details.append(f"hair style structure: {three_d.get('hairStyle')}")
        color_desc = f" (exact custom styling: {', '.join(color_details)})" if color_details else ""

        desc = f"{char.get('name', 'Unnamed')} (a {age} {gender}, height: {height}), hair: {hair}, eyes: {eyes}, skin tone: {skin_tone}, clothing: {clothing}{color_desc}"
        character_descriptions.append(desc)

    joined_characters = ", ".join(
        character_descriptions
    )

    # Customize visual details to create three distinct moments / compositions
    if frame_index == 0:
        framing = "Wide establishing shot, cinematic environmental composition, showing the full setting and character placement"
        focus = "Focusing on the beginning of the scene and the physical space"
    elif frame_index == 1:
        framing = "Medium shot, chest-up composition, focusing on character interaction, expressions, and posture"
        focus = "Focusing on the conversational peak in the middle of the scene"
    else:
        framing = "Close-up cinematic shot, highly focused, highlighting facial detail, micro-expressions, and deep emotion"
        focus = "Focusing on the emotional reaction, climax, or quiet closing moment of the scene"

    prompt = f"""
    Cinematic movie still,
    {framing},
    {direction.get('mood', 'neutral') if direction else 'neutral'},
    {direction.get('lighting', 'soft') if direction else 'soft'},
    {direction.get('weather', 'clear') if direction else 'clear'},
    {direction.get('camera', 'digital film camera') if direction else 'digital film camera'},
    {direction.get('color_palette', 'natural colors') if direction else 'natural colors'},

    Characters:
    {joined_characters}

    Moment Details:
    {focus}

    Scene Context:
    {scene}

    Film grain,
    Emotional atmosphere,
    Photorealistic cinematic lighting,
    Netflix drama aesthetic,
    Detailed background,
    Frame {frame_index + 1} of 3
    """

    return enhance_prompt_with_llm(prompt)