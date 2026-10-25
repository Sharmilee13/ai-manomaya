from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import base64
from groq import Groq
from dotenv import load_dotenv
import os
import json
from memory.memory_store import add_memory, get_memories
from vector_memory.memory_engine import store_memory, retrieve_memories
from data.relationships import (
    get_relationship_state,
    update_relationship_state
)
from vector_memory.memory_extractor import (
    extract_memory
)
from story_bible.story_bible_engine import (
    create_story_bible,
    get_story_bible,
    analyze_story_impact,
    merge_story_bible_analysis,
    format_story_bible_for_prompt,
)
from director_engine import generate_direction
from image_generator import generate_scene_image
from storyboard.schemas import StoryboardResponse
from storyboard.services import generate_storyboard

from visual_prompt_engine import (
    build_visual_prompt
)
from prompt_builders.character_brain import build_character_brain
from character_engine.thought_engine import get_character_hidden_thoughts
from data.emotions import get_emotion_state
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import time
from groq_utils import compute_emotion_deltas, groq_chat_completion, truncate_for_analysis, DEFAULT_MODEL

POST_SCENE_LLM_DELAY_SEC = 2.5

class SceneRequest(BaseModel):
    scene: str
    characters: Optional[List[Dict[str, Any]]] = []
    character: Optional[Dict[str, Any]] = None
    tone: Optional[str] = "neutral"
    past_memories: Optional[str] = None
    relationships: Optional[Dict[str, Any]] = {}
    project_id: Optional[str] = "default_project"

load_dotenv()

app = FastAPI()

# Ensure generated_images directory exists
os.makedirs("generated_images", exist_ok=True)

# Mount the static directory for generated images
app.mount(
    "/generated_images",
    StaticFiles(directory="generated_images"),
    name="generated_images"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

@app.get("/")
def home():
    return {
        "message": "Manomaya AI Backend Running"
    }


@app.get("/story-bible")
def get_story_bible_endpoint(project_id: Optional[str] = None, projectId: Optional[str] = None):
    pid = project_id or projectId or "default_project"
    return get_story_bible(pid)

class RebuildBibleRequest(BaseModel):
    project_id: str
    characters: List[Dict[str, Any]]
    scenes: List[Dict[str, Any]]

@app.post("/rebuild-story-bible")
def rebuild_story_bible_endpoint(request_data: RebuildBibleRequest):
    project_id = request_data.project_id
    characters = request_data.characters
    scenes = request_data.scenes

    # 1. Rebuild relationship states from scenes' emotion_deltas
    from data.relationships import get_relationship_state, relationship_key, relationship_states, save_relationships
    # Clear existing relationship states for this project first to avoid accumulation anomalies
    if project_id in relationship_states:
        relationship_states[project_id] = {}
    
    # Process scenes chronologically to rebuild relationship state
    for scene in scenes:
        deltas = scene.get("emotion_deltas") or {}
        for pair_key, delta_obj in deltas.items():
            parts = pair_key.split("->")
            if len(parts) == 2:
                c1, c2 = parts[0].strip(), parts[1].strip()
                parsed_updates = {}
                for em, val in delta_obj.items():
                    if isinstance(val, dict):
                        new_val = val.get("new")
                        if new_val is not None:
                            parsed_updates[em] = new_val
                    else:
                        parsed_updates[em] = val
                
                state = get_relationship_state(c1, c2, project_id=project_id)
                for em, val in parsed_updates.items():
                    if em in state:
                        state[em] = val
                
                if project_id not in relationship_states:
                    relationship_states[project_id] = {}
                relationship_states[project_id][relationship_key(c1, c2)] = state
    save_relationships()

    # 2. Rebuild the Story Bible via Groq
    bible_data = {
        "important_events": [],
        "active_story_threads": [],
        "character_summaries": {},
        "relationship_summaries": {},
        "world_summary": ""
    }
    if scenes:
        story_texts = []
        for idx, s in enumerate(scenes):
            title = s.get("title") or f"Scene {idx+1}"
            text = s.get("generated_text") or s.get("prompt") or ""
            # Truncate text to avoid huge token count
            if len(text) > 1000:
                text = text[:1000] + "... [truncated]"
            story_texts.append(f"Scene {idx+1}: {title}\nContent:\n{text}")
        full_story = "\n\n=======================\n\n".join(story_texts)
        
        prompt = f"""
        You are a narrative analyst.
        Below is the sequence of scenes for a story generated so far.
        Analyze the full sequence of events and extract:
        1. Important Events: Pivotal turning points in the narrative. Keep descriptions short and punchy.
        2. Active Story Threads: Unresolved plot lines, mysteries, or conflicts in play.
        3. Character Evolution: A summary of each character's growth/changes (one sentence each).
        4. Relationship Dynamics: A summary of the dynamic between active character pairs (one sentence each, key format: "CharA->CharB").
        5. World Summary: A concise description of the setting and overall narrative vibe.

        Return ONLY a valid JSON object matching this structure:
        {{
            "important_events": ["string", "string"],
            "active_story_threads": ["string", "string"],
            "character_summaries": {{ "Name": "one sentence evolution" }},
            "relationship_summaries": {{ "NameA->NameB": "one sentence dynamic" }},
            "world_summary": "one paragraph string"
        }}

        Story Scenes:
        {full_story}
        """
        
        try:
            completion = groq_chat_completion(
                client,
                model=DEFAULT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=800,
            )
            response_text = completion.choices[0].message.content.strip()
            cleaned = _parse_json_response(response_text)
            bible_data = json.loads(cleaned)
        except Exception as e:
            print(f"Error rebuilding story bible: {e}")
            
    # Save the rebuilt story bible
    from story_bible.story_bible_engine import update_story_bible
    update_story_bible(project_id, bible_data)

    # 3. Rebuild Character Arcs via Groq for each character
    from data.character_arcs import character_arcs, save_character_arcs
    if project_id not in character_arcs:
        character_arcs[project_id] = {}
        
    for char in characters:
        char_name = char.get("name")
        if not char_name:
            continue
            
        # Filter scenes featuring this character
        char_scenes = []
        for idx, s in enumerate(scenes):
            char_id = char.get("id") or char.get("_id")
            char_in_scene = False
            # Check if matching characterId exists in characterIds list
            # characterIds could contain string IDs or ObjectIds or represent strings
            scene_char_ids = s.get("characterIds", []) or s.get("character_ids", [])
            # Also normalize to strings
            scene_char_ids_str = [str(cid) for cid in scene_char_ids]
            
            if char_id and str(char_id) in scene_char_ids_str:
                char_in_scene = True
            elif char_name.lower() in (s.get("generated_text") or "").lower() or char_name.lower() in (s.get("prompt") or "").lower():
                char_in_scene = True
                
            if char_in_scene:
                char_scenes.append((idx + 1, s.get("title") or f"Scene {idx+1}", s.get("generated_text") or s.get("prompt") or ""))
                
        if not char_scenes:
            character_arcs[project_id][char_name] = {
                "starting_state": "Emotionally guarded",
                "current_state": "Emotionally guarded",
                "growth_direction": "Becoming emotionally open",
                "current_conflict": "Fear of abandonment",
                "arc_stage": "beginning",
                "arc_progress": 0,
                "history": []
            }
            continue
            
        char_scenes_context = ""
        for s_num, s_title, s_text in char_scenes:
            if len(s_text) > 800:
                s_text = s_text[:800] + "... [truncated]"
            char_scenes_context += f"Scene {s_num}: {s_title}\nContent:\n{s_text}\n\n"
            
        arc_prompt = f"""
        You are a character arc analyzer.
        Analyze this character's profile and their journey across the sequence of scenes they participated in.
        
        Character Profile:
        - Name: {char_name}
        - Core Traits: {char.get("core_traits")}
        - Strengths: {char.get("strengths")}
        - Flaws: {char.get("flaws")}
        - Fears: {char.get("fears")}
        - Goals: {char.get("goals")}
        - Values: {char.get("values")}

        Story Scenes featuring {char_name}:
        {char_scenes_context}

        Your task:
        1. Determine their starting state (at the beginning of the story).
        2. Determine their current state (after the last scene).
        3. Determine their growth direction and current conflict.
        4. Determine their arc stage (choose strictly one of: "beginning", "middle", "climax", "resolution") and current progress percentage (0 to 100).
        5. Generate a chronological history of their state changes for each scene they were in. Each history entry must have the scene index (integer), the scene title (string), the progress percentage reached (integer), and the character's emotional/behavioral state in that scene (string).

        Return ONLY a valid JSON object matching this structure:
        {{
            "starting_state": "string",
            "current_state": "string",
            "growth_direction": "string",
            "current_conflict": "string",
            "arc_stage": "beginning/middle/climax/resolution",
            "arc_progress": integer,
            "history": [
                {{
                    "scene": integer,
                    "title": "string",
                    "progress": integer,
                    "state": "string"
                }}
            ]
        }}
        """
        
        try:
            completion = groq_chat_completion(
                client,
                model=DEFAULT_MODEL,
                messages=[{"role": "user", "content": arc_prompt}],
                response_format={"type": "json_object"},
                max_tokens=600,
            )
            response_text = completion.choices[0].message.content.strip()
            cleaned = _parse_json_response(response_text)
            arc_data = json.loads(cleaned)
            character_arcs[project_id][char_name] = arc_data
        except Exception as e:
            print(f"Error rebuilding character arc for {char_name}: {e}")
            character_arcs[project_id][char_name] = {
                "starting_state": "Emotionally guarded",
                "current_state": "Emotionally guarded",
                "growth_direction": "Becoming emotionally open",
                "current_conflict": "Fear of abandonment",
                "arc_stage": "beginning",
                "arc_progress": 0,
                "history": []
            }
    save_character_arcs()

    return {
        "status": "success",
        "story_bible": bible_data,
        "character_arcs": character_arcs[project_id]
    }


class StoryboardRequest(BaseModel):
    scene: str

@app.post("/generate-storyboard", response_model=StoryboardResponse)
def generate_storyboard_endpoint(request_data: StoryboardRequest):
    return generate_storyboard(request_data.scene)

@app.post("/chat")
def chat(prompt: str):

    completion = groq_chat_completion(
        client,
        model=DEFAULT_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return {
        "response": completion.choices[0].message.content
    }

@app.post("/generate-scene")
def generate_scene(request_data: SceneRequest):
    data = request_data.dict()
    characters = data.get("characters", [])
    if not characters and data.get("character"):
        characters = [data.get("character")]
        
    scene = data.get("scene", "")
    tone = data.get("tone", "neutral")
    project_id = data.get("project_id") or "default_project"

    story_bible_prompt = format_story_bible_for_prompt(project_id)
    
    # Read past memories sent from Next.js (isolated db memory), fallback to local memory_store
    past_memories = data.get("past_memories")
    if not past_memories:
        past_memories = retrieve_memories(
            project_id=project_id,
            current_scene=scene
        )
    
    # Truncate past_memories to prevent exceeding Groq TPM (tokens per minute) limit
    if past_memories and len(past_memories) > 2000:
        past_memories = past_memories[:2000] + "\n... [Truncated for brevity to fit rate limits]"
    
    # Build/fetch directional relationships between all pairs of characters in the scene
    relationships = data.get("relationships", {})
    if not relationships and len(characters) > 1:
        relationships = {}
        for i in range(len(characters)):
            for j in range(len(characters)):
                if i != j:
                    c1 = characters[i].get("name")
                    c2 = characters[j].get("name")
                    if c1 and c2:
                        relationships[f"{c1}->{c2}"] = get_relationship_state(c1, c2, project_id=project_id)
 
    relationships_str = ""
    if relationships:
        relationships_str = "Current Relationship Dynamics (directional emotions):\n"
        for key, ems in relationships.items():
            relationships_str += f"- {key} (how the first character feels towards the second): Trust: {ems.get('trust')}%, Attachment: {ems.get('attachment')}%, Awkwardness: {ems.get('awkwardness')}%, Resentment: {ems.get('resentment')}%, Comfort: {ems.get('comfort')}%\n"
        relationships_str += "\n"

    character_brains_list = []
    hidden_thoughts_dict = {}
    hidden_thoughts_prompt_list = []

    for char in characters:
        char_name = char.get("name", "Unnamed")
        brain_str = build_character_brain(char)
        character_brains_list.append(brain_str)
        
        # Build emotional state for the character
        indiv_emotions = get_emotion_state(char_name, project_id=project_id)
        char_relationships = {}
        for other_char in characters:
            other_name = other_char.get("name")
            if other_name and other_name != char_name:
                char_relationships[f"{char_name}->{other_name}"] = get_relationship_state(char_name, other_name, project_id=project_id)
        
        emotional_state_str = f"Individual Emotions: {indiv_emotions}\n"
        if char_relationships:
            emotional_state_str += "Relationship Dynamics with other characters in the scene:\n"
            for key, val in char_relationships.items():
                emotional_state_str += f"- {key}: {val}\n"
        
        # Generate hidden thoughts for this character
        thoughts = get_character_hidden_thoughts(
            client=client,
            character_name=char_name,
            character_brain=brain_str,
            emotional_state=emotional_state_str,
            memories=past_memories,
            scene=scene
        )
        hidden_thoughts_dict[char_name] = thoughts
        
        # Format the thoughts for prompt injection
        hidden_thoughts_prompt_list.append(
            f"Character: {char_name}\n"
            f"- Hidden Thoughts: \"{thoughts.get('hidden_thoughts', '')}\"\n"
            f"- Motivation: {thoughts.get('motivation', '')}\n"
            f"- Secret Feeling: {thoughts.get('secret_feeling', '')}"
        )

    character_brain_str = "\n".join(character_brains_list)
    hidden_thoughts_prompt_str = "\n\n".join(hidden_thoughts_prompt_list)

    character_arcs_list = []
    from data.character_arcs import get_character_arc
    for char in characters:
        char_name = char.get("name", "Unnamed")
        arc = get_character_arc(char_name, project_id=project_id)
        arc_str = (
            f"Character: {char_name}\n"
            f"- Starting State: {arc.get('starting_state')}\n"
            f"- Current State: {arc.get('current_state')}\n"
            f"- Growth Direction: {arc.get('growth_direction')}\n"
            f"- Current Conflict: {arc.get('current_conflict')}\n"
            f"- Arc Stage: {arc.get('arc_stage')}\n"
            f"- Arc Progress: {arc.get('arc_progress')}%"
        )
        character_arcs_list.append(arc_str)
    character_arcs_str = "\n\n".join(character_arcs_list)

    system_prompt = f"""
You are Manomaya, a modern, cinematic storytelling AI.
Your purpose is to generate highly realistic, emotionally grounded, and immersive scenes that feel like a premium Netflix drama or indie film.

========================
ASSIGNED CHARACTERS (CRITICAL)
========================
You must ONLY use the characters listed below. NEVER introduce new characters automatically.

Assigned Character Brain:

{character_brain_str}

========================
CHARACTER ARCS & GROWTH JOURNEY (VITAL CONTEXT)
========================
These fields represent each character's current stage of development, starting points, and conflicts. Characters must react and speak in a way that respects their current progress along their growth arcs (e.g. beginning stage is much more defensive than resolution stage).

{character_arcs_str}

========================
CHARACTERS' HIDDEN THOUGHTS (INTERNAL STATES FOR THIS SCENE)
========================
These are the private internal thoughts, motivations, and secret feelings of the characters at the start of this scene. 
Use them to guide each character's dialogue subtext, pauses, micro-expressions, and actions. 
CRITICAL: Do NOT print these hidden thoughts verbatim in the screenplay text. They are private internal states and should influence their subtext and behavioral tension without being spoken directly.

{hidden_thoughts_prompt_str}

========================
SCENE PARAMETERS
========================
Target Emotional Tone: {tone}

========================
CURRENT DIRECTIONAL RELATIONSHIP DYNAMICS
========================
{relationships_str}

========================
MEMORY & PAST EVENTS
========================
{past_memories}
{story_bible_prompt}
========================
CURRENT SCENE PROMPT
========================
{scene}

========================
YOUR TASK & CRITICAL RULES
========================
1. Write the scene natively using the tone: '{tone}'.
2. The scene must ONLY involve the assigned characters listed above. NEVER introduce new characters.
3. The writing must be modern, grounded, cinematic, emotionally realistic, subtle, and immersive (like a Netflix emotional drama or indie film).
4. Dialogue should feel human and natural (use short realistic sentences, pauses, realistic reactions).
5. DO NOT use fantasy writing, Shakespearean prose, fanfiction tropes, exaggerated poetry, melodrama, or anime exposition.
6. Enhance the environment cinematically (weather, lighting, sounds, atmosphere) but keep it simple and grounded.
7. REALISTIC EMOTIONAL LIFELIKE SHIFTS (CRITICAL):
   - Real-world relationship dynamics change slowly, subtly, and unevenly.
   - Do NOT make massive jumps in scores (e.g., trust going from 50% to 80% in one brief talk is highly unrealistic). Keep typical shifts incremental (usually 1% to 8% max per scene).
   - Reflect complex, non-ideal human psychology:
     - Characters can get closer (attachment increases) but simultaneously feel more vulnerable or awkward (awkwardness might also increase).
     - Emotion updates must be asymmetric. If character A feels a surge of trust or attachment, character B might remain guarded, feel pressured, or even feel increased resentment. 
     - Do not mirror emotions perfectly between the characters. Each direction of the relationship key must be updated independently based on how that specific character perceived the interaction.
8. Before generating dialogue, consider:
   1. What does this character WANT?
   2. What does this character FEAR?
   3. What emotional state are they in?
   4. What memories affect this scene?
   5. What would this specific character realistically say?
   6. How does their current Character Arc Stage and Progress affect their behavior? A character in 'beginning' stage (low progress) remains defensive and closed-off, while a character in 'middle' or 'climax' (higher progress) starts to show vulnerability.
9. A BALANCE OF DIALOGUE AND NARRATION (CRITICAL):
   - The generated scene MUST contain a rich balance of both spoken dialogue (formatted clearly with character names, e.g., 'CharacterName: "..."') and cinematic visual action description/narration.
   - Do NOT write only narration/prose or only dialogue. The characters must actively speak to each other.


**OUTPUT FORMAT:**
You must return your response as a valid JSON object with EXACTLY two keys:
1. "scene": The full text of the generated scene.
2. "updated_emotions": A JSON object containing updated 0-100 values for trust, attachment, awkwardness, resentment, and comfort for each directional relationship affected by this scene. Use the format "CharacterA_Name->CharacterB_Name" as the keys. Only include pair directions that are actually affected or actively interact in the scene.

Example JSON:
{{
  "scene": "The text of the scene goes here...",
  "updated_emotions": {{
    "Elena->Aisha": {{
      "trust": 55,
      "attachment": 50,
      "awkwardness": 10,
      "resentment": 5,
      "comfort": 45
    }},
    "Aisha->Elena": {{
      "trust": 48,
      "attachment": 52,
      "awkwardness": 12,
      "resentment": 6,
      "comfort": 40
    }}
  }}
}}
"""

    import json
    
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"Generate the scene screenplay based on the prompt: '{scene}'. Respond with a JSON object containing the 'scene' and 'updated_emotions' keys."
        }
    ]

    # Try JSON mode first, fall back to plain text if Groq rejects it
    try:
        completion = groq_chat_completion(
            client,
            model=DEFAULT_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=2500
        )
        response_text = completion.choices[0].message.content
    except Exception as json_err:
        print(f"JSON mode failed, retrying without JSON mode: {json_err}")
        completion = groq_chat_completion(
            client,
            model=DEFAULT_MODEL,
            messages=messages,
            max_tokens=2500
        )
        response_text = completion.choices[0].message.content
    
    print("\n" + "="*50)
    print("RAW RESPONSE FROM GROQ:")
    print(response_text)
    print("="*50 + "\n")
    
    # Robust JSON parsing function with fallbacks
    def safe_parse_json(text):
        import re
        cleaned = text.strip()
        # Strip reasoning tags if present
        cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL).strip()
        try:
            return json.loads(cleaned)
        except Exception:
            pass
            
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        try:
            return json.loads(cleaned)
        except Exception:
            pass

        try:
            start = cleaned.find('{')
            end = cleaned.rfind('}')
            if start != -1 and end != -1:
                return json.loads(cleaned[start:end+1])
        except Exception:
            pass
        return None

    response_data = safe_parse_json(response_text)
    generated_scene = ""
    updated_emotions = None

    # Handle cases where Groq returns a list, string, or malformed data
    if isinstance(response_data, dict):
        generated_scene = response_data.get("scene", "")
        updated_emotions = response_data.get("updated_emotions", None)
    elif isinstance(response_data, list) and len(response_data) > 0 and isinstance(response_data[0], dict):
        generated_scene = response_data[0].get("scene", "")
        updated_emotions = response_data[0].get("updated_emotions", None)

    # Unwrap if generated_scene itself was serialized as a JSON string containing "scene"
    if isinstance(generated_scene, str) and (generated_scene.strip().startswith("{") or '"scene"' in generated_scene):
        try:
            inner = json.loads(generated_scene)
            if isinstance(inner, dict) and "scene" in inner:
                generated_scene = inner.get("scene", generated_scene)
                if not updated_emotions and "updated_emotions" in inner:
                    updated_emotions = inner.get("updated_emotions")
        except Exception:
            pass

    # If parsing failed or couldn't get a valid scene string, manually extract to prevent leak
    if not generated_scene:
        import re
        # Check if the response contains json keys and extract manually
        if '"scene"\s*:' in response_text or "'scene'\s*:" in response_text or "updated_emotions" in response_text:
            try:
                # Extract content inside "scene": "..."
                scene_match = re.search(r'"scene"\s*:\s*"(.*?)(?<!\\)"', response_text, re.DOTALL)
                if scene_match:
                    generated_scene = scene_match.group(1).replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
                else:
                    # Split at updated_emotions to isolate the scene block
                    parts = re.split(r'"updated_emotions"\s*:', response_text, flags=re.IGNORECASE)
                    if len(parts) > 0:
                        scene_part = parts[0]
                        # Remove json wrappers
                        scene_part = re.sub(r'^\s*\{\s*"scene"\s*:\s*"?', '', scene_part, flags=re.IGNORECASE)
                        scene_part = re.sub(r'"?\s*,\s*$', '', scene_part)
                        generated_scene = scene_part.strip().replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
            except Exception as e:
                print("Failed to manually extract scene text:", e)

    # Final fallback if all extraction methods fail
    if not generated_scene:
        generated_scene = response_text

    scene_for_analysis = truncate_for_analysis(generated_scene)

    # --- Memory extraction (non-blocking) ---
    time.sleep(POST_SCENE_LLM_DELAY_SEC)
    try:
        memory_data = extract_memory(scene_for_analysis)
        print("\n" + "="*50)
        print("EXTRACTED EMOTIONAL MEMORY OBJECT:")
        print(json.dumps(memory_data, indent=2))
        print("="*50 + "\n")
        store_memory(
            project_id=project_id,
            memory_data=memory_data
        )
        add_memory(scene)
    except Exception as e:
        print(f"Error in memory extraction/storage: {e}")

    # --- Story bible analysis (non-blocking) ---
    time.sleep(POST_SCENE_LLM_DELAY_SEC)
    story_bible = None
    try:
        bible_analysis = analyze_story_impact(scene_for_analysis)
        print("\n" + "=" * 50)
        print("STORY BIBLE ANALYSIS:")
        print(json.dumps(bible_analysis, indent=2))
        print("=" * 50 + "\n")
        story_bible = merge_story_bible_analysis(project_id, bible_analysis)
    except Exception as e:
        print(f"Error in story bible analysis: {e}")
        story_bible = get_story_bible(project_id)

    # --- Relationship updates (non-blocking) ---
    try:
        if updated_emotions and isinstance(updated_emotions, dict):
            for key, ems in updated_emotions.items():
                parts = key.split("->")
                if len(parts) == 2:
                    c1, c2 = parts[0].strip(), parts[1].strip()
                    curr = get_relationship_state(c1, c2, project_id=project_id)
                    deltas = {}
                    for k, v in ems.items():
                        if k in curr:
                            deltas[k] = v - curr[k]
                    update_relationship_state(c1, c2, deltas, project_id=project_id)
    except Exception as e:
        print(f"Error updating relationships: {e}")

    # --- Character Arc Updates (non-blocking) ---
    try:
        from character_engine.arc_analyzer import analyze_arc_progress
        from data.character_arcs import update_character_arc, get_character_arc
        for char in characters:
            char_name = char.get("name")
            if char_name:
                arc_update = analyze_arc_progress(scene_for_analysis, char)
                arc = get_character_arc(char_name, project_id=project_id)
                scene_num = len(arc.get("history", [])) + 1
                scene_title = f"Scene {scene_num}"
                update_character_arc(char_name, arc_update, scene_title, scene_num, project_id=project_id)
    except Exception as e:
        print(f"Error updating character arcs: {e}")

    # --- Director engine (non-blocking) ---
    time.sleep(POST_SCENE_LLM_DELAY_SEC)
    direction_data = None
    try:
        direction_data = generate_direction(scene_for_analysis)
    except Exception as e:
        print(f"Error generating direction: {e}")

    # NOTE: Image generation has been moved to a separate on-demand endpoint /generate-scene-images

    return {
        "scene": generated_scene,
        "updated_emotions": updated_emotions,
        "direction": direction_data,
        "image": None,
        "images": [],
        "hidden_thoughts": hidden_thoughts_dict,
        "story_bible": story_bible,
    }

class ImageGenerationRequest(BaseModel):
    scene_text: str
    direction: Optional[Dict[str, Any]] = None
    characters: Optional[List[Dict[str, Any]]] = []
    num_frames: Optional[int] = 3

@app.post("/generate-scene-images")
def generate_scene_images(request_data: ImageGenerationRequest):
    """Standalone endpoint to generate storyboard images for an existing scene."""
    scene_text = request_data.scene_text
    direction_data = request_data.direction
    characters = request_data.characters or []
    num_frames = min(request_data.num_frames or 3, 5)

    if not scene_text:
        return {"images": [], "error": "No scene text provided"}

    # If no direction data provided, generate it on the fly
    if not direction_data:
        try:
            direction_data = generate_direction(scene_text[:2000])
        except Exception as e:
            print(f"Error generating direction for images: {e}")
            direction_data = {
                "camera": "wide shot",
                "lighting": "natural",
                "weather": "clear",
                "mood": "neutral",
                "color_palette": "muted tones",
                "scene_intensity": 50
            }

    image_filenames = []
    try:
        from concurrent.futures import ThreadPoolExecutor

        def gen_frame(i):
            try:
                visual_prompt = build_visual_prompt(
                    scene_text,
                    direction_data,
                    characters,
                    frame_index=i
                )
                return generate_scene_image(visual_prompt)
            except Exception as img_err:
                print(f"Error generating image {i+1}: {img_err}")
                return None

        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(gen_frame, range(num_frames)))
            image_filenames = [img for img in results if img]
    except Exception as e:
        print(f"Error in image generation: {e}")

    return {
        "images": image_filenames,
        "image": image_filenames[0] if image_filenames else None,
    }

def generate_progressive_prompts(base_prompt, client=None):
    if not client:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return [
                f"{base_prompt}, step 1: starting action, character begins action sequence",
                f"{base_prompt}, step 2: mid-point action, character gestures, dynamic movement",
                f"{base_prompt}, step 3: ending action, character completes action, resolving pose"
            ]
        client = Groq(api_key=api_key)
        
    system_prompt = (
        "You are a professional film director and master AI prompt engineer. "
        "Given a single visual prompt for a cinematic movie still, generate a sequence of exactly 3 progressive "
        "visual prompts that describe consecutive moments in a continuous 10-second video shot. "
        "The setting, environment details, color palette, camera lens, and character physical features (face shape, age, "
        "clothing, hair, eyes) must remain completely identical. "
        "Only describe a progression of subtle actions, hand gestures, facial expression changes, or body movements "
        "that tell a micro-story (e.g. step 1: character starts speaking, eyes narrow; step 2: character shifts weight, reaches out slightly with their hand; step 3: character finishes speaking, looks down with a quiet expression). "
        "Ensure each prompt is highly detailed and cinematic for FLUX, using photographic descriptions instead of buzzwords like 'photorealistic'. "
        "Return a JSON object with exactly one key 'prompts' containing a list of exactly 3 strings (each representing a step)."
    )
    
    try:
        completion = groq_chat_completion(
            client,
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Cinematic base prompt: {base_prompt}"}
            ],
            response_format={"type": "json_object"},
            max_tokens=600,
            temperature=0.7
        )
        data = json.loads(completion.choices[0].message.content)
        prompts = data.get("prompts", [])
        if isinstance(prompts, list) and len(prompts) == 3:
            return prompts
    except Exception as e:
        print(f"Error generating progressive prompts: {e}")
        
    # Fallback to simple appending
    return [
        f"{base_prompt}, step 1: starting action, character begins action sequence",
        f"{base_prompt}, step 2: mid-point action, character gestures, dynamic movement",
        f"{base_prompt}, step 3: ending action, character completes action, resolving pose"
    ]

class ClipsGenerationRequest(BaseModel):
    images: Optional[List[str]] = []
    scene_text: Optional[str] = None
    direction: Optional[Dict[str, Any]] = None
    characters: Optional[List[Dict[str, Any]]] = []
    num_frames: Optional[int] = 3

@app.post("/generate-scene-clips")
def generate_scene_clips(request_data: ClipsGenerationRequest):
    """Generate dynamic progressive transition video MP4 clips from scene context."""
    scene_text = request_data.scene_text
    direction_data = request_data.direction
    characters = request_data.characters or []
    num_frames = min(request_data.num_frames or 3, 5)
    
    clip_filenames = []
    
    try:
        from video_generator import generate_progressive_video_clip
        
        if scene_text:
            if not direction_data:
                try:
                    direction_data = generate_direction(scene_text[:2000])
                except Exception as e:
                    print(f"Error generating direction for video clips: {e}")
                    direction_data = {
                        "camera": "wide shot",
                        "lighting": "natural",
                        "weather": "clear",
                        "mood": "neutral",
                        "color_palette": "muted tones",
                        "scene_intensity": 50
                    }
                    
            from concurrent.futures import ThreadPoolExecutor
            import uuid

            def gen_video_clip_frame(i):
                try:
                    # 1. Build base visual prompt for this storyboard moment
                    base_visual_prompt = build_visual_prompt(
                        scene_text,
                        direction_data,
                        characters,
                        frame_index=i
                    )
                    
                    # 2. Call Groq to generate 3 progressive action prompts from the base
                    prog_prompts = generate_progressive_prompts(base_visual_prompt, client)
                    
                    # 3. Generate the 3 image frames concurrently using Flux
                    def gen_sub_frame(idx_and_prompt):
                        idx, prompt_str = idx_and_prompt
                        try:
                            filename = generate_scene_image(prompt_str)
                            return filename
                        except Exception as sub_err:
                            print(f"Failed to generate subframe {idx+1} for clip {i+1}: {sub_err}")
                            return None
                            
                    with ThreadPoolExecutor(max_workers=3) as frame_executor:
                        img_files = list(frame_executor.map(gen_sub_frame, enumerate(prog_prompts)))
                        
                    # Filter out any failed image frames
                    valid_files = [f for f in img_files if f]
                    
                    # If we don't have exactly 3 frames, try to duplicate/pad to make it 3, or fail gracefully
                    if len(valid_files) < 3:
                        if len(valid_files) > 0:
                            # Pad to 3 files
                            while len(valid_files) < 3:
                                valid_files.append(valid_files[-1])
                        else:
                            raise Exception("Could not generate any subframes for progressive clip")
                            
                    image_paths = [os.path.join("generated_images", f) for f in valid_files]
                    
                    # 4. Compile the 3 images with crossfades into a single 10s video clip
                    clip_name = f"video_clip_{uuid.uuid4()}.mp4"
                    generate_progressive_video_clip(image_paths, clip_name, duration=10)
                    
                    # 5. Clean up temporary source images
                    for img_p in image_paths:
                        try:
                            if os.path.exists(img_p):
                                os.remove(img_p)
                                print(f"Deleted temporary video subframe image: {img_p}")
                        except Exception as rm_err:
                            print(f"Failed to delete temporary image {img_p}: {rm_err}")
                            
                    return clip_name
                except Exception as frame_err:
                    print(f"Error generating progressive video clip for frame {i+1}: {frame_err}")
                    return None

            with ThreadPoolExecutor(max_workers=3) as executor:
                results = list(executor.map(gen_video_clip_frame, range(num_frames)))
                clip_filenames = [c for c in results if c]
                
        else:
            # Fallback to existing storyboard images if scene_text is not provided
            images = request_data.images or []
            if not images:
                return {"clips": [], "error": "No storyboard images or scene text provided"}
                
            from video_generator import generate_video_clip
            for img in images:
                img_path = os.path.join("generated_images", img)
                if os.path.exists(img_path):
                    try:
                        base = os.path.splitext(img)[0]
                        clip_name = f"{base}.mp4"
                        generate_video_clip(img_path, clip_name, duration=10)
                        clip_filenames.append(clip_name)
                    except Exception as clip_err:
                        print(f"Error generating clip for {img}: {clip_err}")
                        
    except Exception as e:
        print(f"Error in clips generation pipeline: {e}")
        
    return {
        "clips": clip_filenames
    }

@app.delete("/generated_images/{filename}")
def delete_image(filename: str):
    file_path = os.path.join("generated_images", filename)
    # Basic security check to prevent path traversal
    if ".." in filename or filename.startswith("/") or filename.startswith("\\"):
        return {"status": "error", "message": "Invalid filename"}
    
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            return {"status": "success", "message": f"Deleted {filename}"}
        except Exception as e:
            return {"status": "error", "message": f"Failed to delete file: {str(e)}"}
    return {"status": "error", "message": "File not found"}

def _parse_json_response(response_text):
    cleaned = response_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()

class CharacterSummaryRequest(BaseModel):
    character: Dict[str, Any]
    scenes: Optional[List[Dict[str, Any]]] = []
    story_bible: Optional[Dict[str, Any]] = None

@app.post("/character-summary")
def character_summary(request_data: CharacterSummaryRequest):
    char = request_data.character
    scenes = request_data.scenes or []
    bible = request_data.story_bible or {}
    
    char_name = char.get("name", "Unnamed")
    
    scenes_context = ""
    if scenes:
        scenes_context = "Scenes history involving this character:\n"
        for idx, s in enumerate(scenes):
            scenes_context += f"- Scene {idx+1}: {s.get('title', 'Untitled')} - Prompt: {s.get('prompt')}\n"
            if s.get('direction'):
                scenes_context += f"  Direction: {s.get('direction')}\n"
    
    prompt = f"""
You are Manomaya, a narrative design assistant.
Analyze the following character profile, story bible context, and history of scenes they participated in.

Character Profile:
- Name: {char_name}
- Age: {char.get("age")}
- Gender: {char.get("gender")}
- Core Traits: {char.get("core_traits")}
- Strengths: {char.get("strengths")}
- Flaws: {char.get("flaws")}
- Fears: {char.get("fears")}
- Goals: {char.get("goals")}
- Values: {char.get("values")}
- Attachment Style: {char.get("attachment_style")}
- Communication Style: {char.get("communication_style")}
- Voice Style: {char.get("voice_style")}

Story Bible context:
{json.dumps(bible)}

{scenes_context}

Your task:
1. Generate a concise, high-quality, professional narrative summary of this character's state, personality, goals, fears, relationships, and evolution so far (about 2-3 sentences).
2. Generate an evolution timeline showing how they shifted emotionally or behaviorally across the scenes they were in. Give each timeline entry a clear "sceneNumber" (e.g. 1, 5, 10 or whatever the index is) and a short "state" label (e.g. "Confident", "Emotionally Guarded", "Trusting Again").

You MUST return your output as a valid JSON object with exactly two keys:
1. "summary": The paragraph summary text.
2. "evolution": A list of objects, each containing:
   - "sceneNumber": The scene number index (integer, e.g. 1, 2, 3...)
   - "title": The title of the scene (string)
   - "state": A short description of the character's emotional state or shift in that scene (string, e.g. "Emotionally Guarded")

Example JSON output format:
{{
  "summary": "Aisha is an emotionally reserved but deeply loyal individual. Following Ryan's confession, she became more cautious in expressing her feelings. Although trust remains strong, unresolved tension continues to influence their interactions.",
  "evolution": [
    {{ "sceneNumber": 1, "title": "First Meeting", "state": "Confident" }},
    {{ "sceneNumber": 2, "title": "The Confession", "state": "Emotionally Guarded" }},
    {{ "sceneNumber": 3, "title": "Reconciliation", "state": "Trusting Again" }}
  ]
}}
"""

    messages = [
        {"role": "user", "content": prompt}
    ]
    
    completion = groq_chat_completion(
        client,
        model=DEFAULT_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        max_tokens=1000
    )
    
    response_text = completion.choices[0].message.content.strip()
    cleaned = _parse_json_response(response_text)
    
    try:
        return json.loads(cleaned)
    except Exception as e:
        return {
            "summary": f"{char_name} is a character in the universe.",
            "evolution": []
        }

@app.get("/character-arc")
def get_character_arc_endpoint(character_name: str, project_id: Optional[str] = None, projectId: Optional[str] = None):
    pid = project_id or projectId or "default_project"
    from data.character_arcs import get_character_arc
    return get_character_arc(character_name, project_id=pid)

@app.post("/extract-features")
async def extract_features(file: UploadFile = File(...)):
    contents = await file.read()
    base64_image = base64.b64encode(contents).decode("utf-8")
    
    prompt = """
    Analyze this photo of a human character.
    Extract the following physical attributes in structured JSON format with these exact keys:
    - "gender": string (e.g., "male", "female", "non-binary")
    - "age": integer or string (e.g., 25, "mid-twenties")
    - "hair": string description of hair style, length, and color (e.g., "short curly black hair")
    - "eyes": string description of eye color/shape (e.g., "brown almond eyes")
    - "skinTone": string (e.g., "fair", "olive", "dark")
    - "clothing": string describing what they are wearing (e.g., "dark leather jacket and grey shirt")
    - "faceShape": string (choose strictly one of: "oval", "round", "square", "heart", "chiseled")
    - "appearance_summary": string combining these features into a 1-sentence prompt description (e.g., "A 25-year-old female with short curly black hair, olive skin, wearing a dark leather jacket.")
    
    Return ONLY a valid JSON object matching this structure. Do not write any markdown code block wraps (like ```json) or explanation.
    """
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        response_text = completion.choices[0].message.content.strip()
        
        cleaned = response_text
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        return json.loads(cleaned)
    except Exception as e:
        print(f"Error in vision feature extraction: {e}")
        return {
            "gender": "unknown",
            "age": "unknown",
            "hair": "unknown",
            "eyes": "unknown",
            "skinTone": "unknown",
            "clothing": "unknown",
            "faceShape": "round",
            "appearance_summary": "A human character reference."
        }