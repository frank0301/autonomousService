import openai
import os
import io
import base64
from openai import OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
_BASED_MODEL = "gpt-4o-mini-2024-07-18"
SYSTEM_PROMPT_WORD = '''
You are an advanced multimodal assistant integrated into a mobile robot.
I will give you a command that tells the robot how to move.

Your job is to:
1. Find all static objects (like chair, door, table, shelf, wall painting). Do not include people, pets, or things that can move.
2. For each object, also say the spatial relation: how the robot should move around it.
  : Use one of these: "near" (default), "at", "through", "past", "toward", or "facing"
  : If you give an object (not "null"), you must also give a relation (not "null")
3. Find all simple turn actions like "turn left", "turn right", or multiple turns in sequence.
  : "turn left" means: 90
  : "turn right" means -90
Assign one of the following **spatial relations**:

- `"near"` : default; stop close to the object
- `"at"` : go directly to the object
- `"through"` : pass through the object (e.g., a door or hallway)
- `"past"` : move past or alongside the object without stopping
- `"toward"` : move in the general direction of the object
**If an object is not `"null"`, its relative value must NOT be `"null"`**

Rules for formatting:
- When you extract an object, its "turn" must be "null"
- When you extract a turn, its "object" and "relative" must be "null"
- All steps must follow the order of the original sentence
- All lists (`objects`, `relative`, `turn`) must be the same length and step-by-step aligned
- Do not include any step where both "object" and "turn" are "null"
- Every turn must appear at a separate step. If a turn happens **after an object**, insert a new step for it: set "object" and "relative" to "null", and write the angle in "turn"

If a sentence says something like:
"Go to the chair, then turn left, go through the door, turn right"
You must extract 4 steps:
```json
{
  "objects": ["a chair", "null", "a pair of door", "null"],
  "relative": ["near", "null", "through", "null"],
  "turn": ["null", "90", "null", "-90"]
}

notice that, here is a wrong expamle:
{
  "objects": ["a chair", "a pair of door", "a hallway", "a wall", "null"],
  "relative": ["near", "through", "past", "facing", "null"],
  "turn": ["null", "90", "null", "null", "-90"]
}
it has the object and turn not null at the same time!
OK,now start my task:
'''

BUILD_MAP_PROMPT_IMG = '''
You are a multimodal AI assistant embedded in a mobile robot, helping it build a semantic memory map for indoor navigation.

Environment Context:
The robot is currently operating in a {context}. Use this environment context to guide your prediction of the room type. 
For example, if the context is "supermarket", prefer classifications like "aisle", "checkout area", or "storage room". 
If it's "hospital", expect areas like "corridor", "nurse station", or "patient room".

CRITICAL: DOORS ARE THE PRIMARY INDICATORS OF ENVIRONMENTAL TRANSITIONS
- Doors serve as the definitive boundary markers between different rooms and environments
- A door transition indicates the robot has moved from one distinct space to another
- Door detection is essential for accurate room classification and map building
- Without door detection, the robot cannot reliably determine when it has entered a new environment

Your task is to analyze the provided image and:
1. Identify the type of room or environment (e.g., kitchen, hallway, robotics lab, etc.)
2. List static, non-movable objects that define the space (e.g., fridge, shelf, lab bench).
3. Describe the visual context of the room in 1-2 short sentences, even if no static objects are detected.
4. Pay special attention to detecting doors, as they are crucial for environmental boundary identification.

ENVIRONMENTAL TRANSITION RULES:
- DO NOT classify a new room type unless the robot has passed through a door
- Doors are the ONLY reliable indicators of environmental transitions
- If no door transition was detected, assume the robot is still in the same room — do not change the room_id or its associated coordinates
- If a door was passed, update the current room_id and classify the new room using the image
- If the image is ambiguous and lacks distinctive static features (especially doors), skip classification and wait for a clearer view
- Once a room_id and its coordinate have been assigned, do not reassign or update them unless a door transition is confirmed

Exclude movable items like people, bags, laptops, chairs with wheels, or bottles.

Return your result using this strict JSON format:

```json
{{
  "room_type": "bedroom",
  "features": [
    {{ "object": "bed" }},
    {{ "object": "wardrobe" }}
  ],
  "description": "This room has a bed and a wardrobe. It looks like a private sleeping area, likely a bedroom."
}}

If no features are detected:
{{
  "room_type": "robotics lab",
  "features": [],
  "description": "I see a spacious area with robot parts, tools, and no furniture. It's likely a robotics lab."
}}
'''


def ask_gpt4o_with_image(img, question):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    input = None
    response = openai.responses.create(
        model = _BASED_MODEL,
        input = [
            {
                "role": "user",
                "content": [
                    { "type": "input_text", "text": SYSTEM_PROMPT_WORD + question},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                ],
            }
        ],
    )
    print(input,'\n')
    return response.output_text

def ask_gpt_ll(question):
    # input = None
    client = OpenAI()
    response = client.responses.create(
        model=_BASED_MODEL,
        input = SYSTEM_PROMPT_WORD + question,
    )

    # print(input,'\n')
    # print(response.output_text)
    return response.output_text

def gpt_map_build(img, context=""):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # Format the prompt with context
    if context:
        formatted_prompt = BUILD_MAP_PROMPT_IMG.format(context=context)
    else:
        formatted_prompt = BUILD_MAP_PROMPT_IMG.format(context="general indoor environment")
    
    input = None
    response = openai.responses.create(
        model = _BASED_MODEL,
        input = [
            {
                "role": "user",
                "content": [
                    { "type": "input_text", "text": formatted_prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                ],
            }
        ],
    )
    print(input,'\n')
    return response.output_text
