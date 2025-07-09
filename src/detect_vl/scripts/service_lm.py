import openai
import os
import io
import base64
from openai import OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
_BASED_MODEL = "gpt-4o-mini-2024-07-18"
SYSTEM_PROMPT_WORD='''
You are an advanced multimodal assistant integrated into a mobile robot. 
I will give a cmd to the robot to reach a pleace.
Your job is to identify static semantic landmarks from the text and extract simple motion actions.
Only include static, non-movable objects (e.g., door, table, chair, shelf, cabinet, wall painting).
Exclude movable objects such as people, pets, or anything that can move independently.
Ignore the verb "go"; it is considered the default movement and not extracted as an action.
First, extract static objects; for each object, set its corresponding action as null
Then, extract simple movement actions (like "turn left", "turn right"); for each action, set its corresponding object as null
The objects and actions lists must be of the same length, aligning each object or action step by step.
For "turn" actions, describe them using angles. left use "-90", right use "90"
If both object and action are "null" at the same step, do not include that step.
I'm going to give you an example first: "go to the chair, then go out the door, and turn right."
Please output the result using the following JSON structure:
{
    "objects": ["a chair", "a pair of door", "null"],
    "actions": ["null", "null", "90"]
}
'''

BUILD_MAP_PROMPT_IMG = '''
You are a multimodal AI assistant embedded in a mobile robot, helping it build a semantic memory map for indoor navigation.

Your task is to analyze the provided image and:
1. Identify the type of room or environment (e.g., kitchen, hallway, robotics lab, etc.)
2. List static, non-movable objects that define the space (e.g., fridge, shelf, lab bench).
3. Describe the visual context of the room in 1-2 short sentences, even if no static objects are detected.


Do not classify a new room type unless the robot has passed through a door.

If no door transition was detected, assume the robot is still in the same room — do not change the room_id or its associated coordinates.

If a door was passed, update the current room_id and classify the new room using the image.

If the image is ambiguous and lacks distinctive static features, skip classification and wait for a clearer view.

Once a room_id and its coordinate have been assigned, do not reassign or update them unless a door transition is confirmed.

Exclude movable items like people, bags, laptops, chairs with wheels, or bottles.

Return your result using this strict JSON format:

```json
{
  "room_type": "bedroom",
  "features": [
    { "object": "bed" },
    { "object": "wardrobe" }
  ],
  "description": "This room has a bed and a wardrobe. It looks like a private sleeping area, likely a bedroom."
}

If no features are detected:
{
  "room_type": "robotics lab",
  "features": [],
  "description": "I see a spacious area with robot parts, tools, and no furniture. It's likely a robotics lab."
}
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

def gpt_map_build(img):
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
                    { "type": "input_text", "text": BUILD_MAP_PROMPT_IMG},
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
