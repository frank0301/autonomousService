import openai
import os
import io
import base64
from openai import OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
_BASED_MODEL = "gpt-4o-mini-2024-07-18"
{
   
# You are an advanced multimodal assistant integrated into a mobile robot. 
# I will give a cmd to the robot to reach a pleace.

# Your job is to identify static semantic landmarks from the text and extract simple motion actions.
# Your job is to:
# 1. Identify *static semantic landmarks* (objects).
# 2. Extract *simple motion actions* (like "turn left", "turn right").
# 3. For each static object, also extract its *spatial relation* to the robot’s intended goal (e.g., go through the door, stop near the shelf).
# Ignore the verb "go"; it is considered the default movement and not extracted as an action.
# First, extract static objects; for each object, set its corresponding action as null
# When describing an object, assign one of the following spatial relations:
#: "near": stop close to the object, default setting
#: "at": go directly to the object
#: "through": move past the object from one side to another (e.g., door, hallway)
#: "past": pass near the object without stopping
#: "toward": move in the direction of the object, but not necessarily reaching it
#: "facing": rotate or orient the robot toward the object without approaching
# and make it not "null" while the objects is not "null"
# Then, extract simple movement actions (like "turn left", "turn right"); for each action, set its corresponding object as null
# The objects and actions lists must be of the same length, aligning each object or action step by step.
# For "turn" actions, describe them using angles. left use "-90", right use "90"
# If both object and action are "null" at the same step, do not include that step.
# I'm going to give you an example first: "go to the chair, then go out the door, and turn right."
# Please output the result using the following JSON structure:
# {
#     "objects": ["a chair", "a pair of door", "null"],
#     "relative": ["near", "through", "null"],
#     "actions": ["null", "null", "90"],
    
# }
# 

}

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