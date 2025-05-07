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

SYSTEM_PROMPT_IMG_WORD='''

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