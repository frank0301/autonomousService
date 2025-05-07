import json
import re

# example_ans='''
# ```json
# {
#     "objects": ["a chair", "a pair of door"],
#     "actions": ["null", "turn right"]
# }
# ```
# '''
# clean_str = re.sub(r"```", "", re.sub(r"```json", "", example_ans).strip()).strip()
# print(clean_str)

# data = json.loads(clean_str)
# # print(data)
# print(data['objects'])

def ans2json(input):
    data_str = re.sub(r"```", "", re.sub(r"```json", "", input).strip()).strip()
    data = json.loads(data_str)
    return data