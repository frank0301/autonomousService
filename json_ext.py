import json
import re
answer = '''
```json
{
    "find_in_img": [
        {
            "object": "monitor",
            "center": [225, 162]
        },
        {
            "object": "cup",
            "center": [314, 250]
        }
    ],
    "target": {
        "object": "cup",
        "center": [314, 250]
    }
}
```
'''

def get_depth_from_box(xy):
        return 1


clean_str = re.sub(r"```", "", re.sub(r"```json", "", answer).strip()).strip()
xy = None
# json.loads(re.sub(r"```", "", re.sub(r"```json", "", answer).strip()).strip()).get("xy",None)
# json.loads(re.sub(r"```", "", re.sub(r"```json", "", answer).strip(
# 解析 JSON
data = json.loads(clean_str)

# 分别提取 object 和 center
objects = [item['object'] for item in data['find_in_img']]
centers = [item['center'] for item in data['find_in_img']]

# 目标 target
target_object = data['target']['object']
target_center = data['target']['center']

print("objects:", objects)
print("centers:", centers)
print("target:", (target_object, target_center))

print(target_center)
get_depth_from_box(target_center)
# data = json.loads(clean_str) 
# xy = data["xy"] 
# print(xy)