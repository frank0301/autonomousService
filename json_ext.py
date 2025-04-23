import json
import re
answer = '''
```json
{
    "object": "keyboard",
    "xy": [[450, 350], [580, 400]]
}
```
'''
clean_str = re.sub(r"```", "", re.sub(r"```json", "", answer).strip()).strip()
xy = None
xy = json.loads(re.sub(r"```", "", re.sub(r"```json", "", answer).strip()).strip()).get("xy",None)
(x1, y1), (x2, y2) = xy
print(x1,y1)
# data = json.loads(clean_str) 
# xy = data["xy"] 
# print(xy)