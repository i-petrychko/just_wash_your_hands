import json

with open('data/icip/annotations/val.json', 'r') as f:
    data = json.load(f)
    for ann in data['annotations']:
        ann['iscrowd'] = 0

with open('data/icip/annotations/val_iscrowd.json', 'w') as f:
    json.dump(data, f)
