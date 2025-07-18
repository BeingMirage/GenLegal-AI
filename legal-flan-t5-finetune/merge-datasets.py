import json

files = ['data/constitution_qa.json', 'data/crpc_qa.json', 'data/ipc_qa.json']
all_data = []

for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        all_data.extend(json.load(f))

with open('data/combined_legal_qa.json', 'w', encoding='utf-8') as out:
    json.dump(all_data, out, indent=2)

print(f"Merged {len(all_data)} examples into data/combined_legal_qa.json")
