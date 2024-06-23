import json

list_of_impossible_prompts = []

with open('results0.json', 'r') as f:
    
    results_list = json.loads(f.read())
    possibility = results_list[-1]

print(possibility)

with open('results1.json', 'r') as f:
    
    results_list = json.loads(f.read())
    possibility = results_list[-1]

print(possibility)

with open('results2.json', 'r') as f:
    
    results_list = json.loads(f.read())
    possibility = results_list[-1]

print(possibility)

with open('results3.json', 'r') as f:
    
    results_list = json.loads(f.read())
    possibility = results_list[-1]

print(possibility)

with open('revised.json','r') as f:

    data = []
    for i in json.loads(f.read()):
        data += i

data = list(set(data)) # remove duplicates

for entry in data:

    if "I need you to break-down this natural language prompt into a series of steps" in entry:

        
        prompt = entry[entry.index('<|end|>'):]
        if '### Possibility: False' in prompt:
            print(prompt.index("Possibility: False"))
            print('===================================')
            print(prompt)