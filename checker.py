import json

with open('eval_test_ckpt500.json','r') as f:
    data = json.loads(f.read())

for arr in data:
    prompt, gt, pred = arr
    print('prompt:',prompt)
    print('\n')
    print('gt:',arr[1])
    print('\n')
    print('pred:',arr[2])
    print('\n')
    break