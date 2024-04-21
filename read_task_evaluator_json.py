import ast, json, math
chkpt_number = 2000

def get_pd_answer(pd: str):

    answer = pd.split('### Answer:')[1]
    while '</s>' in answer: answer = answer.replace('</s>','')

    return answer

def format_answer(gt: str):

    while '</s>' in gt: gt = gt.replace('</s>','').strip()
    
    return gt


with open(f"task_evaluation_{chkpt_number}.json", 'r') as f:

    prompts, gt, pd = json.loads(f.read())


correct = [0,0,0,0,0,0]
incorrect = [0,0,0,0,0,0]

delta = [0, 0, 0]
n_delta = [0, 0, 0]

correct_task = 0
faulty_task = 0

faultless = True
for i in range(len(gt)):

    if gt[i] == '/':
        if faultless: correct_task += 1
        else: faulty_task += 1
        continue

    gt_curr = list(ast.literal_eval(format_answer(gt[i])).values())
    pd_curr = list(ast.literal_eval(get_pd_answer(pd[i])).values())

    if gt_curr[0] != pd_curr[0]: # state number
        faultless = False
        incorrect[0] += 1
        continue
    else:
        correct[0] += 1

    if not math.isclose(gt_curr[1], pd_curr[1], abs_tol=0.174533): # orientation, 10 degrees deviation
        faultless = False
        incorrect[1] += 1
        delta[0] += abs(gt_curr[1] - pd_curr[1])
        n_delta[0] += 1
        continue
    else:
        correct[1] += 1
    
    if not math.isclose(gt_curr[2], pd_curr[2], abs_tol=0.01): # distance to next point, 0.2m deviation
        faultless = False
        incorrect[2] += 1
        delta[1] += abs(gt_curr[2] - pd_curr[2])
        n_delta[1] += 1
        continue
    else:
        correct[2] += 1

    if not math.isclose(gt_curr[3], pd_curr[3], abs_tol=0.01): # execution length
        faultless = False
        incorrect[3] += 1
        delta[2] += abs(gt_curr[3] - pd_curr[3])
        n_delta[2] += 1
        continue
    else:
        correct[3] += 1

    if gt_curr[4] != pd_curr[4]: # movement message
        faultless = False
        incorrect[4] += 1
        continue
    else:
        correct[4] += 1

    if gt_curr[5] != pd_curr[5]: # instruction complete
        faultless = False
        incorrect[5] += 1
    else:
        correct[5] += 1

print("Correct , Incorrect:", correct_task, faulty_task)
print("Ratio", round(correct_task / (faulty_task + correct_task)))
print("Accuracies:")

props = ["state number",
"orientation",
"distance to next point",
"execution length",
"movement message",
"instruction complete"]

for i in range(6):
    total = incorrect[i] + correct[i]
    print(props[i],':',round(correct[i] / total * 100,2), correct[i], incorrect[i])

average_errors = [round(delta[i] / n_delta[i],3) for i in range(3)]
print("Average absolute float errors:", average_errors)