import json, zlib, os, shutil, threading

# NOTE: does NOT delete files. Only moves them to another directory

data_path = 'training_data_pre_rephrased'
redundant_files_path = 'redundant'

redundant_files_path = os.path.join(os.getcwd(), redundant_files_path)
if not os.path.exists(redundant_files_path): os.mkdir(redundant_files_path)

def get_list_of_files(folder):

    p = os.getcwd()
    folder_path = os.path.join(p, folder)
    return ([os.path.join(folder_path, i) for i in os.listdir(folder_path)], os.listdir(folder_path))

def remove_cwd_path(path:str):

    return path.replace(os.getcwd(), '')

def open_data_packet(path):

    with open(path, 'rb') as f:
        # Read the entire file content as bytes
        file_content = f.read()
    # Decompress the bytes-like object and then load it as JSON
    return json.loads(zlib.decompress(file_content))

file_list, unmodified = get_list_of_files(data_path)
to_prune = 0
existing_prompts = []

for j, i in enumerate(file_list):
    
    if j % 100 == 0: print(j, '/', len(file_list), '|', j / len(file_list) * 100) 

    try:
        packet = open_data_packet(i)
    except Exception as e:
        print(e)
        print(f'File with path {i} seems to be corrupted, deleting.')
        os.remove(i)
        continue

    prompt = packet['natural_language_prompt']

    if prompt not in existing_prompts:
        existing_prompts.append(prompt)

    else:
        print('Duplicate found')
        shutil.move(i, os.path.join(redundant_files_path, unmodified[j]))
        to_prune += 1

print(f"Found {to_prune} duplicates out of {len(file_list)}. Moved to the following directory: {redundant_files_path}")