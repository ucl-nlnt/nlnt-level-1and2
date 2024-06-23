import json, zlib, ast

def open_data_packet(path: str):

    with open(path, 'rb') as f:
        # Read the entire file content as bytes
        file_content = f.read()
    # Decompress the bytes-like object and then load it as JSON
    return json.loads(zlib.decompress(file_content))

def save_data_packet(data_dict: dict, path: str):

    with open(path,'wb') as f:

        f.write(zlib.compress(json.dumps(data_dict).encode('utf-8')))