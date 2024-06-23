import os
import shutil

def bisect_folder_contents(base_folder):
    # Path to the original folder
    src_folder = os.path.join(base_folder, 'training_data_pre')
    
    # Paths to the subfolders
    dest_folder1 = os.path.join(base_folder, 'training_data_pre_1')
    dest_folder2 = os.path.join(base_folder, 'training_data_pre_2')
    
    # Create the subfolders if they do not exist
    os.makedirs(dest_folder1, exist_ok=True)
    os.makedirs(dest_folder2, exist_ok=True)
    
    # List all files and directories in the original folder
    items = os.listdir(src_folder)
    
    # Calculate the index to split the items list
    mid_index = len(items) // 2
    
    # Move the first half of items to subfolder 1
    for item in items[:mid_index]:
        shutil.move(os.path.join(src_folder, item), dest_folder1)
    
    # Move the second half of items to subfolder 2
    for item in items[mid_index:]:
        shutil.move(os.path.join(src_folder, item), dest_folder2)

# Example usage:
bisect_folder_contents(os.getcwd())
