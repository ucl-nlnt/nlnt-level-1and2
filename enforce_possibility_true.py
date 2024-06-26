import os
import zlib
import json
import quart_funcs
import random
# from nlnt_types import StatesList
from transformers import AutoTokenizer
import math
import sys
from collections import deque
import shutil


def open_data_packet(path):

    with open(path, 'rb') as f:
        # Read the entire file content as bytes
        file_content = f.read()
    # Decompress the bytes-like object and then load it as JSON
    return json.loads(zlib.decompress(file_content))

def quaternion_to_yaw(x, y, z, w):  # Generated by GPT-4
    """
    Convert a quaternion into yaw (rotation around z-axis in radians)
    """
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    return yaw_z

def find_keyframes_indexes(state_packet):

    frames = state_packet['states']

    twist_last = frames[0]['twist']
    keyframes = [0]
    for i, frame in enumerate(frames):
        if frame['twist'] != twist_last:
            twist_last = frame['twist']
            keyframes.append(i)

    return keyframes

def prep_frame_data(state_list: list, keyframe_indexes):

    data = [state_list[i] for i in keyframe_indexes]

    # shift and rotate odometry data
    reference_position = data[0]['odometry']['pose_position']
    reference_rotation = data[0]['odometry']['pose_orientation_quarternion']

    inverse_reference_rotation = quart_funcs.inverse_quarternion(
        reference_rotation)

    for i, frame in enumerate(data):

        odom = frame['odometry']
        old_position, old_rotation = odom['pose_position'], odom['pose_orientation_quarternion']

        new_position = quart_funcs.adjust_position_origin(
            reference_position, old_position)
        new_rotation = quart_funcs.adjust_orientation_origin(
            inverse_reference_rotation, old_rotation)

        odom['pose_position'] = new_position
        odom['pose_orientation_quarternion'] = new_rotation

        data[i]['odometry'] = odom

    return data

def compute_distance(coord1, coord2):

    total = 0
    for i in range(3):
        total += (coord2[i] - coord1[i])**2

    return math.sqrt(total)

file_paths = [os.path.join(os.getcwd(),'training_data_pre_rephrased',i) for i in os.listdir("training_data_pre_rephrased")]
moved_no_labels, moved_to_false = 0, 0
for i in file_paths:

    packet = open_data_packet(i)

    if "### Possibility" not in packet['explanation']:
        print("No possibility")
        #print('======================================')
        #print(packet['explanation'])
        new_path = os.path.join(os.getcwd(),i.replace(os.path.join(os.getcwd(),'training_data_pre_rephrased'), "no_possibility_label"))
        shutil.move(i, new_path)
        #print(f"Moved {i} to {new_path}.")
        moved_no_labels += 1
        continue

    elif "### Possibility: False" in packet['explanation']:
        print("Failed")
        new_path = os.path.join(os.getcwd(),i.replace(os.path.join(os.getcwd(),'to_be_relabeled_true'), "no_possibility_label"))
        #print('======================================')
        #print(packet['explanation'])
        shutil.move(i, new_path)
        #print(f"Moved {i} to {new_path}.")
        moved_to_false += 1
        continue

print("No labels:", moved_no_labels)
print("Incorrect possibility labels:", moved_to_false)