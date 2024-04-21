import pygame
import sys
import threading
import time
import socket
import struct
import uuid
import os
import ast
import json
import base64
import numpy as np
import cv2
import argparse
import random
import zlib

from KNetworking import DataBridgeServer_TCP

# from data_receiver_pygame.py

class turtlebot_controller:

    def __init__(self, manual_control=True):

        self.data_buffer = None

        self.gathering_data = False

        # will be useful once data gathering automation is started
        self.manual_control = manual_control

        #
        self.sesh_count = 1

        # keyboard stuff
        self.keyboard_input = None
        self.keyboard_impulse = False
        self.killswitch = False
        self.debug = True
        self.debug_window_process = True
        self.label = None

        # socket programming stuff
        self.server_data_receiver = DataBridgeServer_TCP(port_number=50000)
        print(f'Server Listening on port 50000')

        self.movement_data_sender = DataBridgeServer_TCP(port_number=50001)
        print(f'Server Listening on port 50001')

        # multithreading processes
        self.window_thread = threading.Thread(target=self.window_process)   # Pygame window process
        self.window_thread.start() # Thread 1
        self.listener_thread = threading.Thread(target=self.kb_listener)    # Keyboard listener
        self.listener_thread.start() # Thread 2
        self.data_listener_thread = threading.Thread(target=self.super_json_listener)
        self.data_listener_thread.start() # Thread 3

    def super_json_listener(self):

        t = time.time() + 1.0
        x = 0

        while True:
            
            if not self.gathering_data: time.sleep(0.007); continue
            if time.time() > t: 
                t = time.time() + 1.0
                x = 0
            x += 1
            data = json.loads(self.server_data_receiver.receive_data().decode())
            if self.data_buffer == None: print("WARNING: data buffer is still None type."); continue

            self.data_buffer.append(data)

    def kb_listener(self):  # do Turtlebot controller stuff here

        """
        Instructions: // all data instructions are 5 bytes wide
        @XXXX, where the 4 X's are the instructions to be sent to the robot.
        @0000 - Stop
        @FRWD - Forward
        @LEFT - Turn Left
        @RGHT - Turn Right
        @STRT - start recording data
        @STOP - stop recording data
        @KILL - stop program
        @RNDM - randomize position, -5 <= x,y <= 5, 0 <= theta <= 2pi # NOTE: may not be used for now

        Each data send starts with sending an 8-byte long size indicator. Each data segment is sent in 1024-byte-sized chunks.
        Once all is received, receiver sends an 'OK' to sender to indicate that is done processing whatever data was
        sent over.
        """

        # start host loop -> enter prompt -> start logging -> do stuff -> end logging -> generate unique id -> confirm save -> save data as a json with unique id

    
        # assumption: socket connection is successful
        while not self.killswitch:  # Start host loop

            if prompt != '$CONTROL':
                print('sending START')
                self.data_buffer = []                               # reset buffer. this will be filled up somewhere else (self.super_json_listener)
                self.gathering_data = True
                self.movement_data_sender.send_data('@STRT')        # send data gathering start signal
                print('waiting for CONT')
                self.movement_data_sender.receive_data()            # wait for @CONT
            else:
                print('Activating movement debug mode.')

            while True: # Inner loop, data collection
            
                if not self.keyboard_impulse: time.sleep(0.01667); continue
                print("input received:", self.keyboard_input)

                if self.keyboard_input == 'w': 
                    self.movement_data_sender.send_data(b'@FRWD')
                    self.keyboard_impulse = False                   # set to False to be able to tell when user lets go of the key.

                elif self.keyboard_input == 'a':
                    self.movement_data_sender.send_data(b'@LEFT')
                    self.keyboard_impulse = False

                elif self.keyboard_input == 'd':
                    self.movement_data_sender.send_data(b'@RGHT')
                    self.keyboard_impulse = False

                elif self.keyboard_input == None:                   # Key was let go
                    self.movement_data_sender.send_data(b'@0000')
                    self.keyboard_impulse = False

                elif self.keyboard_input == ']':
                    self.movement_data_sender.send_data(b'@STOP')
                    self.keyboard_impulse = False
                    print("Data collection is finished for this iteration.")
                    break

                time.sleep(0.01667)

            # confirm if data is good to be saved
            self.gathering_data = False

            if prompt == '$CONTROL':
                continue

            while True:
                confirmation = input("Save log? (y/n/e) << ")
                if confirmation == 'y' or confirmation == 'n': break
                elif confirmation == 'e':
                    prompt = input("Revised Prompt: ")

            if confirmation == 'y':

                # save data into a buffer
                
                json_file = {
                            "username":self.current_user, "natural_language_prompt": prompt,
                            "timestamp_s":time.ctime(), "timestamp_float":time.time(),
                            "states":self.data_buffer
                            }
                
                fname = self.generate_random_filename()

                if not args.disable_log_compression:
                    fname = fname + ".compressed"

                with open(os.path.join("datalogs",fname),'wb') as f:
                    if args.disable_log_compression:
                          f.write(json.dumps(json_file, indent=4).encode('utf-8'))
                    else:
                        f.write(zlib.compress(json.dumps(json_file, indent=4).encode('utf-8')))

                with open(os.path.join("datalogs",fname),'rb') as f:
                    compressed = f.read()

                if args.disable_log_compression:
                    print("Saved as: " + fname)
                    print(json.loads(compressed.decode('utf-8')).keys())
                else:
                    print(json.loads(zlib.decompress(compressed).decode('utf-8')).keys())
                
                print("Instance saved.")

                self.sesh_count += 1
            
            else:
                print("Instance removed.")

    def send_data(self, data: bytes):

        data = data.encode()

        # NOTE: data may or may not be in string format.

        length_bytes = struct.pack('!I', len(data))

        if self.debug:
            print('[S] Sending byte length...')
        self.client.sendall(length_bytes)
        ack = self.client.recv(2)  # wait for other side to process data size
        if ack != b'OK':
            print(f'[S] ERROR: unmatched send ACK. Received: {ack}')
        if self.debug:
            print('[S] ACK good')

        if self.debug:
            print('[S] Sending data...')
        self.client.sendall(data)  # send data
        if self.debug:
            print('[S] Data sent; waiting for ACK...')
        ack = self.client.recv(2)  # wait for other side to process data size
        if ack != b'OK':
            print(f'[S] ERROR: unmatched send ACK. Received: {ack}')
        if self.debug:
            print('[S] ACK good. Data send success.')

    def receive_data(self):

        # NOTE: Returns data in BINARY. You must decode it on your own

        if self.debug:
            print('[R] Waiting for byte length...')
        length_bytes = self.client.recv(4)
        length = struct.unpack('!I', length_bytes)[0]
        if self.debug:
            print(f'[R] Byte length received. Expecting: {length}')
        data, data_size = b'', 0

        self.client.send(b'OK')  # allow other side to send over the data
        if self.debug:
            print(f'[R] ACK sent.')
        while data_size < length:

            chunk_size = min(2048, length - data_size)
            data_size += chunk_size
            data += self.client.recv(chunk_size)
            if self.debug:
                print(f'[R] RECV {chunk_size}')

        if self.debug:
            print('[R] Transmission received successfull. Sending ACK')
        self.client.send(b'OK')  # unblock other end
        if self.debug:
            print('[R] ACK sent.')
        return data  # up to user to interpret the data


x = turtlebot_controller()
while not x.killswitch:
    time.sleep(5)
