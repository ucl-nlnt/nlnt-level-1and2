"""
The following script is a template that can be used for when people are finally running tasks on the project
"""

import threading
import queue
import time

class robotPilot:

    def __init__(self, task_queue: queue.Queue, response_queue: queue.Queue):

        self.task_stack = task_queue            # append instructions to this to send it to the ttb container
        self.response_stack = response_queue    # responses from turtlebot should show up here
        self.task_threads = []                  #
        
        self.garbage_man = threading.Thread(target=self.garbage_handler)
        self.garbage_man.daemon = True
        self.garbage_man.start()

        # ===================================================================
        # do rest of initialization here
    
        pass

        # ===================================================================


    # initialize helper functions here



    # ===================================================================
    # DO NOT MODIFY:

    def asynchronous(self, target, args = None, start = True):
    
        if type(args,(list, tuple)) == list:
            self.task_threads.append(threading.Thread(target=target, args=args))
        
        else:
            self.task_threads.append(threading.Thread(target=target))

        if start:
            self.task_threads[-1].daemon = True
            self.task_threads[-1].start()

    def garbage_handler(self):

        while True:

            self.task_threads = [t for t in self.task_threads if t.is_alive()]
            time.sleep(0.5)