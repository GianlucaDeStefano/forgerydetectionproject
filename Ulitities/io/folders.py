import os
import time

DEBUG_ROOT = os.path.abspath("Data/Debug/")

def create_debug_folder():
    debug_folder = os.path.join(DEBUG_ROOT, str(time.time()))
    os.makedirs(debug_folder)
    return debug_folder