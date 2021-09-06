import os
import time

def create_debug_folder(debug_root):
    debug_folder = os.path.join(debug_root, str(time.time()))
    os.makedirs(debug_folder)
    return debug_folder