import os
import sys

# Add the root directory to sys.path so pytest can find 'world', 'expert', etc.
sys.path.append(os.path.abspath(os.path.dirname(__file__)))