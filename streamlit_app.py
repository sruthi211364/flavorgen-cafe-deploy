import sys
import os

# Add both root and flavorgen subfolder to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "flavorgen"))

from flavorgen.app_streamlit import *