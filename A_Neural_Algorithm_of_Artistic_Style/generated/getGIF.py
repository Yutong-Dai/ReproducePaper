import glob
import re
import sys
sys.path.append('../')
from utils import get_gif

# src = 'pikachu-starry'
# src = 'lehigh-starry'
src = 'girl-starry'
fp_in = f"./{src}/*.png"

filelist = glob.glob(fp_in)
filelist.sort(key=lambda f: int(re.sub('\D', '', f)))

get_gif(filelist, f'./gifs/{src}.gif', duration=0.3)