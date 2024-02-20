import numpy as np

from julia.api import Julia
jpath = "/home/computer/julia-1.6.3/bin/julia" # path to Julia, from current directory (your path may be slightly different)
jl = Julia(runtime=jpath, compiled_modules=False) # compiled_modules=True may work for you; it didn't for me
# jl.include("classifier.jl")
# jl.eval("importall julia_util")  # I used importall, sure other stuff also works.

# Import Julia Modules
from julia import Main
Main.include("classifier.jl")

train_input_dir = '../data/training1.txt'
train_label_dir = '../data/training1_label.txt'
pred_file = 'result'

# Reading data
train_data = np.loadtxt(train_input_dir,skiprows=0)

[num, _] = train_data.shape

# coeffs = red_c - blue_c
# mid = (red_c + blue_c)/2.0
#  
# line_x = np.linspace(-5, 5)
# line_y2 = [-(coeffs[0]/coeffs[1])*(i - mid[0]) + mid[1] for i in line_x]
