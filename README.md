## Geometric Sketch
To build the cpp CLI, run cmake in the cpp folder.
Afterwards, run main.py in the python folder - you might need to change the path to the executable 
which is defined at the top of the main.py file, if its not in the expected path. 
You need a folder named figures for the output, and a folder named pcaps containing a trace named capture.txt with one integer key per line. If you do not have access to the CAIDA trace mentioned in the paper, you can generate a synthetic trace with a similar zipfian
parameter by running generate_synth_caida_trace.py. You should also generate the synthetic traces used in the paper by running zipf.py.

You can generate the graphs in parallel (default), or in serial order, by replacing parallel() with serial() in main.py.
Use serial if your computer is weak.

The tag VLDB_submission_version marks the state of GS when we submitted our article. Some minor modifications were made since - adding tests, refactoring code to be more readable etc.