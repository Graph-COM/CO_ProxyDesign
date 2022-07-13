import json
import pandas as pd
import pickle

# read the solutions and save into compatible format
with open('new_test/gumbel/2.pkl', 'rb') as f:
    data = pickle.load(f)
    
gumbel = data["output_rounded"].tolist()

# generate solutions
dir_gumbel = []
for i in range(21):
    cur = []
    for j in range(len(gumbel)):
        if gumbel[j][i] == 10:
            cur.append(j + 1)
    dir_gumbel.append(cur)

df = pd.DataFrame([dir_gumbel],index=["solution"])
df.to_json(r'case_2_gumbel_solution.json')