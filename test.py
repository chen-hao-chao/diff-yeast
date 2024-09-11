import pandas as pd
from ast import literal_eval 

loaded_df = pd.read_csv('/datasets/yeast-imgs/single_cell_annotations/group_by_correctedMaxCycle_num_rep1.csv', index_col='correctedMaxCycle_num')
filenames = loaded_df['filename'].apply(literal_eval)
result_dict = filenames[0][:10]
print(result_dict)