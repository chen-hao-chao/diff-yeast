import pandas as pd
from ast import literal_eval

loaded_df = pd.read_csv('/datasets/yeast-imgs/single_cell_annotations/group_by_protein_stage_rep1_filename_dict.csv')
df = loaded_df[loaded_df['correctedMaxCycle_num'] == 0]['ORF'][2:]

ORFs = [df[df.index.values.astype(int)[i]] for i in range(len(df))]
print(ORFs)

df_list = []
for i in range(6):
    print("Loading the {}-th stage...".format(str(i)))
    df_ = loaded_df[loaded_df['correctedMaxCycle_num'] == i]
    df_ = df_[df_['ORF'] == ORFs[0]]["filename"].apply(literal_eval)
    file = df_[df_.index.values.astype(int)[0]]
    df_list.append(file)

print(df_list)