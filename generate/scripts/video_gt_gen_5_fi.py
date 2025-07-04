
import frame_interpolation
from omegaconf import DictConfig
import hydra
import pandas as pd
import pdb

@hydra.main(version_base=None, config_path="conf", config_name="base")
def tuner(cfg : DictConfig):
    cfg = frame_interpolation.flatten_cfg(cfg)
    
    print("loading files...")
    loaded_df = pd.read_csv('/datasets/yeast-imgs/single_cell_annotations/group_by_protein_stage_rep1_filename_dict.csv')
    df = loaded_df[loaded_df['correctedMaxCycle_num'] == 0]['ORF'][2:]
    ORF_list = [df[df.index.values.astype(int)[i]] for i in range(len(df))]
    ORF_list = list(set(ORF_list))[2400:2700]
    print("Total ORF: ", len(ORF_list))
    cfg['target_dir'] = '/projects/yeast-cell-diffusion/generated_video'

    for ORF in ORF_list:
        for method in ['nucleus', 'random', 'structure']:
            cfg['ORF'] = ORF
            cfg['method'] = method
            frame_interpolation.main(cfg)
            with open("fi_5.txt", "a+") as f:
                f.write("ORF: {} | Method: {}\n".format(cfg['ORF'], cfg['method']))
            
    with open("fi_5.txt", "a+") as f:
        f.write("finish!")

if __name__ == "__main__":
    tuner()