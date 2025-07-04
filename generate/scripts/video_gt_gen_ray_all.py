import frame_matching
from omegaconf import DictConfig
import hydra
import pandas as pd

# ====================================
@hydra.main(version_base=None, config_path="conf", config_name="base")
def tuner(cfg : DictConfig):
    cfg = frame_matching.flatten_cfg(cfg)
    cfg['target_dir'] = '/projects/yeast-cell-diffusion/generated_video'

    loaded_df = pd.read_csv('/datasets/yeast-imgs/single_cell_annotations/group_by_protein_stage_rep1_filename_dict.csv')
    df = loaded_df[loaded_df['correctedMaxCycle_num'] == 0]['ORF'][2:]
    ORF_list = [df[df.index.values.astype(int)[i]] for i in range(len(df))]
    ORF_list = list(set(ORF_list))

    for ORF in ORF_list:
        for method in ['nucleus', 'random', 'structure']:
            cfg['ORF'] = ORF
            cfg['method'] = method
            print("ORF: {} | Method: {}".format(cfg['ORF'], cfg['method']))
            frame_matching.main(cfg)

if __name__ == "__main__":
    tuner()