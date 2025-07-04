
import ray
ray.init(num_cpus=32, num_gpus=1, include_dashboard=False)

from ray import tune

import frame_interpolation
from omegaconf import DictConfig
import hydra
import pandas as pd
import pdb
def runner(tuner, cfg):
    cfg['ORF'] = tuner['ORF']
    cfg['method'] = tuner['method']
    cfg['target_dir'] = '/projects/yeast-cell-diffusion/generated_video'
    print("ORF: {} | Method: {}".format(cfg['ORF'], cfg['method']))
    frame_interpolation.main(cfg)
    with open("fi_2.txt", "a+") as f:
        f.write("ORF: {} | Method: {}\n".format(cfg['ORF'], cfg['method']))

# ====================================
@hydra.main(version_base=None, config_path="conf", config_name="base")
def tuner(cfg : DictConfig):
    
    cfg = frame_interpolation.flatten_cfg(cfg)
    
    loaded_df = pd.read_csv('/datasets/yeast-imgs/single_cell_annotations/group_by_protein_stage_rep1_filename_dict.csv')
    df = loaded_df[loaded_df['correctedMaxCycle_num'] == 0]['ORF'][2:]
    ORF_list = [df[df.index.values.astype(int)[i]] for i in range(len(df))]
    print("Total ORF: ", len(ORF_list))
    
    search_space = {
        "ORF": tune.grid_search(ORF_list[1500:3000]),
        "method": tune.grid_search(['nucleus', 'random', 'structure']),
    }

    wrapped_runner = lambda x: runner(x, cfg)
    print("Start tuning...")
    analysis = tune.run(
        wrapped_runner, 
        storage_path="/h/chchao/diff-yeast/results_ray_2_fi",
        resources_per_trial={'cpu': 6, 'gpu': 0.2},
        config=search_space,
    )

if __name__ == "__main__":
    tuner()