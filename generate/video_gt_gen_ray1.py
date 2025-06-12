
import ray
from ray import tune

import frame_matching
from omegaconf import DictConfig
import hydra
import pandas as pd

def runner(tuner, cfg):
    cfg['ORF'] = tuner['ORF']
    cfg['method'] = tuner['method']
    print("ORF: {} | Method: {}".format(cfg['ORF'], cfg['method']))
    frame_matching.main(cfg)

# ====================================
@hydra.main(version_base=None, config_path="conf", config_name="base")
def tuner(cfg : DictConfig):
    ray.init(num_cpus=16)
    cfg = frame_matching.flatten_cfg(cfg)
    
    loaded_df = pd.read_csv('/datasets/yeast-imgs/single_cell_annotations/group_by_protein_stage_rep1_filename_dict.csv')
    df = loaded_df[loaded_df['correctedMaxCycle_num'] == 0]['ORF'][2:]
    ORF_list = [df[df.index.values.astype(int)[i]] for i in range(len(df))]

    search_space = {
        "ORF": tune.grid_search(ORF_list[:250]),
        "method": tune.grid_search(['nucleus', 'random', 'structure']),
    }

    wrapped_runner = lambda x: runner(x, cfg)

    analysis = tune.run(
        wrapped_runner, 
        storage_path="/h/chchao/diff-yeast/results_ray_1",
        resources_per_trial={'cpu': 8, 'gpu': 0},
        config=search_space,
    )

if __name__ == "__main__":
    tuner()