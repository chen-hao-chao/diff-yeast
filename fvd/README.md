# fvd

Visualization tool for the plate images in `/datasets/yeast-imgs/gt_video/videos/Images`.

Filename pattern: `r{row}c{col}f{field}p{plane}-ch{channel}sk{frame}fk1fl1.tiff`
- `row`/`col`: well position (see `PROTEIN_TABLE` in the script for protein/ORF names)
- `field`: field of view within the well (f01-f24)
- `channel`: ch1 phase contrast, ch2 protein-GFP, ch3 nucleus+septin+cytoplasm bleed-through, ch4 cytoplasm (E2Crimson)
- `frame`: sk1-sk20, timepoints of the video

## Usage

```bash
source ~/projects/aip-rahulgk/chchao0/venvs/diff-yeast/bin/activate

python3 visualize_images.py list                                     # wells with data + protein names
python3 visualize_images.py montage --protein Nuf2 --field 1 --sk 1   # 4 channels + composite, PNG
python3 visualize_images.py video --protein Nuf2 --gif --mp4          # sk1-sk20 timelapse
python3 visualize_images.py plate --field 1 --sk 1                    # one thumbnail per well
```

Wells can be selected with `--protein NAME` or `--row A --col 4`. Outputs go to `viz_out/`.

## Composite modes (`--mode`, default `clean`)

- `rgb`: R=ch3, G=ch2, B=ch4
- `clean`: R=(ch3-ch4) cytoplasm-subtracted, G=ch2 (default)
- `no_ch4`: R=ch3 raw, G=ch2, ch4 ignored

```bash
python3 visualize_images.py montage --protein Nuf2 --mask                          # Otsu threshold (auto)
python3 visualize_images.py video --protein Nuf2 --mask --gif
python3 visualize_images.py video --protein Nuf2 --mask --gif --mp4 --fps 4

```