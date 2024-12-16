# Install
## Pixi environment
If pixi not installed: `curl -fsSL https://pixi.sh/install.sh | bash`
- For developers:` pixi shell -e dev`
- For users:` pixi shell`
## Python kernel (useful for notebooks)
`make python_kernel`
Then select `alise_kernel `in notebooks
# CropRot dataset
## Split the dataset
Change if necessary config into `./config/split_croprot.yaml`
1. `cd script`
2. `python split_croprot.py --dataset_path=path_to_crop_rot`
# Trainings
