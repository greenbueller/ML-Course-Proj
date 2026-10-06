# Team Pawsitive Reinforcement

# Installation

To install, first clone the repo.
1. Navigate to where you would like to store the file
2. Open your terminal
3. In the terminal type `git clone https://github.com/greenbueller/ML-Course-Proj.git`
4. Run `pip install -r requirements.txt`
    1. If you already have a version of torch and torch vision installed locally (i.e. you have CUDA-enabled), proceed
    2. If you do not, then:
       1. If you have a CUDA compatible GPU, then follow https://pytorch.org/get-started/locally.
       2. Otherwise, run `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` before proceeding to install the CPU version of Torch.
5. When running the Kaggle datasets, get a legacy API file and store it as `[local user directory]/.kaggle/kaggle.json`

All of the necessary components should now be ready.

# Getting the images

From here, you now need to download the datasets to your machine for testing.

This is done via the dataset manager for the algorithm, or via `scripts/download_data.py`.

Simply run `python scripts/download_data.py` and you will get the available commands:
- `list` will list the 3 datasets that can be downloaded
- `all` will download all 3 datasets
- `set_[n]` will download one of the databases (1 to 3)

# Normalisation

Before training can occur, you must normalise the images. Keep in mind that only data sets 1 and 2 are used.

To do this, run `python src/normalise.py`. This will then get every image into the appropriate dimensions.

# Training

Finally, you can conduct training. Run `python src/train.py`.

# Removing the images

If you are concerned about storage, or just want to wipe this project, you can run a command to clear the cached images

Simply run `python scripts/clean_data.py` and you will get the available commands:
- `list` will list the datasets cached
- `all` will clear your entire cache
- `set_[n]` will delete only the specified image cache