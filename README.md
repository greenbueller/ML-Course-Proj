# Team Pawsitive Reinforcement

# Installation

To install, first clone the repo.
1. Navigate to where you would like to store the file
2. Open your terminal
3. In the terminal type `git clone https://github.com/greenbueller/ML-Course-Proj.git`
4. Run `pip install -r requirements.txt`
5. When running the Kaggle datasets, get a legacy API file and store it as `[local user directory]/.kaggle/kaggle.json`

All of the necessary components should now be ready.

# Getting the images

From here, you now need to download the datasets to your machine for testing.

This is done via the dataset manager for the algorithm, or via `scripts/download_data.py`.

Simply run `python scripts/download_data.py` and you will get the available commands:
- `list` will list the 3 datasets that can be downloaded
- `all` will download all 3 datasets
- `set_[n]` will download one of the databases (1 to 3)

# Removing the images

If you are concerned about storage, or just want to wipe this project, you can run a command to clear the cached images

Simply run `python scripts/clean_data.py` and you will get the available commands:
- `list` will list the datasets cached
- `all` will clear your entire cache
- `set_[n]` will delete only the specified image cache