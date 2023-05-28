# ray_benchmarks
This is sub-repository to ray-poc.

## Repository setup
### Install `awscli`
https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html#cliv2-linux-install

### Cofigure your AWS `credentials` file
https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#

### Install Anaconda
https://docs.anaconda.com/free/anaconda/install/linux/
https://www.youtube.com/watch?v=MUZtVEDKXsk

### Create conda environment from `*.yaml` file
Open linux terminal using your favourite tool (i.e. `Alt + Ctrl + T`). 
Copy and paste lines below (treat every code cell separately):
```
conda create --name ray_benchmarks python=3.10 -y
conda activate ray_benchmarks
conda install jupyter -y
```
Check if the adequate pip3 is in use - i.e.:
```
(ray_benchmarks) <LINUX_USER_NAME>:~/Documents/programming/python/ray-poc$ pip3 -V
pip 23.0.1 from /home/<LINUX_USER_NAME>/anaconda3/envs/ray_benchmarks/lib/python3.10/site-packages/pip (python 3.10)
```
If everything is alright - continue copying and pasting:
```
pip install torch torchvision torchaudio
pip install numpy pandas boto3 tqdm tabulate
pip install -U "ray[default]"
```

```
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
```

Do zrobienia:
1. W pliku ray_AWS_CNN_playground.yaml w sekcji `setup_commands` odinstaluj i zainstaluj condę.
2. Zmień wersję ImageId w `availble_node_types`