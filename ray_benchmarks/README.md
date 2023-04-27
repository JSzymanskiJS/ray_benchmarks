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
pip install numpy pandas boto3 tqdm
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

```
Do przegadania:
https://docs.ray.io/en/latest/data/key-concepts.html#fault-tolerance

Pomysł: Napisanie zwykłej Ray-owej funkcji (Bez Ray[Data]) do odczytu zdjęć z Datasetu i utworzenie z tego DataLoader-a.
Pomysł: Napisanie Ray-owej funkcji na bazie Ray[Data] do odczytu zdjęć z Datasetu i utworzenie z tego Ray-data-owego 
    DataLoader-a.

https://docs.ray.io/en/latest/ray-air/getting-started.html#project-status
https://modal.com/

Czy 'ray.init(num_cpus)'

Znam programistę Rust, który szuka pracy.

Jak wygląda flow tworzenia oprogramowania w oparciu o AI?
    - Zaczynasz od celu biznesowego i budżetu klienta.
    - Do celu wybierasz metrykę.
    - Do metryki wybierasz funkcję kosztu (typ problemu).
    - *Do funkcji kosztu dobieramy algorytm AI.*
    - *Do algorytmu AI dobieramy framework-i i języki, w których potrafimy pisać.*
    - *Algorytmy ubieramy w przydatną biznesowo i softwarowo abstrakcję.*
    - *Zaimplementowanie algorytmu w architekturę spełniającą oczekiwania UX-owe klienta.*
    - *Deploy.*
    - *Support.*
*Zależy od kosztu.

```