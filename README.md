Python Language classification problem
======================================

## Project' structure ##

```
docs/
    report.md           Report document
    task.md             Task description

pylangkit/
    naive_bayes.py      Implementation of Naive Bayes language classification algorithm

resources/
    lang_data.csv       Input dataset
    lang_data_test.csv  Test dataset
    lang_data_train.csv Train dataset
    test_model.pickle   Pickled trained model for testing purposes
    train_model.pickle  Pickled trained model for classification demo task

tests/
    test_naive_bayes.py Tests of Naive Bayes language classification algorithm

demo.py                 Demo app of language classification task

Input_Data_Analysis_and_Model_Training.ipynb    Ipyton notebook with data analysis, training model, classification task and evaluation.

LICENSE                 License file

README.md               this file with short description of project

requirements.txt        Requirements of libraries and packages

setup.py                Setup file for package installation
```

## Environment setup ##

Install virtualenv

```bash
sudo apt install python-pip
pip install virtualenvwrapper
```

Add next lines to ~/.bashrc (~/.profile)

```bash
export WORKON_HOME=$HOME/.virtualenvs
export PROJECT_HOME=$HOME/Devel

# load virtualenvwrapper for python (after custom PATHs)
venvwrap="virtualenvwrapper.sh"
/usr/bin/which $venvwrap
if [ $? -eq 0 ]; then
    venvwrap=`/usr/bin/which $venvwrap`
    source $venvwrap
fi
```

Run script

```bash
. ~/.local/bin/virtualenvwrapper.sh
```

Create virtual environment

```bash
mkvirtualenv -p python3.5 langkit
```

## Install dependencies ##

```bash
pip install -r requirements.txt
```

## install the package locally (for use on our system) ##

```bash
pip install -e .
```

## Run tests ##

```bash
pytest tests
```

## Run demo app ##

```bash
python demo.py
```