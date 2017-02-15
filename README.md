# python_language_classification
Python Language classification problem
======================================

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