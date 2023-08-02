# Example notebooks

Examples of the things you can do and build with our data! Static versions of the notebooks are available on the [developers site](https://developers.wellcomecollection.org).

## Running the notebooks

### Using docker (recommended)

```bash
docker run \
    --rm \
    -p 8888:8888 \
    -v $(git rev-parse --show-toplevel)/notebooks:/home/jovyan/work \
    jupyter/scipy-notebook:python-3.8.8 \
    jupyter lab \
    --NotebookApp.token='' \
    --NotebookApp.password=''
```

### Using a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

To exit the virtual environment, run `deactivate`.
