<center>
<h2>Peptide retention time prediction</h2>
</center>

____

<h2> Getting started </h2>

To recreate the computational environment follow one of the following setup instructions.

**Advanced setup:** 

Follow these instructions to obtain the full functionality. To set up the environment, you will need to install `docker` and `docker-compose`. By running

```
> docker-compose up -d
```

Several services will be launched:

- A jupyter server with a fully functional environment to run the notebooks and scripts
- A MLflow server for experiment tracking and artifact logging
- S3 server for object storage

**Light-weight setup:**

If you wish to start only the jupyter service, you first need to build the image:

```
> docker build --tag <your_image_name:tag> ./docker/jupyter/
```
And subsequently start the container:
```
> docker container run -p 8888:8888 -v ./:/work/ -n <your_container_name> <your_image_name:tag>
```
Remark: Append `jupyter-lab` to the above command if you wish to start jupyter-lab instead of jupyter-notebook.


**Basic setup:**

If you do not wish to install docker, you can directly run the notebooks in the `./notebooks/` directory. The environment file to recreate the environments are in `./docker/jupyter` (environment.yml for `conda`, requirements.txt for `pip`)


The services can be accessed at

- jupyter: http://localhost:8888 (access token: `peptide`)
- mlflow: http://localhost:5000 
- S3 storage: http://localhost:9000 (user: `mlflow_user`, password: `mlflow_passwd`)


If you chose a basic or lightweight setup, some of the code might not be fully functional, e.g. the model logging via mlflow.


<h2> Overview </h2>

The notebooks `./notebooks/` contain exploratory code and results. The naming scheme is 

<center> VERSIONNUMBER-DESCRIPTION.ipynb </center>

which reflects the order and purpose of each notebook. The conclusions of these explorations are then distilled into re-usable code in the `./src` directory, which will then be used in later versions of notebooks.

<h2> Project structure </h2>

- `./data`: contains the raw data
- `./docker`: Dockerfiles and other environment files
- `./documentation`: figures
- `./models`: exported models (not synced)
- `./notebooks`: jupyter notebooks that were used during exploration
- `./src`: re-usable and modular source code


**Project structure**:
```
├── data
├── docker
│   ├── jupyter
│   └── mlflow
├── documentation
│   └── figures
├── models
├── notebooks
└── src
    ├── data
    ├── models
    └── util
```