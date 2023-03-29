FROM ubuntu:20.04

ENV SHELL=/bin/bash

RUN apt-get update 
RUN apt-get install -y git wget dumb-init

### Install conda
RUN wget --quiet \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -u -b -p /opt/conda \
    && rm ~/miniconda.sh \
    && /opt/conda/bin/conda clean -tipsy \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && mkdir /root/.jupyter \
    && printf "c.NotebookApp.allow_root = True\nc.NotebookApp.open_browser = False\nc.NotebookApp.ip = \"0.0.0.0\"\nc.NotebookApp.port = 8888" >> /root/.jupyter/jupyter_notebook_config.py \
    && printf ". /opt/conda/etc/profile.d/conda.sh" >> /etc/skel/.bashrc
ENV PATH=/opt/conda/bin:$PATH

COPY environment.yml .
# # COPY docker/jupyter/requirements.txt .

RUN conda env create --file environment.yml \
    && conda init bash \
    && conda install jupyter \
    && . /opt/conda/etc/profile.d/conda.sh \
    && conda activate peptide \
    && python -m ipykernel install --user --name=peptide \
    && printf "conda activate peptide" >> /root/.bashrc

EXPOSE 8888
ENTRYPOINT ["/usr/bin/dumb-init", "--"] 
CMD ["bash", "-c", "jupyter-notebook"]