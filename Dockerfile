FROM nvidia/cuda:10.0-devel-ubuntu18.04

#RUN yes | unminimize


RUN apt-get update && apt-get install -y wget bzip2
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"
RUN conda config --set always_yes yes

RUN conda install pytorch==1.0.1 torchvision==0.2.2 cudatoolkit=10.0 -c pytorch
RUN conda install -c menpo opencv
RUN pip install tensorboardX scikit-image tqdm pyyaml easydict future h5py torchnet pip
RUN apt-get install unzip

COPY ./ /BF3S
RUN pip install -e /BF3S

WORKDIR /BF3S

# Test imports
RUN python -c "import scripts.train_fewshot"
RUN python -c "import scripts.test_fewshot"
RUN python -c "import scripts.test_fewshot_imagenet"
RUN python -c "import scripts.save_features_imagenet"
