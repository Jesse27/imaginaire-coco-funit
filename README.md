# Imaginaire - COCO-FUNIT TUTORIAL
A step by step tutorial to train [COCO-FUNIT](https://arxiv.org/abs/2007.07431) using the animal faces dataset. Forked from the [imaginaire](https://github.com/NVlabs/imaginaire) project. The original readme can be found [here](OLD_README.md)

This tutorial was tested on Pop! OS 22.04

## License

Imaginaire is released under [NVIDIA Software license](LICENSE.md).
For commercial use, please consult [NVIDIA Research Inquiries](https://www.nvidia.com/en-us/research/inquiries/).

## Linux Installation

### Installing Dependency's
1. **Install Docker**

    Install docker engine using the [official guide](https://docs.docker.com/engine/install). The link for each platform is shown under the server heading.

    After installation you should add your user to the docker group :
    ```bash
    sudo groupadd docker
    sudo usermod -aG docker $USER
    ```
    Then restart so that your group membership is re-evaluated.

    You can verify that docker has been set up correctly by running the `hello-world` image :
    ```
    docker run hello-world
    ```

2. **Install NVIDIA Container Toolkit**

    Currently at the time of writing I could not get the [official installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) method to work. 
    The following commands were performed to install NVIDIA Container Toolkit version 1.10.0-1 :

    * Ubuntu:
        ```bash
            distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
            && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
            && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
                sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
                sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        ```
        ```bash
            sudo apt-get update
        ```
        ```bash
            sudo apt-get install nvidia-container-toolkit=1.10.0-1
            sudo apt-get install libnvidia-container1=1.10.0-1
            sudo apt-get install libnvidia-container-tools=1.10.0-1
        ```

    * Pop! os:

        Because the nvidia-container-toolkit is only supported by a couple of distribution, you have some manipulations to do to be able to install it on Pop! OS

        ```bash
            distribution="ubuntu22.04" \
            && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
            && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
        ```
        ```bash
            sudo apt-get update
        ```
        ```bash
            sudo apt-get install nvidia-container-toolkit=1.10.0-1
            sudo apt-get install libnvidia-container1=1.10.0-1
            sudo apt-get install libnvidia-container-tools=1.10.0-1
        ```

    * Testing:

        A working setup can be tested by running a base CUDA container:
        ```
        docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
        ```
        This should result in a console output shown below:
        ```
            +-----------------------------------------------------------------------------+
        | NVIDIA-SMI 450.51.06    Driver Version: 450.51.06    CUDA Version: 11.0     |
        |-------------------------------+----------------------+----------------------+
        | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
        | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
        |                               |                      |               MIG M. |
        |===============================+======================+======================|
        |   0  Tesla T4            On   | 00000000:00:1E.0 Off |                    0 |
        | N/A   34C    P8     9W /  70W |      0MiB / 15109MiB |      0%      Default |
        |                               |                      |                  N/A |
        +-------------------------------+----------------------+----------------------+

        +-----------------------------------------------------------------------------+
        | Processes:                                                                  |
        |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
        |        ID   ID                                                   Usage      |
        |=============================================================================|
        |  No running processes found                                                 |
        +-----------------------------------------------------------------------------+
        ```

### Setup COCO-FUNIT
3. **Clone the repo**
    ```bash
    git clone https://github.com/Jesse27/imaginaire-coco-funit.git
    ```

4. **Build docker image**

    After cloning navigate to the `path/to/imaginaire-coco-funit/` directory and run the build script :

    *Note* all scripts should be run from this directory.

    ```bash
    bash scripts/build_docker.sh 21.06
    ```

5. **Start docker image**
    ```bash
    bash scripts/start_local_docker.sh 21.06
    ```
    This should result in a console output shown below where *0f388ec0d8b2* is the docker CONTAINER ID:
    ```
    root@0f388ec0d8b2:/workspace/coco-funit#
    ```
    If you have run the code previously you may receive the error below. 
    ```
    docker: Error response from daemon: Conflict. The container name "/coco-funit" is already in use by container ...
    ```
    This can be solved by running the following commands to stop and remove the existing container.
    ```bash
    docker stop coco-funit
    docker rm coco-funit
    ```
    If you wish to keep the existing container you can use the following command to open it's terminal shell
    ```bash
    docker exec -it coco-funit /bin/bash
    ```

## Training on the animal faces dataset
6. **Downloading the data**

    The example animal-faces dataset can be downloaded using the `download_dataset.py` script. 
    This should be **run in the docker container** from the `/workspace/coco-funit/` directory
    ```bash
    python scripts/download_dataset.py --dataset animal_faces
    ```

7. **Build the lmdbs**
    ```python
    for f in train train_all val; do
    python scripts/build_lmdb.py \
    --config  configs/projects/coco_funit/animal_faces/base64_bs8_class119.yaml \
    --data_root dataset/animal_faces/${f} \
    --output_root projects/coco_funit/data/lmdb/training/animal_faces/${f} \
    --overwrite
    done
    ```

8. **Start Training**

    `--nproc_per_node=1` configures the number of GPU's used in training, it is set to 1 by default. 

    Other configuration parameters are found in `configs/projects/coco_funit/animal_faces/base64_bs8_class119.yaml`

    ```bash
    python -m torch.distributed.launch --nproc_per_node=1 train.py --config configs/projects/coco_funit/animal_faces/base64_bs8_class119.yaml --logdir logs/projects/coco_funit/animal_faces/base64_bs8_class119.yaml
    ```

    *Note* that you may encounter a git config issue. This can be solved by running the command displayed in terminal then re-running the train command:
    ```bash
    git config --global --add safe.directory path/to/imaginaire-coco-funit/
    ```

9. **Output**

    The output contains images, TensorBoard logs and model checkpoints.

    The training output is found in `path/to/imaginaire-coco-funit/logs/projects/coco_funit/animal_faces/base64_bs8_class119.yaml/`
    
    The number of rows shown in the output image is equal to the batch size per GPU.

10. **TensorBoard**

    TensorBoard logs should be opened within the docker container. To access the docker container while the model is training open another terminal and run the following command:
    
    *Note* that *coco-funit* is the name of the docker container, this can be found by running `docker container ls`. It is set to *coco-funit* by default.
    ```bash
    docker exec -it coco-funit /bin/bash
    ```
    This should result in a console output shown below where *0f388ec0d8b2* is the docker CONTAINER ID:
    ```
    root@0f388ec0d8b2:/workspace/coco-funit#
    ```
    To start TensorBoard run the following command in the docker container: 
    ```bash
    tensorboard --logdir logs/projects/coco_funit/animal_faces/base64_bs8_class119.yaml/tensorboard
    ```
    TensorBoard can then be opened at `0.0.0.0:8083` on the local machine.
    