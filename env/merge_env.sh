mamba create -y -n frigid-ice python=3.10
mamba install -y rdkit openbabel cairosvg gcc_linux-64 gxx_linux-64 sysroot_linux-64
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install torchdata==0.8.0
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html

pip install lightning==2.5.1 transformers==4.56.2 hydra-core==1.3.2 omegaconf==2.3.0 \
    pandas==2.1.0 numpy==1.26.4 datasets==2.18.0 wandb==0.13.5 \
    safe-mol==0.1.13 pytdc==0.4.1 easydict openbabel-wheel==3.1.1.22 \
    bionemo-moco==0.0.2.1 einops==0.7.0 jaxtyping>=0.2.34

pip install -r ms-pred-dev/requirements.txt
pip install -e ./frigid --no-deps
pip install -e ./ms-pred-dev

bash frigid/env/fix_safe_imports.sh

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
