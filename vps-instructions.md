# VPS instructions

```bash
sudo apt update && sudo apt upgrade -y && sudo apt install -y \
    python3-pip \
    python3-venv \
    python-is-python3 \
    tmux \
    git \
    build-essential

git clone https://github.com/gusalbukrk/tabpfn-gbdt.git

cd tabpfn-gbdt

python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

wandb login

# tmux new -s training
```

- on local machine
    - `scp -r /home/gusalbukrk/Downloads/Kaggle/taiwanese_bankruptcy_prediction.npz root@49.12.205.70:/root/tabpfn-gbdt`
    - `scp -r /home/gusalbukrk/Downloads/Kaggle/website_phishing.npz root@168.119.253.34:/root/tabpfn-gbdt/archives`
    - `scp -r /home/gusalbukrk/Downloads/Kaggle/vps-2 root@204.168.155.1:/root/tabpfn-gbdt`
    - `scp -r /home/gusalbukrk/Downloads/Kaggle/vps-3 root@168.119.253.34:/root/tabpfn-gbdt`