

python3.9 -m venv .venv
source .venv/bin/activate
which python

pip install -e .

pip install torch
pip install black
pip install scipy

git clone https://github.com/saleml/gym-minigrid.git
cd gym-minigrid
git status
git switch minigrid-no-warning
git status
pip install -e .
cd ../