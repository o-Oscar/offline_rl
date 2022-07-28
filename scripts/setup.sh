

python3.9 -m venv .venv
source .venv/bin/activate
which python

pip install -e .

git clone https://github.com/saleml/gym-minigrid.git
cd gym-minigrid
git switch minigrid-no-warning
pip install -e .
cd ../