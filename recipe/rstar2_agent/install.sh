SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") &>/dev/null && pwd -P)
WORKSPACE=$SCRIPT_DIR/../../..

cd $WORKSPACE

sudo apt-get update -y && sudo apt-get install redis -y
pip install -r $SCRIPT_DIR/requirements.txt

git clone https://github.com/0xWJ/code-judge.git
cd code-judge
pip install -e .
pip install -r requirements.txt
cd ..
