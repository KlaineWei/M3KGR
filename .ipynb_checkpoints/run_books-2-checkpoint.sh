# python run_base.py --model CKE --dataset Amazon_Books --config_file books.yaml
# python run_base.py --model BPR --dataset Amazon_Books --config_file books.yaml
# python run_base.py --model NNCF --dataset Amazon_Books --config_file books.yaml
# python run_base.py --model LightGCN --dataset Amazon_Books --config_file books.yaml
# python run_base.py --model SGL --dataset Amazon_Books --config_file books.yaml
python run_base.py --model KGCN --dataset Amazon_Books --config_file books.yaml
python run_base.py --model KGIN --dataset Amazon_Books --config_file books.yaml
python run_base.py --model KGAT --dataset Amazon_Books --config_file books.yaml
python run_base.py --model NGCF --dataset Amazon_Books --config_file books.yaml
# python run_base.py --model NCL --dataset Amazon_Books --config_file books.yaml