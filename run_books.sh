# python run_base.py --model CKE --dataset Amazon_Books --config_file books.yaml
# python run_base.py --model BPR --dataset Amazon_Books --config_file books.yaml
# python run_base.py --model NNCF --dataset Amazon_Books --config_file books.yaml
# python run_base.py --model LightGCN --dataset Amazon_Books --config_file books.yaml
# python run_base.py --model SGL --dataset Amazon_Books --config_file books.yaml
# python run_base.py --model KGCN --dataset Amazon_Books --config_file books.yaml
# python run_base.py --model KGIN --dataset Amazon_Books --config_file books.yaml
# python run_base.py --model KGAT --dataset Amazon_Books --config_file books.yaml
# python run_base.py --model NGCF --dataset Amazon_Books --config_file books.yaml
# python run_base.py --model NCL --dataset Amazon_Books --config_file books.yaml

# python run_M3KGR.py --k 73728
# python run_M3KGR.py --k 81920
# python run_M3KGR.py --k 90112
python run_M3KGR.py --k 73728 --dataset Amazon_Books --config_file books.yaml --phi 0.95
python run_M3KGR.py --k 73728 --dataset Amazon_Books --config_file books.yaml --phi 0.9
python run_M3KGR.py --k 73728 --dataset Amazon_Books --config_file books.yaml --phi 0.85
python run_M3KGR.py --k 73728 --dataset Amazon_Books --config_file books.yaml --phi 0.8
python run_M3KGR.py --k 73728 --dataset Amazon_Books --config_file books.yaml --phi 0.75
python run_M3KGR.py --k 73728 --dataset Amazon_Books --config_file books.yaml --phi 0.65
python run_M3KGR.py --k 73728 --dataset Amazon_Books --config_file books.yaml --phi 0.6
# python run_M3KGR.py --k 106496 --dataset Amazon_Books --config_file books.yaml