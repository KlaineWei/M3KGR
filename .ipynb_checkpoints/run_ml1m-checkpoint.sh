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

# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 98304 --phi 0.7
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 106496 --phi 0.7
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 88064 --phi 0.7
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7 --use_att False --multi_step 8750
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7 --use_att False --multi_step 4
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7 --use_image False --use_att False --multi_step 4
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7 --multi_step 4 --att_type 2
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7 --use_text False --use_att False --multi_step 4
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7 --multi_step 10 --att_type 2
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7 --multi_step 20 --att_type 2
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7 --multi_step 50 --att_type 2
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7 --multi_step 100 --att_type 2
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7 --multi_step 500 --att_type 2
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7 --multi_step 5000 --att_type 2
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7 --multi_step 5 --att_type 3
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7 --multi_step 87500 --att_type 2
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7 --multi_step 0 --att_type 2
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7 --use_att False
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7 --multi_step 5 --att_type 2 --cts_weight 0.03
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7 --multi_step 5 --att_type 2 --cts_weight 0.007
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7 --multi_step 5 --att_type 2 --cts_weight 0.005
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7 --multi_step 5 --att_type 2 --cts_weight 0.003
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7 --multi_step 5 --att_type 2 --cts_weight 0.001
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7 --multi_step 5 --att_type 2 --cts_weight 0.013
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7 --multi_step 5 --att_type 2 --cts_weight 0.015
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7 --multi_step 5 --att_type 2 --cts_weight 0.017
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7 --multi_step 5 --att_type 2 --cts_weight 0.02
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7 --multi_step 5 --att_type 2
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7 --multi_step 5 --att_type 2 --m 0.9
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7 --multi_step 5 --att_type 2 --m 0.99
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7 --multi_step 5 --att_type 2 --m 0.9999
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7 --multi_step 5 --att_type 2 --t 0.01
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7 --multi_step 5 --att_type 2 --m 0
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7 --multi_step 5 --att_type 2 --t 0.03
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7 --multi_step 5 --att_type 2 --t 0.05
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7 --multi_step 5 --att_type 2 --t 0.09
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7 --multi_step 5 --att_type 2 --t 0.1
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7 --multi_step 5 --att_type 2 --t 0.3
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7 --multi_step 5 --att_type 2 --t 0.5
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7 --multi_step 5 --att_type 2 --t 0.7
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --k 90112 --phi 0.7 --multi_step 5 --att_type 2 --t 0.9
python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --user_inter_num_interval '[20,inf)'
python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --user_inter_num_interval '[30,inf)'
python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --user_inter_num_interval '[40,inf)'
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --user_inter_num_interval '[50,inf)'
python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --user_inter_num_interval '[60,inf)'
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --user_inter_num_interval '[70,inf)'
python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --user_inter_num_interval '[80,inf)'
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --user_inter_num_interval '[90,inf)'
# python run_M3KGR.py --dataset ml-1m --config_file ml-1m.yaml --user_inter_num_interval '[100,inf)'