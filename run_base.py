from recbole.quick_start import run_recbole
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--config_file', type=str)
parser.add_argument('--user_inter_num_interval',type=str,default="[5,inf)")
parser.add_argument('--item_inter_num_interval',type=str,default="[5,inf)")
args = parser.parse_args()
parameter_dict = {
#   'neg_sampling': None,
}

model=args.model
dataset=args.dataset
config_file_list = [args.config_file]
parameter_dict['user_inter_num_interval']=args.user_inter_num_interval
parameter_dict['item_inter_num_interval']=args.item_inter_num_interval
run_recbole(model=model, dataset=dataset, config_file_list=config_file_list,config_dict=parameter_dict)
# run_recbole(model='CKE', dataset=dataset, config_file_list=config_file_list)
# run_recbole(model='BPR', dataset=dataset, config_file_list=config_file_list)
# run_recbole(model='NNCF', dataset=dataset, config_file_list=config_file_list)
# run_recbole(model='LightGCN', dataset=dataset, config_file_list=config_file_list)
# run_recbole(model='SGL', dataset=dataset, config_file_list=config_file_list)
# run_recbole(model='KGCN', dataset=dataset, config_file_list=config_file_list)
# run_recbole(model='KGAT', dataset=dataset, config_file_list=config_file_list)