from recbole.quick_start import run_recbole
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--phi', type=float,default=0.7)
parser.add_argument('--k', type=int,default=90112)
parser.add_argument('--dataset', type=str)
parser.add_argument('--config_file', type=str)
parser.add_argument('--use_image', default='True')
parser.add_argument('--use_text', default='True')
parser.add_argument('--use_att', default='True')
parser.add_argument('--multi_step', type=int, default=5)
parser.add_argument('--att_type',type=int,default=2)
parser.add_argument('--cts_weight',type=float,default=0.015)
parser.add_argument('--m',type=float,default=0.999)
parser.add_argument('--t',type=float,default=0.07)
parser.add_argument('--user_inter_num_interval',type=str,default="[5,inf)")
args = parser.parse_args()
parameter_dict = {
#   'neg_sampling': None,
}
if args.phi is not None:
    parameter_dict['phi']=args.phi
if args.k is not None:
    parameter_dict['k']=args.k
parameter_dict['use_image']=args.use_image
parameter_dict['use_text']=args.use_text
parameter_dict['use_att']=args.use_att
parameter_dict['multi_step']=args.multi_step
parameter_dict['att_type']=args.att_type
parameter_dict['cts_weight']=args.cts_weight
parameter_dict['user_inter_num_interval']=args.user_inter_num_interval
if args.m is not None:
    parameter_dict['m']=args.m
if args.t is not None:
    parameter_dict['t']=args.t
dataset=args.dataset
config_file_list = [args.config_file]
# config_file_list = ['ml-1m.yaml']
# config_file_list = ['books.yaml']
# config_file_list = ['ml-10m.yaml']
# run_recbole(model='M3KGR', dataset='ml-1m', config_file_list=config_file_list, config_dict=parameter_dict)
run_recbole(model='M3KGR', dataset=dataset, config_file_list=config_file_list, config_dict=parameter_dict)
# run_recbole(model='M3KGR', dataset='ml-10m', config_file_list=config_file_list, config_dict=parameter_dict)