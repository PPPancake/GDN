################################################################################
# 暂不支持multi_run(multi_run和grid未完成) #
################################################################################
import argparse
import time, datetime
import torch
import numpy as np
import logging
import yaml
from collections import defaultdict, OrderedDict

from model_handler import ModelHandler


timestamp = time.time()
timestamp = datetime.datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H:%M:%S')
logging.basicConfig(filename='result.log', level=logging.INFO)
logging.info(timestamp)

# 解析命令行参数
def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-config', '--config', required=True, type=str, help='path to the config file')
	parser.add_argument('--multi_run', action='store_true', help='flag: multi run')
	args = vars(parser.parse_args())
	return args

# 加载yaml配置文件
def get_config(config_path="config.yml"):
	with open(config_path, "r") as setting:
		config = yaml.load(setting, Loader=yaml.FullLoader)
	return config

# 生成所有超参数的排列组合
def grid(kwargs):
	class MncDc:
		def __init__(self, a):
			self.a = a
		def __call__(self):
			return self.a
	
	####################

	sin = OrderedDict({k: v for k, v in kwargs.items() if isinstance(v, list)})
	for k, v in sin.items():
		copy_v = []
		for e in v:
			copy_v.append(MncDc(e) if isinstance(e, tuple) else e)
		sin[k] = copy_v
	
	grd = np.array(np.meshgrid(*sin.values()), dtype=object).T.reshape(-1, len(sin.values()))
	#####################

# 设计随机种子
def set_random_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)

# 在log文件中打印配置信息
def print_config(config):
	logging.info("**************** MODEL CONFIGURATION ****************")
	for key in sorted(config.keys()):
		val = config[key]
		keystr = "{}".format(key) + (" " * (24 - len(key)))
		logging.info("{} -->   {}".format(keystr, val))
	logging.info("**************** MODEL CONFIGURATION ****************")

# 单组超参数时运行
def main(config):
	print_config(config)
	set_random_seed(config['seed'])
	model = ModelHandler(config)
	f1_mac_test, auc_test, gmean_test = model.train()
	print("F1-Macro: {}".format(f1_mac_test))
	print("AUC: {}".format(auc_test))
	print("G-Mean: {}".format(gmean_test))

# 多组超参数时运行
def multi_run_main(config):
	print_config(config)
	hyperparams = []
	for k, v in config.items():
		if isinstance(v, list):
			hyperparams.append(k)
	
	f1_list, f1_1_list, f1_0_list, auc_list, gmean_list = [], [], [], [], []
	configs = grid(config)
	# for i, cnf in enumerate(configs):
	#	 print("Running {}:\n".format(i))
	#	 for k in hyperparams:
	#		 cnf['save_dir']


if __name__ == '__main__':
	print(torch.cuda.is_available())
	cfg = get_args()
	config = get_config(cfg['config'])
	if cfg['multi_run']: 
		multi_run_main(config)
	else:
		main(config)