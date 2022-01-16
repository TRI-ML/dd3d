import argparse
import json
# from tridet.data.datasets.nuscenes.build import build_nuscenes_dataset
from tridet.data.datasets.metropolis.build import build_metropolis_dataset


def save_json(dictionary, out_path):
	"""
	Saves dictionary into .json format
	"""
	with open(out_path, "w") as fd:
		json.dump(dictionary, fd, indent=4)


def pares_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('dataset_dir', type=str, help='Root directory to dataset')
	parser.add_argument('dataset_name', type=str, help='Dataset name (train, test, etc.)')

	return parser.parse_args()


if __name__ == '__main__':
	args = pares_args()
	if 'nuscenes' in args.dataset_dir.lower():
		dataset_dicts = build_nuscenes_dataset(name=args.dataset_name, root_dir=args.dataset_dir)
		print(dataset_dicts[0])
	elif 'metropolis' in args.dataset_dir.lower():
		dataset_dicts = build_metropolis_dataset(name=args.dataset_name, root_dir=args.dataset_dir)
		for idx in range(2):
			print(dataset_dicts[idx])
	else:
		raise f'Unknown dataset with root dir: {args.dataset_dir}'
