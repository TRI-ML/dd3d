import argparse
import json
from tridet.data.datasets.nuscenes.build import NuscenesDataset, build_nuscenes_dataset


def save_json(dictionary, out_path):
	"""
	Saves dictionary into .json format
	"""
	with open(out_path, "w") as fd:
		json.dump(dictionary, fd, indent=4)


def pares_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('dataset_dir', type=str, help='Path to dataset')
	parser.add_argument('dataset_name', type=str, help='Dataset name')

	return parser.parse_args()


if __name__ == '__main__':
	args = pares_args()

	dataset_dicts = build_nuscenes_dataset(name=args.dataset_name, root_dir=args.dataset_dir)
	for idx, dataset_dict in enumerate(dataset_dicts):
		save_json(dataset_dict, f'dataset_dict_{idx}.json')

	print(dataset_dicts)
