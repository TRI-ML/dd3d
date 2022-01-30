from collections import OrderedDict


CATEGORIES = [  # TODO: List categories which have instance in the train set
	'human--person',
	'human--person--group',
	'human--rider--bicyclist',
	'object--vehicle--bicycle',
	'object--vehicle--bus',
	'object--vehicle--car',
	'object--vehicle--motorcycle',
	'object--vehicle--truck',
	'object--vehicle--group',
]


CATEGORY_IDS = OrderedDict({key: idx for idx, key in enumerate(CATEGORIES)})
