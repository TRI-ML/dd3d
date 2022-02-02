"""
File with specification of Metropolis Dataset categories and their assigned IDs
"""
from collections import OrderedDict

CATEGORIES = [  # Note: List of categories which have instances in the TRAIN set
	"construction--flat--crosswalk-plain",  # Crosswalk Plain
	"human--person",  # Person
	"human--person--group",  # Person Group
	"human--rider--bicyclist",  # Bicyclist
	"human--rider--other-rider",  # Other rider
	"marking--discrete--arrow--left",  # Lane Marking - Arrow (Left)
	"marking--discrete--arrow--other",  # Lane Marking - Arrow (Other)
	"marking--discrete--arrow--right",  # Lane Marking - Arrow (Right)
	"marking--discrete--arrow--straight",  # Lane Marking - Arrow (Straight)
	"marking--discrete--crosswalk-zebra",  # Lane marking - crosswalk
	"marking--discrete--other-marking",  # Lane marking - Other
	"marking--discrete--text",  # Lane marking - text
	"object--banner",  # Banner
	"object--bench",  # Bench
	"object--bike-rack",  # Bike rack
	"object--catch-basin",  # Catch basin
	"object--cctv-camera",  # CCTV camera
	"object--fire-hydrant",  # Fire hydrant
	"object--junction-box",  # Junction box
	"object--mailbox",  # Mailbox
	"object--manhole",  # Manhole
	"object--sign--advertisement",  # Signage - Advertisement
	"object--sign--information",  # Signage - Information
	"object--sign--other",  # Signage - Other
	"object--sign--store",  # Signage - Store
	"object--street-light",  # Street light
	"object--support--pole",  # Pole
	"object--support--traffic-sign-frame",  # Traffic sign frame
	"object--support--utility-pole",  # Utility pole
	"object--traffic-light",  # Traffic Light
	"object--traffic-sign",  # Traffic sign
	"object--trash-can",  # Trash can
	"object--tunnel-light",  # Tunnel Light
	"object--tunnel-light--group",  # Tunnel Light Group
	"object--vehicle--bicycle",  # Bicycle
	"object--vehicle--bus",  # Bus
	"object--vehicle--car",  # Car
	"object--vehicle--group",  # Vehicle Group
	"object--vehicle--motorcycle",  # Motorcycle
	"object--vehicle--other-vehicle",  # Other vehicle
	"object--vehicle--truck",  # Truck
	"object--vehicle--wheeled-slow",  # Wheeled slow vehicle
]
# Small subset of categories. To use this - uncomment this part and comment previous.
# TODO: Make it as config parameter
# CATEGORIES = [  # Note: List of categories which have instances in the TRAIN set
# 	"object--banner",  # Banner
# 	"object--bench",  # Bench
# 	"object--bike-rack",  # Bike rack
# 	"object--catch-basin",  # Catch basin
# 	"object--cctv-camera",  # CCTV camera
# 	"object--fire-hydrant",  # Fire hydrant
# 	"object--junction-box",  # Junction box
# 	"object--mailbox",  # Mailbox
# 	"object--street-light",  # Street light
# 	"object--support--pole",  # Pole
# 	"object--support--traffic-sign-frame",  # Traffic sign frame
# 	"object--support--utility-pole",  # Utility pole
# 	"object--traffic-light",  # Traffic Light
# 	"object--trash-can",  # Trash can
# 	"object--vehicle--bicycle",  # Bicycle
# 	"object--vehicle--bus",  # Bus
# 	"object--vehicle--car",  # Car
# 	"object--vehicle--truck",  # Truck
# ]


CATEGORY_IDS = OrderedDict({key: idx for idx, key in enumerate(CATEGORIES)})
