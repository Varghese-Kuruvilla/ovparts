import json

MAPPING_PATH = "/home/varghese/OV_PARTS/partimagenet_ood/imagenet_class_index.json"

with open(MAPPING_PATH, 'r') as f:
    raw_mapping = json.load(f)
    synset_to_name = {v[0]: v[1].replace('_', ' ') for v in raw_mapping.values()}

name_to_synset = {v: k for k, v in synset_to_name.items()}

def get_synset(name):
    if name in name_to_synset:
        return name_to_synset[name]
    for k, v in synset_to_name.items():
        if v.lower() == name.lower():
            return k
    return None

base_categories_names = [
    "tiger", "giant panda", "leopard", "gazelle", "green mamba", 
    "green lizard", "Komodo dragon", "tree frog", "yawl", "pirate", 
    "barracouta", "goldfish", "killer whale", "albatross", "goose", 
    "garbage truck", "minibus", "ambulance", "mountain bike", "moped", 
    "gorilla", "orangutan", "beer bottle", "water bottle", "warplane"
]

val_synsets = [
    "n01484850", "n01614925", "n01685808", "n01689811", "n01748264",
    "n02009229", "n02099601", "n02109525", "n02125311", "n02134084",
    "n02442845", "n02483708", "n02490219", "n02514041", "n02814533",
    "n02930766", "n04483307", "n04509417", "n04557648"
]

val_names = {s: synset_to_name.get(s, "UNKNOWN") for s in val_synsets}
print("Classes in val folder:")
for s, n in val_names.items():
    print(f"  {s} -> {n}")

print("\nBase categories mapping:")
for name in base_categories_names:
    s = get_synset(name)
    if s:
        print(f"  {name} -> {s}")
    else:
        print(f"  {name} -> NOT FOUND")

