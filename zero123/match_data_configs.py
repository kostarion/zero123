import fire
import json
import os
from glob import glob


def create_data_config(data_dir, out_path=""):
    paths = glob(os.path.join(data_dir, "*.json"))
    valid_paths = []
    for path in paths:
        with open(path) as f:
            j = json.load(f)
        if j['objaverse']['license'] not in ('by-nc-sa', 'by-nc'):
            valid_paths.append(path)
    print(f"Valid {len(valid_paths)} out of {len(paths)}")
    
    if out_path == "":
        out_path = os.path.join(data_dir, "data_config.json")
    with open(out_path, "w") as f:
        valid_objects = [os.path.basename(vpath).split('.')[0] for vpath in valid_paths]
        json.dump(valid_objects, f)


if __name__ == '__main__':
  fire.Fire(create_data_config)