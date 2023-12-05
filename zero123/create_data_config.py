import fire
import json
import os
from glob import glob


# def create_data_config(data_dir, shards=False, out_path=""):
#     paths = []
#     if shards:
#         for shard in os.listdir(data_dir):
#             shard_dir = os.path.join(data_dir, shard)
#             if os.path.isdir(shard_dir):
#                 # paths.extend([os.path.join(shard, x) for x in os.listdir(shard_dir) if x in valid_tars])
#                 paths.extend([os.path.join(shard, x) for x in os.listdir(shard_dir)])
#     else:
#         paths = os.listdir(data_dir)
    
#     if out_path == "":
#         out_path = os.path.join(data_dir, "data_config.json")
#     with open(out_path, "w") as f:
#         json.dump(paths, f)

def create_data_config(data_dir, out_path=""):
    paths = glob(os.path.join(data_dir, "*.json"))
    valid_paths = []
    for path in paths:
        with open(path) as f:
            j = json.load(f)
        if j['objaverse']['license'] not in ('by-nc-sa', 'by-nc', 'by-sa'):
            valid_paths.append(path)
    print(f"Valid {len(valid_paths)} out of {len(paths)}")
    
    if out_path == "":
        out_path = os.path.join(data_dir, "data_config.json")
    with open(out_path, "w") as f:
        valid_objects = [os.path.basename(vpath).split('.')[0] for vpath in valid_paths]
        untarred_objects_old = os.listdir('/scratch/objaverse_untar_2')
        filtered_present_objects = list(set(valid_objects).intersection(set(untarred_objects_old)))
        print(f'Final dataset size: {len(filtered_present_objects)} (out of {len(untarred_objects_old)})')
        json.dump(filtered_present_objects, f)


if __name__ == '__main__':
  fire.Fire(create_data_config)