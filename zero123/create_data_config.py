import fire
import json
import os


def create_data_config(data_dir, shards=False, out_path=""):
    paths = []
    if shards:
        for shard in os.listdir(data_dir):
            shard_dir = os.path.join(data_dir, shard)
            if os.path.isdir(shard_dir):
                # paths.extend([os.path.join(shard, x) for x in os.listdir(shard_dir) if x in valid_tars])
                paths.extend([os.path.join(shard, x) for x in os.listdir(shard_dir)])
    else:
        paths = os.listdir(data_dir)
    
    if out_path == "":
        out_path = os.path.join(data_dir, "data_config.json")
    with open(out_path, "w") as f:
        json.dump(paths, f)


if __name__ == '__main__':
  fire.Fire(create_data_config)