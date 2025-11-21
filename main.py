import json

import yaml


def main():
    with open("checkpoints/dqn/20251120-2326/config.json") as f:
        config = json.load(f)
    print(config)
    with open("checkpoints/dqn/20251120-2326/config.yaml", "w") as f:
        yaml.dump(config, f)


if __name__ == "__main__":
    main()
