import trlx
from trlx.data.configs import TRLConfig
import math
from pathlib import Path


if __name__ == "__main__":
    def reward_fn(samples, prompts=None, outputs=None):
        return [s.count('s') for s in samples]


    config_path = Path(__file__).parent / "configs/ppo_gpt2.yml"
    config = TRLConfig.load_yaml(config_path)
    trlx.train(
        "lvwerra/gpt2-imdb",
        reward_fn=reward_fn,
        config=config,
    )