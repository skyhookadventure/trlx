import trlx
from trlx.data.configs import TRLConfig
import math

if __name__ == "__main__":
    def reward_fn(list_of_str):
        return [s.count('s')/math.sqrt(len(s)) for s in list_of_str]


    trlx.train(
        "lvwerra/gpt2-imdb",
        reward_fn=reward_fn,
        config=TRLConfig.load_yaml("configs/ppo_gpt2.yml"),
    )