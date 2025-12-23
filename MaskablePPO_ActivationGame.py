print("* Importing standard libs")
from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
# This is a drop-in replacement for EvalCallback
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

print("* Importing game environment")
from ActivationGame import ActivationGameEnv, ActivationGameCNN

# Was having trouble within MaskablePPO, presumably due to rounding error
# This removes the error, but I have not checked that it was only rounding error that was the problem
from torch.distributions import Distribution
Distribution.set_default_validate_args(False)

print("* Setting up env and model")
policy_kwargs = dict(
    features_extractor_class=ActivationGameCNN,
    features_extractor_kwargs=dict(features_dim=128),
)

#env = InvalidActionEnvDiscrete(dim=80, n_invalid_actions=60)
env = ActivationGameEnv(gridsize=10)
model = MaskablePPO("CnnPolicy", env, gamma=0.9, verbose=1, policy_kwargs=policy_kwargs)

print("* Training model")
model.learn(5_000)

print("* Evaluating model")
evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=90, warn=False)

print("* Saving and loading model")
model.save("ppo_mask")
del model # remove to demonstrate saving and loading

model = MaskablePPO.load("ppo_mask")

print("* Running model")
obs, _ = env.reset()
while not terminated:
    # Retrieve current action mask
    action_masks = get_action_masks(env)
    action, _states = model.predict(obs, action_masks=action_masks)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    input("Press Enter to continue...")