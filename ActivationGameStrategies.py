from ActivationGame import ActivationGameWorld
import numpy as np
from OracleStrategy import oracle_score


def DepthFirst(world: ActivationGameWorld, render=False):
    """
    A depth-first search strategy for the Activation Game.
    """

    # Get all the characters that are enchanted at the start of the search
    # Store them in a list against key 0 in the stack dictionary
    stack = {0: [char for char in world.characters if char.isEnchanted]}
 
    while not world.is_solved():
        # Get the list of most recently added characters
        level = max(stack.keys())
        chars = stack[level]
        num_to_obs = [count_num_to_observe(char,world) for char in stack[level]]

        # Select the character that can observe the maximum number of new cells
        max_index = np.argmax(num_to_obs)
        char = chars.pop(max_index)
        num = num_to_obs.pop(max_index)

        # Instead of actively removing chars from the stack
        # If we are at a level and no chars have any new cells to observe
        # we can just delete that level from the stack and go back to try a lower level
        if num == 0:
            del stack[level]
            continue

        # If we've taken the last character at this level, remove the level
        if len(chars) == 0:
            del stack[level]

        if not char.isEnchanted:
            # Find the possible enchanters
            enchanters = [c for c in world.characters if char in c.couldEnchant and c.isEnchanted]
            # Activate the character using the first enchanter for now (I believe it's unimportant)
            world.step([enchanters[0].location, char.location])
            if render:
                print(f"Step {world.nsteps}: Activated {char.chartype} at {char.location} using enchanter at {enchanters[0].location}, then sensed")

        # Register who is currently observed so we can easily find the newly observed chars
        currently_observed = [c for c in world.characters if c.isObserved]
        # Do the sensing action
        world.step([char.location, "Sense"])
        if render:
            world.render()
        # Work out who is newly observed
        now_observed = [c for c in world.characters if c.isObserved]
        newly_observed = [c for c in now_observed if c not in currently_observed]
        # Add the newly observed characters to the stack
        if len(newly_observed)>0:
            stack[level + 1] = newly_observed

    return world.nsteps


def count_num_to_observe(char, world):
    """
    Count the number of new cells that could be observed by the character
    """
    min_x = max(0, char.location[0] - char.range)
    max_x = min(world.gridwidth - 1, char.location[0] + char.range)
    min_y = max(0, char.location[1] - char.range)
    max_y = min(world.gridheight - 1, char.location[1] + char.range)

    count = sum([world.obs_mask[x][y] == 0 for x in range(min_x, max_x + 1) for y in range(min_y, max_y + 1)])
    
    return count


def linear_features_strategy(world: ActivationGameWorld,weights=[1,1,1]):
    """
    Linear features based strategy
    The features are:
    - number of newly observed cells
    - number of unobserved cells that could now be observed on next step
    - length of chain that is discareded by taking the action
    """
    pass


def eval_strategy(strategy=DepthFirst, nsamples=10):
    """
    Evaluate a strategy over multiple samples of the world
    """
    nsteps = np.zeros(nsamples)
    oracle = np.zeros(nsamples)
    initial_seed = np.random.randint(0,10000)
    for ii in range(nsamples):
        world = ActivationGameWorld(seed=initial_seed+ii,silent=True)
        nsteps[ii] = strategy(world)
        world = ActivationGameWorld(seed=initial_seed+ii,silent=True)
        oracle[ii] = oracle_score(world)
    return(nsteps,oracle)  

if __name__ == "__main__":
    world = ActivationGameWorld()
    nsteps = DepthFirst(world,render=True)
    #world.render()
    print(f"Solved in {nsteps} steps")

#    nsteps, oracle = eval_strategy()
#    print(f"Average number of steps over samples: {np.mean(nsteps)}")
#    print(f"Maximum number of steps over samples: {np.max(nsteps)}")
#    print(f"Average oracle score over samples: {np.mean(oracle)}")
#    print(f"Average difference between strategy and oracle: {np.mean(nsteps - oracle)}")
