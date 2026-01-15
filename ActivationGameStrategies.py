from ActivationGame import ActivationGameWorld
import numpy as np

def DepthFirst(world: ActivationGameWorld):
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

        # Register who is currently observed so we can easily find the newly observed chars
        currently_observed = [c for c in world.characters if c.isObserved]
        # Do the sensing action
        world.step([char.location, "Sense"])
        # Work out who is newly observed
        now_observed = [c for c in world.characters if c.isObserved]
        newly_observed = [c for c in now_observed if c not in currently_observed]
        # Add the newly observed characters to the stack
        if len(newly_observed)>0:
            stack[level + 1] = newly_observed

 


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


if __name__ == "__main__":
    world = ActivationGameWorld()
    DepthFirst(world)
    world.render()
    print(f"Solved in {world.nsteps} steps")