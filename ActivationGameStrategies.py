from ActivationGame import ActivationGameWorld
import numpy as np
from OracleStrategy import oracle_score
import networkx as nx


def DepthFirst(world: ActivationGameWorld, render=False, **kwargs):
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

def nonmyopic_num_to_observe(char, world, discount_factor=1.0):
    """
    Count the number of new cells that could be observed by the character
    using a non-myopic strategy up to a given depth with discounting
    """
    loc = char.location
    obs_range = char.range

    new_observees = [[x,y] for x in range(max(0, loc[0] - obs_range), min(world.gridwidth - 1, loc[0] + obs_range) + 1)
                        for y in range(max(0, loc[1] - obs_range), min(world.gridheight - 1, loc[1] + obs_range) + 1)
                        if world.obs_mask[x][y] == 0]
    
    discount = 1.0
    num_new_observees = len(new_observees)
    score = discount*num_new_observees
    observees = new_observees.copy()
 
    # Now fix obs_range to be the longest range in the game - this is somewhat arbitrary
    obs_range = max([c.range for c in world.characters])

    while num_new_observees > 0:
         # Add all cells observable from those discovered the previous step
        prev_new_observees = new_observees.copy()
        new_observees = []
        for loc in prev_new_observees:
            new_observees.extend(
                [[x,y] for x in range(max(0, loc[0] - obs_range), min(world.gridwidth - 1, loc[0] + obs_range) + 1)
                        for y in range(max(0, loc[1] - obs_range), min(world.gridheight - 1, loc[1] + obs_range) + 1)
                        if world.obs_mask[x][y] == 0
                        and [x,y] not in observees and [x,y] not in new_observees]
                )
    
        num_new_observees = len(new_observees)
        discount *= discount_factor
        score += discount*(num_new_observees)
       
        observees.extend(new_observees)

    return score
        

def linear_features_strategy(world: ActivationGameWorld,weights=[1,-1,-5],myopic=True,render=False,max_steps=1000,discount_factor=0.5,**kwargs):
    """
    Linear features based strategy
    The features are:
    - number of newly observed cells that could be sensed from the character's location
    - length of chain that is discarded by taking the action
    - the number of actions required to enchant the character
    The weights are given by the weights parameter
    """

    stack = [char for char in world.characters if char.isEnchanted]
    if len(stack) > 1:
        print("Warning: multiple enchanted characters at start of linear features strategy")
    
    while not world.is_solved() and world.nsteps < max_steps:
        action_matrix = []
        for char in stack:
            if myopic:
                f1 = count_num_to_observe(char,world)
            else:
                f1 = nonmyopic_num_to_observe(char,world,discount_factor=discount_factor)
            paths = find_paths_to_enchant(char,world)
            action_matrix.extend([[f1,path[0],path[1],path[2]] for path in paths])
        feature_matrix = np.array([a[:-1] for a in action_matrix])
        scores = feature_matrix @ np.array(weights)
        max_index = np.argmax(scores)
        best_path = action_matrix[max_index][-1]
        
 
        # Register who is currently observed so we can easily find the newly observed chars
        currently_observed = [c for c in world.characters if c.isObserved]
        # Do the actions in the best_path
        for action in best_path:
            world.step(action)
        char = [c for c in world.characters if c.location == best_path[-1][0]][0]
        if render:
            print(f"Chosen action path: {best_path}")
            world.render()
        # Work out who is newly observed
        now_observed = [c for c in world.characters if c.isObserved]
        newly_observed = [c for c in now_observed if c not in currently_observed]
        # Add the newly observed characters to the stack
        stack.extend(newly_observed)
        # Remove from the stack any characters that have outlived their usefulness
        stack = [char for char in stack if count_num_to_observe(char,world)>0]
    
    if world.nsteps >= max_steps:
        print("Warning: maximum number of steps reached before solving the world")
    return world.nsteps



def find_paths_to_enchant(target_char,world):
    """
    Find all paths to enchant the target character from currently enchanted characters
    Returns a list of Path objects
    """
    # Build an nx graph of currently observed characters
    G = nx.DiGraph()
    observed_chars = [char for char in world.characters if char.isObserved]
    node2char = {ii:char for ii,char in enumerate(observed_chars)}
    nodes = node2char.keys()
    target_node = [ii for ii,char in enumerate(observed_chars) if char==target_char][0]
    edges = [(ii,jj) for ii in nodes for jj in nodes if ii!=jj and node2char[ii].inRange(node2char[jj].location)]
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Find the current enchantment path
    enchantment_path = [ii for ii in nodes if node2char[ii] == world.characters[0]]
    while node2char[enchantment_path[-1]].isEnchanting is not None:
        enchantment_path.extend([ii for ii in nodes if node2char[ii] == node2char[enchantment_path[-1]].isEnchanting])
    
    # Work backward from the end of the path, finding paths to the target node
    paths = []
    redux = 0
    while len(enchantment_path) > 0:
        start_node = enchantment_path.pop()
        if nx.has_path(G, start_node, target_node):
            path_nodes = nx.shortest_path(G, start_node, target_node)
            # If this shortest path does not revisit any nodes in the enchantment path (except start)
            if not any([node in enchantment_path for node in path_nodes[1:]]):
                actions = [[node2char[path_nodes[i]].location, node2char[path_nodes[i+1]].location] for i in range(len(path_nodes)-1)]
                actions.append([node2char[target_node].location, "Sense"])
                extension = len(actions)
                paths.append([redux,extension,actions])
        redux += 1
    return paths

def oracle_strategy(world: ActivationGameWorld, render=False, **kwargs):
    """
    Oracle strategy for the Activation Game
    """
    _, oracle_steps = oracle_score(world)
    return oracle_steps

def WeightedDepthFirst(world: ActivationGameWorld):
    """
    A depth-first search strategy for the Activation Game.
    """

    # Get all the characters that are enchanted at the start of the search
    # Store them in a list against key 0 in the stack dictionary
    stack = {0: [char for char in world.characters if char.isEnchanted]}

    # Create Gaussian heatmap i.e., prioritise hidden points nearer the centre
    x, y = np.meshgrid(np.linspace(-1, 1, world.gridheight),
                    np.linspace(-1, 1, world.gridwidth))
    dst = np.sqrt(x**2 + y**2)
    # Smaller sigma - narrower gaussian. 
    sigma = 0.3
    # Changing muu from zero gives a decentralised gaussian
    muu = 0
    # Normal part of the Gaussian function
    normal = 1 / (2 * np.pi * sigma**2)
    # Calculating Gaussian filter
    gaussian_heatmap = 100*(np.exp(-((dst - muu)**2 / (2.0 * sigma**2))) * normal)
    gaussian_heatmap_inverse = np.max(gaussian_heatmap) - gaussian_heatmap + 1 
    
 
    while not world.is_solved():
        # Get the list of most recently added characters
        level = max(stack.keys())
        chars = stack[level]
        num_to_obs = [count_num_to_observe_weighted(char,world,gaussian_heatmap_inverse) for char in stack[level]]

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
            print

        # Register who is currently observed so we can easily find the newly observed chars
        currently_observed = [c for c in world.characters if c.isObserved]
        # Do the sensing action
        world.step([char.location, "Sense"])
        # print(char.location)
        # Work out who is newly observed
        now_observed = [c for c in world.characters if c.isObserved]
        newly_observed = [c for c in now_observed if c not in currently_observed]
        # Add the newly observed characters to the stack
        if len(newly_observed)>0:
            stack[level + 1] = newly_observed
    # world.render()
    return world.nsteps


def count_num_to_observe_weighted(char, world,gaussian_heatmap):
    """
    Count the number of new cells that could be observed by the character
    """
    min_x = max(0, char.location[0] - char.range)
    max_x = min(world.gridwidth - 1, char.location[0] + char.range)
    min_y = max(0, char.location[1] - char.range)
    max_y = min(world.gridheight - 1, char.location[1] + char.range)

    count = np.zeros((world.gridwidth, world.gridheight))
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            if world.obs_mask[x][y] == 0:
                count[x][y] = int(gaussian_heatmap[x][y])

    total_weighted_sum = np.sum(count)
    
    return total_weighted_sum


def eval_strategy(strategies=[DepthFirst,oracle_strategy], nsamples=10, **kwargs):
    """
    Evaluate a strategy over multiple samples of the world
    """
    nsteps = np.zeros((nsamples,len(strategies)))
    initial_seed = np.random.randint(0,10000)
    for ii in range(nsamples):
        try:
            world = ActivationGameWorld(seed=initial_seed+ii,silent=True)
        except:
            continue
        for si,strategy in enumerate(strategies):
            nsteps[ii,si] = strategy(world,**kwargs)
            world = ActivationGameWorld(seed=initial_seed+ii,silent=True)
    return nsteps 

if __name__ == "__main__":
    world = ActivationGameWorld()
    nsteps = linear_features_strategy(world,render=True,myopic=False)
    #world.render()
    print(f"Solved in {nsteps} steps")

#    nsteps, oracle = eval_strategy()
#    print(f"Average number of steps over samples: {np.mean(nsteps)}")
#    print(f"Maximum number of steps over samples: {np.max(nsteps)}")
#    print(f"Average oracle score over samples: {np.mean(oracle)}")
#    print(f"Average difference between strategy and oracle: {np.mean(nsteps - oracle)}")
