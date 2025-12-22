"""
Code to support the EMA workshop at the INI in January 2026
This encodes the Activation Game posed by a workshop sponsor,
in a Gymnasium style environment.

David S Leslie
d.leslie@lancaster.ac.uk
"""

import numpy as np
import random

class ActivationGameCharacter:
    """
    Class representing a character in the game
    Attributes are:
    - chartype: the type of the character (valid types stored in allowed_types)
    - range: the range of detection/influence of the character, determined by type
    - location: a list of length 2 giving the location in the world
    - isEnchanted: a Boolean indicating whether the character is currently enchanted
    - isEnchanting: a GameCharacter which this GameCharacter is enchanting
    - couldEnchant: a set of GameCharacters that are in range and have been observed
    - isObserved: a Boolean indicating whether this character has been observed yet
    Methods are:
    - enchant: set the agent to enchanted
    - disenchant: remove the enchanted status of this agent and all recursively enchanted agents
    - inRange: indicate whether a location is in the observable range of this character
    - step: take an action (sense or enchant a particular character)

    """
    def __init__(self, chartype, location=(-1,-1)):
        
        # Check and stoee the character's type
        allowed_types = ["Farmer","Knight","King"]
        if chartype not in allowed_types:
            raise ValueError(f"chartype is {chartype} but must be one of {allowed_types}")
        self.chartype = chartype
        # Allocate the range based on the type
        if chartype == "Farmer":
            self.range = 3
        elif chartype == "Knight":
            self.range = 4
        else:
            self.range = 0
        
        # Check and store the character's location
        try:
            x,y = location
        except:
            raise ValueError(f"location is {location} but needs to be a tuple with two entries")
        self.location = [x,y]

        # Characters start non-enchanted and non-observed
        self.isEnchanted = False
        self.isEnchanting = None
        self.couldEnchant = set()
        self.isObserved = False


    def enchant(self):
        """
        Method to set the enchanted status of the character
        """
        self.isEnchanted = True

    def disenchant(self):
        """
        Method to recursively disenchant all "children"
        """
        if self.chartype != "King":
            self.isEnchanted = False
        if self.isEnchanting is not None:
            # self.isEnchanting is the GameCharacter that we are currently emchanting
            # This character automatically needs to be disenchanted too
            self.isEnchanting.disenchant()
            self.isEnchanting = None
    
    def observe(self,world):
        """
        Method to take actions when the character is observed
        """
        for char in world.characters:
            if char.inRange(self.location) and char != self:
                char.couldEnchant.add(self)
        self.isObserved = True
        
    
    def inRange(self,target_location):
        """
        Docstring for inRange
        
        :param target_location: tuple with two entries, location of a putative target

        Returns True if the target_location is in the sensing range, False otherwise
        """
        try:
            x, y = target_location
        except:
            raise ValueError(f"target_location is {target_location} but needs to be a tuple with two entries")
        
        if (abs(x-self.location[0])<=self.range) and (abs(y-self.location[1])<=self.range):
            return True
        else:
            return False


    def step(self,action,world):
        """
        Method for acting int the world
        
        :param action: the action to take. Either "sense" or a character to enchant
        :param world: the list of characters in the world

        Sets self.couldEnchant to be the characters that this agent could choose to enchant
        """

        if self.isEnchanted == False:
            raise RuntimeError(f"Character {self.location} is not enchanted but has been asked to perform an action")
        if self.chartype not in ["Farmer","Knight"]:
            raise ValueError("Only Farmers and Knights can act in the world")
        
        # If asked to take an action, any existing enchantments are broken
        if self.isEnchanted:
            self.disenchant()
            self.enchant()
 
        if action == "Sense":
            # Sensing means that we observe all the in-range characters
            for char in world.characters:
                if self.inRange(char.location):
                    char.observe(world)
            # Need to change the obs_mask of the world too, so that we know which locations are known to be empty
            # Just using a loop here. There are probably better ways, but I can't be bothered to be clever about
            # handling the edges of the world
            for ii in range(max(self.location[0]-self.range,0),min(self.location[0]+self.range+1,world.gridheight)):
                for jj in range(max(self.location[1]-self.range,0),min(self.location[1]+self.range+1,world.gridwidth)):
                    world.obs_mask[ii,jj] = True
        elif isinstance(action,ActivationGameCharacter):
            # if the action is a suitable character in the game, enchant that character
            if action not in self.couldEnchant:
                raise ValueError("Enchantment target at {action.location} is not enchantable right now")
            action.enchant()
            self.isEnchanting = action
        else:
            raise ValueError(f"Character action is {action} but must either be 'Sense' or a character to enchant")


class ActivationGameWorld:
    """
    A game world for the activation game
    """
    def __init__(self,gridsize = 20, composition=[10,10,4], seed = None):
        """
        Docstring for __init__
        
        :param gridsize: size of the square game grid
        :param composition: a list of integers with 
        """

        # Check and process inputs
        if isinstance(gridsize,int) and (gridsize > 0):
            self.gridwidth = gridsize
            self.gridheight = gridsize
        else:
            raise ValueError(f"gridsize ({gridsize}) miust be a positive integer")

        try:
            nfarmers, nknights, nkings = composition
        except:
            raise ValueError("composition must be a list/tuple of form [nfarmers,nknights,nkings]")
        if not isinstance(nfarmers,int) or not isinstance(nknights,int) or not isinstance(nkings,int):
            raise ValueError("The entries in composition must be integers")
        
        # Set the seed if it has been passed
        if seed is not None:
            random.seed(seed)

        # Sample a world until we get a solvable one
        self.sample_world(composition)
        ntries = 0
        while not self.is_solvable() and ntries < 100:
#            print("Resampling world - previous world was not solvable")
            ntries += 1
            self.sample_world(composition)
        if ntries > 0:
            print(f"Sampled {ntries} worlds before finding a solvable world")


        # Declare the whole grid to be unobserved
        self.obs_mask = np.zeros((self.gridheight,self.gridwidth), dtype = np.bool)

        # Activate the first farmer
        self.characters[0].enchant()
        self.characters[0].observe(self)

    def sample_world(self,composition):
        """
        Method to sample a new world configuration
        """
        # Sample a set of locations for the characters uniformly across the grid
        nfarmers, nknights, nkings = composition
        character_locations = [divmod(ii,self.gridwidth) 
                               for ii in random.sample(range(self.gridwidth*self.gridheight),
                                                       nfarmers+nknights+nkings)
                                ]
        
        # Allocate characters to these locations
        self.characters = [ActivationGameCharacter("Farmer",loc) for loc in character_locations[:nfarmers]]
        self.characters.extend(
            [ActivationGameCharacter("Knight",loc) for loc in character_locations[nfarmers:(nfarmers+nknights)]]
        )
        self.characters.extend(
            [ActivationGameCharacter("King",loc) for loc in character_locations[(nfarmers+nknights):]]
        )

    def is_solvable(self):
        """
        Method to check whether the current world is solvable
        Returns True if it is, False otherwise
        """

        to_expand = [self.characters[0]]
 #       print(self.characters[0].location)
        could_be_enchanted = set([self.characters[0]])

        # Recuresively extend the putative enchantment set
        while len(to_expand) > 0:
            current = to_expand.pop()
            could_be_enchanted.add(current)
            # Add all characters in range from the current char to the to_expand list
            # if they have not already been expanded from
            for char in self.characters:
                if current.inRange(char.location) and char not in could_be_enchanted:
                    to_expand.append(char)
            
        # Now see if all kings are in the enchanted set
        for char in self.characters:
            if char.chartype == "King":
                if char not in could_be_enchanted:
                    return False
        return True


    def get_actions(self):
        """
        This method returns all the valid actions in the world
        The return value is a list of locations of the initiating agent, and either "Sense" or the location of an enchantable agent
        The reason for the format is so that we can use it reasonably in RL algorithms
        
        We assume for now that an already enchanted agent cannot be enchanted again by a different enchantress,
        but this is an interesting proposition - could we pass off the enchantment to someone else?
        """

        sense_actions = [[char.location,"Sense"] for char in self.characters if char.isEnchanted]
        # This is horribly loopy...
        enchant_actions = []
        for char in self.characters:
            if char.isEnchanted:
                for target in char.couldEnchant:
                    enchant_actions.append([char.location,target.location])
        
        actions = sense_actions + enchant_actions

        return actions
            
    def get_action_mask(self):
        """
        Docstring for get_action_mask
        
        Returns the action mask for the current world state
        This is a np.array vector of length gridheight*gridsize+(gridheight*gridsize)**2
        The first gridheight*gridsize actions are the sense actions.
        The remainder are the enchant actions, ordered lexicographically
        by (initiator location, target location) where location is location[0]*gridwidth + location[1]
        """

        valid_actions = self.get_actions()

        action_mask = np.zeros(self.gridheight*self.gridwidth + (self.gridheight*self.gridwidth)**2,dtype=np.bool)

        for (initiator,target) in valid_actions:
            initiator_index = initiator[0]*self.gridwidth + initiator[1]
            if target=="Sense":
                action_mask[initiator_index] = True
            else:
                target_index = target[0]*self.gridwidth + target[1]
                enchant_action_index = (initiator_index+1)*(self.gridheight*self.gridwidth) + target_index
                action_mask[enchant_action_index] = True

        return action_mask
    
    def render(self,observer="Human"):
        """
        Docstring for gen_obs, method to generate observations for a player of the game
        
        :param observer:    if "SB3" then generate sb3-compatible observations
                            if "Human" then generate human-compatible observations
        """
        if observer == "Human" or observer == "Gamesmaster":
            # Start by noting all the observed grid
            if observer == "Human":
                grid = np.full((self.gridheight,self.gridwidth), fill_value='     ')
                for ii in range(self.gridheight):
                    for jj in range(self.gridwidth):
                        if self.obs_mask[ii][jj]:
                            grid[ii][jj] = '.    '
            else:
                grid = np.full((self.gridheight,self.gridwidth), fill_value='.    ')
            
            # Now insert the observed characters
            for char in self.characters:
                if char.isObserved or observer == "Gamesmaster":
                    if not char.isObserved:
                        enchanting_symbol = "x   "
                    elif char.isEnchanting is not None:
                        diff0 = char.isEnchanting.location[0]-char.location[0]
                        diff1 = char.isEnchanting.location[1]-char.location[1]
                        enchanting_symbol = f"{diff0:+2d}{diff1:+2d}"
                    else:
                        enchanting_symbol = '    '

                    if char.chartype == "Farmer":
                        typesymbol = 'F' if char.isEnchanted else 'f'
                    elif char.chartype == "Knight":
                        typesymbol = 'K' if char.isEnchanted else 'k'
                    else:
                        typesymbol = 'R' if char.isEnchanted else 'r'

                    symbol = typesymbol + enchanting_symbol
                    grid[char.location[0],char.location[1]] = symbol
        
            print(' ','   '.join([f"{ii:2d}" for ii in range(self.gridwidth)]))
            for ii in range(self.gridheight):
                print(f"{ii:2d}",''.join(grid[ii,:]))
            print("\n")
        elif observer == "sb3":
            raise NotImplementedError("Yet to implement the sb3 observation method")
        else:
            raise ValueError("observer must be in ['Human','Gamesmaster','SB3']")
        

    def step(self,action):
        """
        Method to take a step in the world
        
        :param action: the action to take, in the format returned by get_actions
        """

        try:
            initiator_loc, target = action
        except:
            raise ValueError(f"action is {action} but must be a list/tuple of form [initiator_location,target]")

        # Find the initiating character
        initiator = None
        for char in self.characters:
            if char.location == initiator_loc:
                initiator = char
                break
        if initiator is None:
            raise ValueError(f"No character found at location {initiator_loc}")

        # Find the target character if needed
        if target != "Sense":
            target_char = None
            for char in self.characters:
                if char.location == target:
                    target_char = char
                    break
            if target_char is None:
                raise ValueError(f"No character found at target location {target}")
            initiator.step(target_char,self)
        else:
            initiator.step("Sense",self)

    def is_solved(self):
        """
        Method that checks whether all Kings have been enchanted
        """
        return(all([char.isEnchanted for char in self.characters if char.chartype=="King"]))