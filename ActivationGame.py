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
        allowed_types = ["farmer","knight","king"]
        if chartype not in allowed_types:
            raise ValueError("chartype is {chartype} but must be one of {allowed_types}")
        self.chartype = chartype
        # Allocate the range based on the type
        if chartype == "farmer":
            self.range = 3
        elif chartype == "knight":
            self.range = 4
        else:
            self.range = 0
        
        # Check and store the character's location
        try:
            x,y = location
        except:
            raise ValueError("location is {location} but needs to be a tuple with two entries")
        self.location = [x,y]

        # Characters start non-enchanted and non-observed
        self.isEnchanted = False
        self.isEnchanting = None
        self.couldEnchant = set()
        self.isObserved = False


    def enchant():
        """
        Method to set the enchanted status of the character
        """
        self.isEnchanted = True

    def disenchant():
        """
        Method to remove the enchanted status of the character.
        Need to recursively disenchant all "children" too
        """
        self.isEnchanted = False
        if self.isEnchanting is not None:
            # self.isEnchanting is the GameCharacter that we are currently emchanting
            # This character automatically needs to be disenchanted too
            self.isEnchanting.disenchant()
            self.isEnchanting = None
    
    def observe(world):
        """
        Method to take actions when the character is observed
        """
        if self.isObserved == False:
            for char in world.characters:
                if char.inRange(self.location):
                    char.couldEnchant.add(self)
        self.isObserved = True
        
    
    def inRange(target_location):
        """
        Docstring for inRange
        
        :param target_location: tuple with two entries, location of a putative target

        Returns True if the target_location is in the sensing range, False otherwise
        """
        try:
            x, y = target_location
        except:
            raise ValueError("target_location is {target_location} but needs to be a tuple with two entries")
        
        if (abs(x-self.location[0])<=self.range) and (abs(y-self.location[1])<=self.range):
            return True
        else:
            return False


    def step(action,world):
        """
        Method for acting int thr world
        
        :param action: the action to take. Either "sense" or a character to enchant
        :param world: the list of characters in the world

        Sets self.CouldEnchant to be the characters that this agent could choose to enchant
        """

        if self.isEnchanted == False:
            raise RuntimeError(f"Character {self.location} is not enchanted but has been asked to perform an action")
        if self.chartype not in ["farmer","knight"]:
            raise ValueError("Only farmers and knights can sense the world")
        
        # If asked to take an action, any existing enchantments are broken
        if self.isEnchanted:
            self.disenchant()

        if char_action == "Sense":
            # Sensing means that we observe all the in-range characters
            for char in world.characters:
                if self.inRange(char.location):
                    char.observe(world)
            # Need to change the obs_mask of the world too, so that we know which locations are known to be empty
            # Just using a loop here. There are probably better ways, but I can't be bothered to be clever about
            # handling the edges of the world
            for ii in range(max(self.location[0]-self.range,0),min(self.location[0]+self.range,self.gridheight)):
                for jj in range(max(self.location[1]-self.range,0),min(self.location[1]+self.range,self.gridwidth)):
                    world.obs_mask[ii,jj] = True
        elif isinstance(action,GameCharacter):
            # if the action is a suitable character in the game, enchant that character
            if action not in self.CouldEnchant:
                raise ValueError("Enchantment target at {action.location} is not enchantable right now")
            action.enchant()
            self.isEnchanting = action
        else:
            raise ValueError(f"Character action is {action} but must either be 'Sense' or a character to enchant")


class ActivationGameWorld:
    """
    A game world for the activation game
    """
    def __init__(gridsize = 20, composition=[10,10,4], seed = None):
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


        # Declare the whole grid to be unobserved
        self.obs_mask = np.zeros((self.gridheight,self.gridwidth), dtype = np.bool)



        # Sample a set of locations for the characters uniformly across the grid
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

        # Activate the first farmer
        self.characters[0].enchant()
        self.characters[0].observe()


    def get_actions():
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
                    enchant_actions.extend([char.location,target.location])
        
        actions = sense_actions + enchant_actions

        return actions
            