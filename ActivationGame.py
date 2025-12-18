"""
Code to support the EMA workshop at the INI in January 2026
This encodes the Activation Game posed by a workshop sponsor,
in a Gymnasium style environment.

David S Leslie
d.leslie@lancaster.ac.uk
"""

class ActivationGameCharacter:
    """
    Class representing a character in the game
    Attributes are:
    - chartype: the type of the character (valid types stored in allowed_types)
    - range: the range of detection/influence of the character, determined by type
    - location: a list of length 2 giving the location in the world
    - IsEnchanted: a Boolean indicating whether the character is currently enchanted
    - IsEnchanting: a GameCharacter which this GameCharacter is enchanting
    - IsObserved: a Boolean indicating whether this character has been observed yet
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
            self.range = None
        
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
        self.isObserved = True
        for char in world.characters:
            if char.inRange(self.location):
                char.couldEnchant.add(self)

    
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
        elif isinstance(action,GameCharacter):
            if not self.CouldEnchant(action):
                raise ValueError("Enchantment target at {action.location} is not enchantable right now")
            action.enchant()
            self.isEnchanting = action
        else:
            raise ValueError(f"Character action is {action} but must either be 'Sense' or a character to enchant")


