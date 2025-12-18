"""
Code to support the EMA workshop at the INI in January 2026
This encodes the Activation Game posed by a workshop sponsor,
in a Gymnasium style environment.

David S Leslie
d.leslie@lancaster.ac.uk
"""

class GameCharacter:
    """
    Class representing a character in the game
    Attributes are:
    - chartype: the type of the character (valid types stored in allowed_types)
    - range: the range of detection/influence of the character, determined by type
    - location: a list of length 2 giving the location in the world
    - IsEnchanted: a Boolean indicating whether the character is currently enchanted
    - CouldEnchant: the other characters that this agent could enchant
            This set once an agent senses the world around it
    Methods are:
    - enchant: set the agent to enchanted
    - disenchant: remove the enchanted status of this agent and all recursively enchanted agents
    - action_sense: observe everything in the observation range
    - action_enchant: enchant a target agent
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

        # Characters start non-enchanged
        self.IsEnchanted = False

    def enchant():
        """
        Method to set the enchanted status of the character
        """

        self.IsEnchanted = True

    def disenchant():
        """
        Method to remove the enchanted status of the character.
        Need to recursively disenchant all "children" too
        """
        self.IsEnchanted = False
        if self.IsEnchanting is not None:
            # self.IsEnchanting is the GameCharacter that we are currently emchanting
            # This character automatically needs to be disenchanted too
            self.IsEnchanting.disenchant()
            self.IsEnchanting = None

    def action_sense(world):
        """
        Method for sensing the world nearby
        
        :param world: the list of characters in the world

        Sets self.CouldEnchant to be the characters that this agent could choose to enchant
        """
        if self.chartype not in ["farmer","knight"]:
            raise ValueError("Only farmers and knights can sense the world")
        if self.IsEnchanted:
            self.disenchant()
        self.CouldEnchant = [char in world.characters if all]