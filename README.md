## Repository for the activation game studied at the INI-EMA workshop in Jan 2026.

### Rules of the game
- played on a grid (default grid size is 20*20)
- each grid square has up to one character on it
- each character is one of three types: farmer, knight, king
- farmers and knights can be either enchanted/activated or not
- at the start of the game, one farmer is observed and enchanted, and no other characters are observed or enchanted
- each turn, one enchanted character can take an action which is either:
    - "Sense" so that all characters within their range become observed
    - enchant precisely one observed (by any character) farmer or knight within their range
- farmers have range 3, and knights have range 4 (I have taken the set of cells that a character can sense/enchant to be the subgrid consisting of all cells with L_infty norm less than or equal to 3 or 4)
- enchantment is transitive and transitory; if A enchants B enchants C, and the next action is A enchants D, then B and C are no longer enchanted
- the objective is for all the kings to be observed as soon as possible

### Classes defined
- ```ActivationGameCharacter```. You should never need to touch this yourself, other than to fix my bugs or otherwise improve things.
- ```ActivationGameWorld```, which largely consists of a set of ```ActivationGameCharacter```s, and what has been observed on the grid, and associated useful methods eg for taking actions and determining if the game is solved. This is the object to play with. Amendments to this class will be avoided if at all possible.
- ```ActivationGameEnv```, which is a [Gymnasium](https://gymnasium.farama.org/index.html) wrapper of ```ActivationGameWorld```. This wrapper may well evolve but I will try to ensure backward compatibility.
- ```ActivationGameCNN```, which is a first and likely very poor attempt at a feature extractor for [StableBaselines3](https://stable-baselines3.readthedocs.io/en/master/) algorithms learning on ```ActivationGameEnv```. Anything could happen to this. Indeed it probably shouldn't even be in the main file, since it's more a solution method than an environment to learn.

### How to experiment
I found it useful to play with an instantiation of an ```ActivationGameWorld```. On initiation you may specify ```gridsize=g``` (a scalar - only square grids for now - with default value 20) and/or ```composition=[x,y,z]``` where x is the number of farmers, y the number of knights and z the number of kings (default is [10,10,4]). A random placement of characters occurs, and then a check is performed to ensure the environment is solvable (if it is not, we resample, until eithe we find a solvable world or a maximum number of resamples occurs).
Important methods here are:
- ```render()```, which can display the grid as if you are the player of the game, or if passed ```observer="Gamesmaster"``` it will show the full game including unobserved characters
- ```step(action)```, which takes actions. The format of an ```action``` is ```(location,target)```, where location is the location of the character taking the action and target is either "Sense" or the location of the character to be enchanted
- ```is_solved()``` tests whether the game is complete or not
- ```getActions()``` returns the set of actions that are available given the current game state

### Useful initial commands to run
```
from ActivationGame import ActivationGameWorld
world = ActivationGameWorld()
world.render()
world.render(observer=“Gamesmaster”)
world.get_actions()
world.step(world.get_actions()[0])
world.is_solved()
world.render()
```
