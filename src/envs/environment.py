from math import sqrt
import sys
sys.path.append("../..")

import numpy as np

from src import core
from src.envs.pacman.ghostAgents import RandomGhost
from src.envs.pacman.graphicsDisplay import PacmanGraphics
from src.envs.pacman.layout import getLayout
from src.envs.pacman.pacman import GameState


class Environment(core.Environment):
    """An RL environment of the game of Pacman.
    
    Attributes:
        _layout (pacman.Layout): A Pacman game layout object.
        _num_ghosts (int): The number of ghosts agents.
        _ghosts (list): A list of random ghost agents.
        _gameState (pacman.GameState): A GameState object representing the Pacman game.
        _shape (tuple(int)): A tuple of ints giving the shape of the numpy array
            representing the observable state of the game.
        _idxToAction (dict): A mapping from action index to game action.
        _actToIdx (dict): A reverse mapping from game action to action index.
        _num_actions (int): The total number of possible actions in the game.
        _display (pacman.PacmanGraphics): Graphical display for the Pacman game.
    """

    def __init__(self, layout="originalClassic", num_ghosts=4, graphics=False):
        """Initialize an environment object for the game of Pacman.
        
        Args:
            layout (string): The name of the game layout to be loaded.
            num_ghosts (int): Number of ghost agents.
            graphics (bool, optional): If true, a graphical interface is displayed.
                Default value is False.
        """
        # Initialize the game layout.
        self._layout = getLayout(layout)

        # Initialize ghosts.
        self._num_ghosts = min(num_ghosts, self._layout.getNumGhosts())
        self._ghosts = [RandomGhost(i+1) for i in range(self._num_ghosts)]

        # Initialize the game state.
        self._gameState = GameState()
        self._gameState.initialize(self._layout, self._num_ghosts)
        self._shape = self._observe(self._gameState).shape

        # Initialize action-to-idx mappings.
        self._idxToAction = dict(enumerate(self._gameState.getAllActions()))
        self._idxToAction[len(self._idxToAction)] = "Stop"
        self._actToIdx = {v:k for k, v in self._idxToAction.items()}
        self._num_actions = len(self._idxToAction)

        self._display = None
        if graphics:
            self._display = PacmanGraphics(zoom=1.0, frameTime=0.1)

        self.reset()

    def reset(self):
        """Resets the environment to the initial state.

        Returns:
            timestep (core.TimeStep): A namedtuple containing:
                observation (np.Array): A numpy array representing the observable initial
                    state of the environment.
                reward (float): 0.
                done (bool): False.
                info (dict}: {}.
        """
        self._gameState = GameState()
        self._gameState.initialize(self._layout, self._num_ghosts)

        if self._display is not None:
            self._display.initialize(self._gameState.data)

        return core.TimeStep(self._observe(self._gameState), 0, False, [])

    def actions(self):
        """Return a list with the ids of the legal actions for the current state."""
        return list(map(lambda x: self._actToIdx[x], self._gameState.getLegalPacmanActions()))

    def num_actions(self):
        """The total number of possible actions in the environment."""
        return self._num_actions

    def shape(self):
        """The shape of the numpy array representing the observable state of the environment."""
        return self._shape

    def close(self):
        """Close the graphics display."""
        if self._display is not None:
            self._display.finish()

    def step(self, actID):
        """This method performs one full ply by executing one move from every player
        present in the game layout. First, the action selected by the agent is performed
         by moving Pacman in the respective direction. After that every ghost makes a
         single move. The environment uses `RandomGhosts`, thus ghosts select actions
         uniformly from the list of legal actions.

        Args:
            actID (int): The index of the action selected by the agent.

        Returns:
            timestep (core.TimeStep): A namedtuple containing:
                observation (np.Array): A numpy array representing the observable state of
                    the environment.
                reward (float): The reward obtained by Pacman after all the players make a
                    move.
                done (bool): A boolean value indicating whether the episode has finished.
                info (dict}: {}.
        """
        # Create a dummy agent taking the given action.
        pacman_dummy_agent = lambda: None
        pacman_dummy_agent.getAction = lambda x: self._idxToAction[actID]

        # Loop over all agents (pacman and ghosts) to form a single ply.
        agents = [pacman_dummy_agent] + self._ghosts
        next_state = self._gameState
        for idx, ag in enumerate(agents):
                next_state = next_state.generateSuccessor(idx, ag.getAction(next_state))
                if self._display is not None:
                    self._display.update(next_state.data)
                reward = next_state.getScore() - self._gameState.getScore()
                done = (next_state.isWin() or next_state.isLose())
                if done: break

        info = []
        self._gameState = next_state
        return core.TimeStep(self._observe(next_state), reward, done, info)

    def _observe(self, gameState):
        """Constructs a numpy array representing the observable state of the environment.

        Args:
            gameState (pacman.GameState): The game state to be observed.

        Returns:
            observable (np.Array): A 1D numpy array of shape (size,). The size of the
                array depends on the size of the game layout and the number of ghosts.
                size = (width x height) + num_ghosts
        """
        observable = []
        width, height = gameState.data.layout.width, gameState.data.layout.height
        walls = gameState.getWalls()
        food = gameState.getFood()
        food_positions = ([(c, r) for r in range(food.height)
                            for c in range(food.width) if food[c][r]])
        capsule_positions = gameState.getCapsules()
        pacman_position = gameState.getPacmanPosition()
        ghosts = gameState.getGhostStates()
        ghost_positions = [g.getPosition() for g in ghosts]

        x, y = pacman_position

        # Calculate the number of relevant objects in my direction.
                #  East     West    North   South
        deltas = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for dx, dy in deltas:
            num_food, num_capsules, num_scared_ghosts, num_active_ghosts = 0, 0, 0, 0
            x_new = x + dx
            y_new = y + dy
            while x_new > 0 and x_new <= width and y_new > 0 and y_new <=height and not walls[x_new][y]:
                if x_new < food.width and y_new < food.height and food[x_new][y_new]:
                    num_food += 1
                if (x_new, y_new) in capsule_positions:
                    num_capsules += 1
                for ghost in ghosts:
                    if (x_new, y_new) == ghost.getPosition():
                        if ghost.scaredTimer > 0: num_scared_ghosts += 1
                        else: num_active_ghosts += 1
                x_new += dx
                y_new += dy
            observable.extend((num_food, num_capsules, num_scared_ghosts, num_active_ghosts))

        # Calculate distance to closest object for every direction.
                    #      East      West     North     South
        new_positions = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        for x_new, y_new in new_positions:
            close_food, close_capsule, close_active_ghost, close_scared_ghost = -1, -1, -1, -1
            if x_new > 0 and x_new <= width and y_new > 0 and y_new <= height:
                for food_x, food_y in food_positions:
                    close_food = max(close_food, _distance(food_x, food_y, x_new, y_new))
                for capsule_x, capsule_y in capsule_positions:
                    close_capsule = max(close_capsule, _distance(capsule_x, capsule_y, x_new, y_new))
                for idx, (ghost_x, ghost_y) in enumerate(ghost_positions):
                    if ghosts[idx].scaredTimer > 0:
                        close_scared_ghost = max(
                            close_scared_ghost, _distance(ghost_x, ghost_y, x_new, y_new))
                    else:
                        close_active_ghost = max(
                            close_active_ghost, _distance(ghost_x, ghost_y, x_new, y_new))
            observable.extend((close_food+1, close_capsule+1, close_active_ghost+1, close_scared_ghost+1))
            observable.append(1/(close_active_ghost+2))   # use also the inverse dist ot closest ghost

        # Calculate the scared timer for every ghost.
        scared_times = [g.scaredTimer for g in ghosts]
        observable.extend(scared_times)

        # Count the number of food pallets left.
        observable.append(food.count())

        # Count the number of active ghosts one step away:
                    #      East      West     North     South
        new_positions = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        num_close_ghosts = 0
        for x_new, y_new in new_positions:
            if (x_new, y_new) in ghost_positions:
                num_close_ghosts += 1
        observable.append(num_close_ghosts)

        # Return the observable as numpy array.
        return np.array(observable, dtype=np.float32)

    def _observe_boolean(self, gameState):
        """Constructs a numpy array representing the observable state of the environment.

        Args:
            gameState (pacman.GameState): The game state to be observed.

        Returns:
            observable (np.Array): A 1D numpy array of shape (size,). The size of the
                array depends on the size of the game layout and the number of ghosts.
                size = (width x height) + num_ghosts
        """
        width, height = gameState.data.layout.width, gameState.data.layout.height

        # Get pacman position encoded as one-hot vector.
        pacman = np.zeros((width, height))
        pacman[gameState.getPacmanPosition()] = 1
        pacman = pacman.reshape(-1)

        # Get ghost positions encoded as boolean vector.
        ghosts = np.zeros((width, height))
        for x, y in gameState.getGhostPositions():
            ghosts[int(x), int(y)] += 1
        ghosts = ghosts.reshape(-1)

        # Get food positions encoded as boolean vector.
        food = np.array(gameState.getFood().data, dtype=float).reshape(-1)

        # Get capsule positions encoded as boolean vector.
        capsules = np.zeros((width, height))
        for x, y in gameState.getCapsules():
            capsules[x, y] += 1
        capsules = capsules.reshape(-1)

        # Get scared times for all ghosts.
        scaredTimes = np.array([s.scaredTimer for s in gameState.getGhostStates()])

        # Stack all numpy vectors together.
        observation = np.concatenate([pacman, ghosts, food, capsules, scaredTimes])
        return observation.astype(np.float32)


def _manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def _euclidian_distance(x1, y1, x2, y2):
    return sqrt((x1 - x2)**2 + (y1 - y2)**2)

def _maze_distance(x1, y1, x2, y2, gameState):
    # TODO:
    # Implement function calculating the maze-distance between two points.
    pass

_distance = _manhattan_distance

#