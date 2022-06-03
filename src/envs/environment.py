import math
import sys
sys.path.append("../..")

import numpy as np

from src import core
from src.envs.pacman.ghostAgents import RandomGhost
from src.envs.pacman.graphicsDisplay import PacmanGraphics
from src.envs.pacman.layout import getLayout
from src.envs.pacman.pacman import GameState, SCARED_TIME


class Environment(core.Environment):
    """An RL environment of the game of Pacman.

    Attributes:
        layout (pacman.Layout): A Pacman game layout object.
        num_ghosts (int): The number of ghosts agents.
        ghosts (list): A list of random ghost agents.
        gameState (pacman.GameState): A GameState object representing the Pacman game.
        shape (tuple(int)): A tuple of ints giving the shape of the numpy array
            representing the observable state of the game.
        idxToAction (dict): A mapping from action index to game action.
        actToIdx (dict): A reverse mapping from game action to action index.
        num_actions (int): The total number of possible actions in the game.
        display (pacman.PacmanGraphics): Graphical display for the Pacman game.
        graphics (bool): A boolean flag indicating whether the environment should display
            a graphical interface for the game.
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
        self.layout = getLayout(layout)
        self._observe = self._observe_boolean

        # Initialize ghosts.
        self.num_ghosts = min(num_ghosts, self.layout.getNumGhosts())
        self.ghosts = [RandomGhost(i+1) for i in range(self.num_ghosts)]

        # Initialize the game state.
        self.gameState = GameState()
        self.gameState.initialize(self.layout, self.num_ghosts)
        food = self.gameState.getFood()
        self._initial_food = [(c, r) for r in range(food.height)
                                    for c in range(food.width) if food[c][r]]
        self._initial_caps = self.gameState.getCapsules()
        self.shape = self._observe(self.gameState).shape

        # Initialize action-to-idx mappings.
        self.idxToAction = dict(enumerate(self.gameState.getAllActions()))
        self.idxToAction[len(self.idxToAction)] = "Stop"
        self.actToIdx = {v:k for k, v in self.idxToAction.items()}
        self._num_actions = len(self.idxToAction)

        self.display = PacmanGraphics(zoom=1.0, frameTime=0.1)
        self._graphics = graphics

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
        self.gameState = GameState()
        self.gameState.initialize(self.layout, self.num_ghosts)

        if self._graphics:
            self.display.initialize(self.gameState.data)

        return core.TimeStep(self._observe(self.gameState), 0, False, {})

    def actions(self):
        """Return a list with the ids of the legal actions for the current state."""
        return list(map(lambda x: self.actToIdx[x], self.gameState.getLegalPacmanActions()))

    def num_actions(self):
        """The total number of actions in the environment."""
        return self._num_actions

    def observable_shape(self):
        """The shape of the numpy array representing the observable state of the environment."""
        return self.shape

    def graphics(self, graphics):
        """Set a boolean flag whether a graphical interface should be displayed."""
        self._graphics = graphics
        if not graphics:
            self.display.finish()

    def close(self):
        """Close the graphics display."""
        self.display.finish()

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
        pacman_dummy_agent.getAction = lambda x: self.idxToAction[actID]

        # Loop over all agents (pacman and ghosts) to form a single ply.
        agents = [pacman_dummy_agent] + self.ghosts
        next_state = self.gameState
        for idx, ag in enumerate(agents):
                next_state = next_state.generateSuccessor(idx, ag.getAction(next_state))
                if self._graphics:
                    self.display.update(next_state.data)
                reward = next_state.getScore() - self.gameState.getScore()
                done = (next_state.isWin() or next_state.isLose())
                if done: break

        info = {}
        self.gameState = next_state
        return core.TimeStep(self._observe(next_state), reward, done, info)

    def _observe_dense(self, gameState):
        """Constructs a numpy array representing the observable state of the environment.

        Args:
            gameState (pacman.GameState): The game state to be observed.

        Returns:
            observable (np.Array): A 1D numpy array of shape (size,). The size of the
                array depends on the number of features designed for the game state.
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
        """
        width, height = gameState.data.layout.width, gameState.data.layout.height

        # Get pacman position encoded as one-hot vector.
        pacman = np.zeros((width, height))
        pacman[gameState.getPacmanPosition()] = 1
        pacman = pacman.reshape(-1)

        # Get ghost positions encoded as boolean vector.
        ghosts = np.zeros((width, height))
        for state in gameState.getGhostStates():
            x, y = state.getPosition()
            ghosts[int(x), int(y)] += 1 - (state.scaredTimer/SCARED_TIME)
        ghosts = ghosts.reshape(-1)

        # Get food positions encoded as boolean vector.
        food = np.array(gameState.getFood().data, dtype=float).reshape(-1)

        # Get capsule positions encoded as boolean vector.
        capsules = np.zeros((width, height))
        for x, y in gameState.getCapsules():
            capsules[x, y] += 1
        capsules = capsules.reshape(-1)

        # Stack all numpy vectors together.
        observation = np.concatenate([pacman, ghosts, food, capsules])
        return observation.astype(np.float32)

    def _observe_onehot(self, gameState):
        """Constructs a one-hot numpy array encoding representing the observable state of
        the environment.

        Args:
            gameState (pacman.GameState): The game state to be observed.

        Returns:
            observable (np.Array): A 1D numpy array of shape (size,). The size of the
                array depends on the size of the game layout and the number of ghosts.
        """
        width, height = gameState.data.layout.width, gameState.data.layout.height

        pacman_position = gameState.getPacmanPosition()
        pacman_position = pacman_position[0]*height + pacman_position[1]

        ghosts = gameState.getGhostStates()
        ghost_positions = [g.getPosition()[0]*height + g.getPosition()[1] for g in ghosts]

        positions_cap = width * height

        food = gameState.getFood()
        food_positions = ([(c, r) for r in range(food.height)
                            for c in range(food.width) if food[c][r]])
        food_onehot = []
        i, j = 0, 0
        while i < len(self._initial_food) and j < len(food_positions):
            if food_positions[j] == self._initial_food[i]:
                food_onehot.append(1)
                j += 1
            else:
                food_onehot.append(0)
            i += 1
        if j != len(food_positions):
            raise ValueError("failed to construct one-hot encoding for food positions")
        food_encoding = int("0" + "".join(str(x) for x in food_onehot), 2)
        food_cap = 2 ** len(self._initial_food)

        capsule_positions = gameState.getCapsules()
        capsule_onehot = []
        i, j = 0, 0
        while i < len(self._initial_caps) and j < len(capsule_positions):
            if capsule_positions[j] == self._initial_caps[i]:
                capsule_onehot.append(1)
                j += 1
            else:
                capsule_onehot.append(0)
            i += 1
        if j != len(capsule_positions):
            raise ValueError("failed to construct one-hot encoding for food positions")
        capsule_encoding = int("0" + "".join(str(x) for x in capsule_onehot), 2)
        capsule_cap = 2 ** len(self._initial_caps)

        dense_encoding = [pacman_position] + ghost_positions + [food_encoding, capsule_encoding]
        caps = [positions_cap] + [positions_cap] * len(ghost_positions) + [food_cap, capsule_cap]

        index = 0
        for i in range(len(dense_encoding)):
            index += int(dense_encoding[i] * math.prod(caps[i+1:]))

        one_hot = np.zeros(math.prod(caps), dtype=np.float32)
        one_hot[index] = 1.
        return one_hot


def _manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def _euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def _maze_distance(x1, y1, x2, y2, gameState):
    # TODO:
    # Implement function calculating the maze-distance between two points.
    pass

_distance = _manhattan_distance

#