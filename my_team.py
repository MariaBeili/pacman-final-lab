# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Actions, Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]

##########
# Agents #
##########
class Team_CM(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.shared_data = {'opponent_positions': []}  # Shared data for team coordination

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, action) for action in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist =  9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights.
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}

class OffensiveReflexAgent(Team_CM):
    """
    Offensive agent focusing on collecting food and attacking vulnerable ghosts.
      
    Strategy:
    - Prioritize collecting food on the opponent's side while avoiding active ghosts.
    - Return home safely when carrying significant food or when all food is collected.
    - Chase vulnerable ghosts if they are scared.
    - Adapt dynamically to avoid high-risk areas and blocked paths.
      """

    def choose_action(self, game_state):
        """
        Decides the best action to take based on the agent's strategy.

        This method follows a multi-step decision-making process:
        1. Checks if the agent is carrying enough food to return home safely.
        2. Evaluates if the middle of the map is too risky (blocked by ghosts).
        3. Avoids nearby ghosts by moving to safe positions.
        4. Moves toward the nearest food if no immediate threats are present.
        """

        # Current position of the agent
        my_pos = game_state.get_agent_state(self.index).get_position()

        # Get list of food positions
        food_list = self.get_food(game_state).as_list()

        # Amount of food being carried
        carried_food = game_state.get_agent_state(self.index).num_carrying

        # Get the states of all opponent agents
        opponents = []
        for opponent in self.get_opponents(game_state):
            opponents.append(game_state.get_agent_state(opponent))

        # Filter the opponents to keep only those who are ghosts and have a known position
        ghosts = []
        for opponent in opponents:
            if not opponent.is_pacman and opponent.get_position() is not None:
                ghosts.append(opponent)

        # Convert positions to integers to avoid the floating-point issue, rounding the position
        my_pos = (int(my_pos[0]), int(my_pos[1]))

         # Step 1: Return home if carrying a lot of food or no food remains
        if carried_food >= 5 or not food_list:
            home_positions = self.get_home_positions(game_state)  # Get safe return positions

            # Find the nearest home position and move toward it. Using the min function with a lambda function to find the nearest home position
            # Lambda works such that for each home position in the home_positions list, calculates the maze distance from the current position of the agent
            # Obtaining the minimum distance from the agent's current position to the home position
            nearest_home = min(home_positions, key=lambda home: self.get_maze_distance(my_pos, home))

            # Return the action that moves the agent toward the nearest home position
            return self.get_action_toward(game_state, [nearest_home])

        # Step 2: Check if the middle of the map is blocked by ghosts
        middle_positions = self.get_middle_positions(game_state)

        # Check if the middle is blocked by ghosts, if so, find an alternative path to food
        if self.is_middle_blocked(game_state, middle_positions, ghosts):
            alternative_path = self.find_alternate_path(game_state, my_pos, food_list)

            # If an alternative path is found, use it
            if alternative_path:
                return self.get_action_toward(game_state, [alternative_path[0]])

        # Step 3: Avoid nearby ghosts by moving to safe positions
        # With any(), check if any of the ghosts are within a certain distance of the agent
        # any() returns True if at least one of the ghosts are within the specified distance
        if any(self.get_maze_distance(my_pos, (int(ghost.get_position()[0]), int(ghost.get_position()[1]))) <= 3 for ghost in ghosts):

            # If (one of) the ghosts are nearby, find safe positions and move toward them
            safe_positions = self.get_safe_positions(game_state, my_pos, ghosts)

            if safe_positions:
                return self.get_action_toward(game_state, safe_positions)
                
        # Step 4: Move toward the nearest food
        # Find the nearest food and move toward it using lambda function. For each food, calculate the maze distance from the current position of the agent
        # With lambda function we save the function in a variable and then call it.
        nearest_food = min(food_list, key=lambda food: self.get_maze_distance(my_pos, food))

        # Return the action that moves the agent toward the nearest food
        return self.get_action_toward(game_state, [nearest_food])

    def get_safe_positions(self, game_state, start, ghosts):
        """
        Computes a list of safe positions around the agent that are not close to ghosts.

        - It checks all neighboring positions.
        - Filters out positions that are adjacent to ghosts.
        - Returns a list of valid, safe positions.
        """
        safe_positions = []

        # Iterate over possible moves (down, up, left, right)
        for x, y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:

            # Calculate the next position
            next_pos = (start[0] + x, start[1] + y)

            # Ensure the next position is within valid grid boundaries and not a wall
            if not self.is_valid_position(next_pos, game_state):
                continue

            # Check distances to ghosts and add to safe positions if all distances are safe
            # Ensure positions are valid grid positions before calculating distance. Take the integer part of the position. And [0] and [1] are x and y coordinates.
            if all(self.get_maze_distance(next_pos, (int(ghost.get_position()[0]), int(ghost.get_position()[1]))) > 2 for ghost in ghosts):
                safe_positions.append(next_pos)

        return safe_positions

    def is_valid_position(self, position, game_state):
        """
        Check if a position is valid (inside the grid and not blocked by walls).
        """
        walls = game_state.get_walls()
        x, y = position

        if x < 0 or x >= walls.width or y < 0 or y >= walls.height:
            return False  # Outside the grid
        
        return not walls[x][y]  # True if the position is not a wall

    def get_features(self, game_state, action):
        """
        Computes a set of features for the state based on food collection and ghost avoidance.
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        my_pos = successor.get_agent_state(self.index).get_position()

        # Convert positions to integers to avoid the floating-point issue, rounding the position
        my_pos = (int(my_pos[0]), int(my_pos[1]))

        # Food-related features.
        features['successor_score'] = -len(food_list)  # Fewer food is better

        # Getting the minimum distance to food from the current position
        min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
        features['distance_to_food'] = min_distance

        # Carried food and returning safely
        carried_food = game_state.get_agent_state(self.index).num_carrying
        features['carried_food'] = carried_food
        if carried_food >= 5:  # Prioritize returning when carrying significant food
            features['return_distance'] = self.get_maze_distance(my_pos, self.start)

        # Ghost handling
        opponents = []
        for opponent in self.get_opponents(game_state):
            opponents.append(game_state.get_agent_state(opponent))
            
        ghosts = []
        for opponent in opponents:
            if not opponent.is_pacman and opponent.get_position() is not None:
                ghosts.append(opponent)

        # Regular ghost avoidance
        ghost_distances = []
        if len(ghosts) > 0:
            for g in ghosts:
                if g.scared_timer == 0:
                    ghost_distances.append(self.get_maze_distance(my_pos, (int(g.get_position()[0]), int(g.get_position()[1]))))

            if ghost_distances:
                features['ghost_distance'] = min(ghost_distances)  # Avoid active ghosts

        # Scared ghost handling
        scared_ghosts = []
        for g in ghosts:
            if g.scared_timer > 0:
                scared_ghosts.append(g)

        if len(scared_ghosts) > 0:
            scared_ghost_distances = []
            for g in scared_ghosts:
                scared_ghost_distances.append(self.get_maze_distance(my_pos, (int(g.get_position()[0]), int(g.get_position()[1]))))

            features['scared_ghost_distance'] = min(scared_ghost_distances)  # Chase scared ghosts

        return features

    def get_weights(self, game_state, action):
        """
        Assigns weights to the features to prioritize actions.
        """
        return {
            'successor_score': 100,  # More food means better action
            'distance_to_food': -1,  # Closer to food is better
            'carried_food': 50,  # Carrying food should be prioritized
            'return_distance': -10,  # Prioritize returning home when carrying food
            'ghost_distance': 10,  # Avoid active ghosts
            'scared_ghost_distance': -200  # Strongly prioritize chasing scared ghosts
        }

    def get_home_positions(self, game_state):
        """
        Get all positions on the agent's home side.
        """
        walls = game_state.get_walls()
        width = walls.width

        # The agent's home side is the left side of the map if it is a red agent, and the right side if it is a blue agent
        if self.red:
            middle_x = width // 2 - 1
        else:
            middle_x = width // 2

        # Iterate through each y value from 0 to walls.height-1
        # If the position is not a wall, it means that it is a valid position on the home side
        return [(middle_x, y) for y in range(walls.height) if not walls[middle_x][y]]

    def get_middle_positions(self, game_state):
        """
        Get the positions in the middle column of the map.
        """
        walls = game_state.get_walls()

        # The middle column is the width // 2 column
        middle_x = walls.width // 2

        # Iterate through each y value from 0 to walls.height-1
        # If the position is not a wall, it means that it is a valid position to go to in the middle
        return [(middle_x, y) for y in range(walls.height) if not walls[middle_x][y]]

    def is_middle_blocked(self, game_state, middle_positions, ghosts):
        """
        Check if the middle is blocked by an opponent agent.
        """
        for position in middle_positions:

            # Ensure positions are valid grid positions before calculating distance
            position = (int(position[0]), int(position[1]))

            # If any of the ghosts are within a certain distance of the middle
            if any(self.get_maze_distance(position, (int(ghost.get_position()[0]), int(ghost.get_position()[1]))) <= 2 for ghost in ghosts):
                return True
            
        return False

    def find_alternate_path(self, game_state, start, food_list):
        """
        Finds an alternate path to food if the middle is blocked.
        """
        # This is a simple heuristic where the agent tries to find food on the other side
        food_positions = []
        for food in food_list:
            food_positions.append((int(food[0]), int(food[1])))
        
        # Try to find the food on the side opposite to where the agent is
        opposite_food = min(food_positions, key=lambda f: self.get_maze_distance(start, f))
        
        return [opposite_food]  # For simplicity, returning the target food as the path

    def get_action_toward(self, game_state, targets):
        """
        Determines the best action to move closer to one of the target positions.
        """
        actions = game_state.get_legal_actions(self.index)
        best_action = Directions.STOP
        best_distance = float("inf")

        for action in actions:
            successor = self.get_successor(game_state, action)
            successor_pos = successor.get_agent_position(self.index)
            for target in targets:
                dist = self.get_maze_distance(successor_pos, target)
                if dist < best_distance:
                    best_action = action
                    best_distance = dist

        return best_action

class DefensiveReflexAgent(Team_CM):
    """
    Defensive agent focusing on protecting food and intercepting opponents.
    
    Strategy:
    - Patrol the middle of the map to monitor opponents crossing into the home side.
    - Chase and eliminate opponents entering the home side (Pacmen).
    - Return to the patrol zone when no invaders are detected.
    
    """

    def choose_action(self, game_state):
        """
        Decides the action for the defensive agent. 

        This function simply calls the parent class's `choose_action` method 
        to evaluate all possible actions using the defined features and weights.
        """
        return super().choose_action(game_state)

    def get_features(self, game_state, action):

        features = util.Counter()  # Initialize feature set
        successor = self.get_successor(game_state, action)  # Get the game state after taking the action
        my_state = successor.get_agent_state(self.index)  # Get the agent's state in the successor state
        my_pos = my_state.get_position()  # Current position of the agent

        # Defense Mode: Ensure the agent is defending its home side.
        features['on_defense'] = 1  # Default: The agent is on defense
        if my_state.is_pacman:  # If the agent becomes a Pacman (crosses into enemy territory)
            features['on_defense'] = 0  # Not on defense anymore

        # Track Invaders: Count and track opponents that have entered the home side.
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]  # Get opponent states
        invaders = [enemy for enemy in enemies if enemy.is_pacman and enemy.get_position() is not None]  # Identify visible invaders
        features['num_invaders'] = len(invaders)  # Number of invaders detected

        # Distance to Closest Invader: If there are invaders, calculate the closest distance to intercept them.
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, invader.get_position()) for invader in invaders]
            features['invader_distance'] = min(dists)  # Distance to the closest invader

        # Patrol Midline: If no invaders are detected, patrol the midline to monitor for potential intrusions.
        if len(invaders) == 0:
            middle_x = game_state.get_walls().width // 2  # The x-coordinate of the midline
            patrol_x = middle_x - 1 if self.red else middle_x + 1  # Choose the side of the midline to patrol based on the team
            patrol_positions = [
                (patrol_x, y) for y in range(game_state.get_walls().height)
                if not game_state.has_wall(patrol_x, y)  # Only consider positions without walls
            ]
            if patrol_positions:
                patrol_distance = min([self.get_maze_distance(my_pos, pos) for pos in patrol_positions])
                features['patrol_distance'] = patrol_distance  # Distance to the closest patrol position

        # Avoid Unnecessary Stops: Penalize stopping in place.
        if action == Directions.STOP:
            features['stop'] = 1  # Stopping is undesirable

        # Avoid Reversing: Penalize reversing direction to avoid unnecessary backtracking.
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]  # Reverse of the current direction
        if action == rev:
            features['reverse'] = 1  # Reversing is less desirable

        return features  # Return the computed features

    def get_weights(self, game_state, action):
        return {
            'on_defense': 100,  # Stay defensive
            'num_invaders': -1000,  # Prioritize blocking invaders
            'invader_distance': -10,  # Close to invaders
            'patrol_distance': -1,  # Patrol the midline
            'stop': -100,  # Penalize stopping
            'reverse': -2  # Penalize reversing direction
        }
