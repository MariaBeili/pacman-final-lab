# my_team.py
# ---------------
# Do not modify the filename or the import structure.

import random
import contest.util as util
from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    Returns a list of two agents forming a team.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
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
        values = [self.evaluate(game_state, action) for action in actions]

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        # Return to start if low food remains
        if food_left <= 2:
            best_dist = float("inf")
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
        Finds the next successor.
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
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
        Returns a counter of features for the state.
        """
        return util.Counter()

    def get_weights(self, game_state, action):
        """
        Returns weights for features.
        """
        return {}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    Offensive agent focusing on collecting food and attacking vulnerable ghosts.
    """

    def choose_action(self, game_state):
        """
        Overrides choose_action to prioritize safety and strategic movement.
        """
        my_pos = game_state.get_agent_state(self.index).get_position()
        food_list = self.get_food(game_state).as_list()
        carried_food = game_state.get_agent_state(self.index).num_carrying
        opponents = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [a for a in opponents if not a.is_pacman and a.get_position() is not None]

        # Convert positions to integers to avoid the floating-point issue
        my_pos = (int(my_pos[0]), int(my_pos[1]))  # Round the position

        # If carrying food and a ghost is nearby, return home
        if carried_food >= 5 or not food_list:
            home_positions = self.get_home_positions(game_state)
            nearest_home = min(home_positions, key=lambda h: self.get_maze_distance(my_pos, h))
            return self.get_action_toward(game_state, my_pos, [nearest_home])

        # Check if the middle is blocked (or too risky), then explore other paths
        middle_positions = self.get_middle_positions(game_state)
        if self.is_middle_blocked(game_state, middle_positions, ghosts):
            # If the middle is blocked, find an alternative path to the opposite side
            alternative_path = self.find_alternate_path(game_state, my_pos, food_list)
            if alternative_path:
                return self.get_action_toward(game_state, my_pos, [alternative_path[0]])

        # Find the nearest food and avoid ghosts
        nearest_food = min(food_list, key=lambda f: self.get_maze_distance(my_pos, f))

        # Check if there are ghosts near
        if any(self.get_maze_distance(my_pos, (int(ghost.get_position()[0]), int(ghost.get_position()[1]))) <= 3 for ghost in ghosts):
            safe_positions = self.get_safe_positions(game_state, my_pos, ghosts)
            if safe_positions:
                return self.get_action_toward(game_state, my_pos, safe_positions)

        # If no ghosts nearby, go toward the food
        return self.get_action_toward(game_state, my_pos, [nearest_food])

    def get_safe_positions(self, game_state, start, ghosts):
        """
        Get all safe positions that are not adjacent to a ghost.
        """
        safe_positions = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_pos = (start[0] + dx, start[1] + dy)
            
            # Ensure the next position is within valid grid boundaries
            if not self.is_valid_position(next_pos, game_state):
                continue

            # Convert the next position to integers before passing it into get_maze_distance
            next_pos = (int(next_pos[0]), int(next_pos[1]))

            # Ensure that we only use valid ghost positions and check distances
            ghost_distances_valid = True
            for ghost in ghosts:
                ghost_pos = ghost.get_position()
                ghost_pos = (int(ghost_pos[0]), int(ghost_pos[1]))  # Ensure valid integer position
                if self.get_maze_distance(next_pos, ghost_pos) <= 2:
                    ghost_distances_valid = False
                    break
            
            if ghost_distances_valid:
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

        # Convert positions to integers to avoid the floating-point issue
        my_pos = (int(my_pos[0]), int(my_pos[1]))  # Round the position

        # Food-related features
        features['successor_score'] = -len(food_list)  # Fewer food is better
        if len(food_list) > 0:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        # Carried food and returning safely
        carried_food = game_state.get_agent_state(self.index).num_carrying
        features['carried_food'] = carried_food
        if carried_food >= 5:  # Prioritize returning when carrying significant food
            features['return_distance'] = self.get_maze_distance(my_pos, self.start)

        # Ghost handling
        opponents = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [a for a in opponents if not a.is_pacman and a.get_position() is not None]

        # Regular ghost avoidance
        if len(ghosts) > 0:
            ghost_distances = [self.get_maze_distance(my_pos, (int(g.get_position()[0]), int(g.get_position()[1]))) for g in ghosts if g.scared_timer == 0]
            if ghost_distances:
                features['ghost_distance'] = min(ghost_distances)  # Avoid active ghosts

        # Scared ghost handling
        scared_ghosts = [g for g in ghosts if g.scared_timer > 0]
        if len(scared_ghosts) > 0:
            scared_ghost_distances = [self.get_maze_distance(my_pos, (int(g.get_position()[0]), int(g.get_position()[1]))) for g in scared_ghosts]
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
        mid_x = width // 2 - 1 if self.red else width // 2
        return [(mid_x, y) for y in range(walls.height) if not walls[mid_x][y]]

    def get_middle_positions(self, game_state):
        """
        Get the positions in the middle column of the map.
        """
        walls = game_state.get_walls()
        mid_x = walls.width // 2
        return [(mid_x, y) for y in range(walls.height) if not walls[mid_x][y]]

    def is_middle_blocked(self, game_state, middle_positions, ghosts):
        """
        Check if the middle is blocked by an opponent agent.
        """
        for position in middle_positions:
            # Ensure positions are valid grid positions before calculating distance
            position = (int(position[0]), int(position[1]))
            if any(self.get_maze_distance(position, (int(ghost.get_position()[0]), int(ghost.get_position()[1]))) <= 2 for ghost in ghosts):
                return True
        return False

    def find_alternate_path(self, game_state, start, food_list):
        """
        Finds an alternate path to food if the middle is blocked.
        """
        # This is a simple heuristic where the agent tries to find food on the other side
        walls = game_state.get_walls()
        food_positions = [(int(food[0]), int(food[1])) for food in food_list]
        
        # Try to find the food on the side opposite to where the agent is
        opposite_food = min(food_positions, key=lambda f: self.get_maze_distance(start, f))
        
        return [opposite_food]  # For simplicity, returning the target food as the path

    def get_action_toward(self, game_state, start, targets):
        """
        Get the best action toward one of the targets.
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




class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    Defensive agent focusing on protecting food and intercepting opponents.
    """

    def choose_action(self, game_state):
        """
        Overrides choose_action explicitly to ensure execution.
        """
        return super().choose_action(game_state)

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Stay on defense unless turned into a Pacman
        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] = 0

        # Track invaders
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [enemy for enemy in enemies if enemy.is_pacman and enemy.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, invader.get_position()) for invader in invaders]
            features['invader_distance'] = min(dists)

        # Patrol the midline when no invaders are visible
        if len(invaders) == 0:
            mid_x = game_state.get_walls().width // 2
            patrol_x = mid_x - 1 if self.red else mid_x + 1
            patrol_positions = [(patrol_x, y) for y in range(game_state.get_walls().height) if not game_state.has_wall(patrol_x, y)]
            if patrol_positions:
                patrol_distance = min([self.get_maze_distance(my_pos, pos) for pos in patrol_positions])
                features['patrol_distance'] = patrol_distance

        # Penalize stopping and reversing direction
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {
            'on_defense': 100,  # Stay defensive
            'num_invaders': -1000,  # Prioritize blocking invaders
            'invader_distance': -10,  # Close to invaders
            'patrol_distance': -1,  # Patrol the midline
            'stop': -100,  # Penalize stopping
            'reverse': -2  # Penalize reversing direction
        }