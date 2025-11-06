import pkgutil
# Monkey-patch for pkgutil.ImpImporter on Python 3.12+
if not hasattr(pkgutil, 'ImpImporter'):
    pkgutil.ImpImporter = type('ImpImporter', (), {})

import random
import logging
from typing import List, Optional, Tuple, Dict
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

# --- Relative Imports from within the econ_sim2 package ---
from config import Config
# Economy class is still in sim_2.py for now
from sim_2 import Economy
# Agent class is now in agent.py
from agent import Agent
# Core components like Grid, Nodes, etc. are in core_components.py
from core_components import Grid, ResourceNode, TrainingCenter, ToolFactory, Store, Project, Loan

# --- Common Colors (Define at class/module level for access by methods) ---
WHITE = (255, 255, 255); BLACK = (0, 0, 0); GREY = (200, 200, 200); LIGHT_GREY = (220, 220, 220)
RED = (255, 0, 0); BLUE = (0, 0, 255); GREEN = (0, 255, 0); YELLOW = (255, 255, 0); MAGENTA = (255,0,255)
# --- End Colors ---

# --- Helper function for rendering text safely (moved to top level) ---
def _render_text_safe(surface, text: str, pos: tuple, font, color=BLACK, center=False):
    """Helper to render text with error handling and optional centering."""
    try:
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        surface.blit(text_surface, text_rect)
    except Exception as e:
        logging.warning(f"Failed to render text '{text}' at {pos}: {e}")

# --- Gymnasium Environment ---
class AgentEconomyEnv(gym.Env):
    """Custom Environment for Agent Economy Simulation that follows gym interface."""
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': Config.FPS}

    def __init__(self, num_agents: int, num_resources: int, rl_agent_id: int, max_steps: int = 1000, render_mode: Optional[str] = None):
        super().__init__()

        self.max_steps = max_steps
        self.render_mode = render_mode
        self.sim_num_agents = num_agents # Store initial config
        self.sim_num_resources = num_resources # Store initial config
        self.rl_agent_id = rl_agent_id # The ID of the agent controlled by the RL algorithm
        # Initialize the Economy, providing the rl_agent_id so it can mark the agent
        self.economy = Economy(num_agents=self.sim_num_agents, num_resources=self.sim_num_resources, seed=Config.SEED, rl_agent_id=self.rl_agent_id)
        self.current_step = 0

        # Ensure the RL agent exists initially and is marked
        self.rl_agent = next((a for a in self.economy.agents if a.id == self.rl_agent_id), None)
        if not self.rl_agent:
             raise ValueError(f"RL agent with ID {self.rl_agent_id} not found in initial economy setup.")
        if not self.rl_agent.is_rl_controlled:
             # This check ensures the Economy init correctly marked the agent
             logging.warning(f"RL agent {self.rl_agent_id} was found but not marked as RL controlled in Economy init.")
             self.rl_agent.is_rl_controlled = True # Force it just in case


        # --- Define Action Space ---
        # Discrete actions corresponding to Agent's potential tasks/goals
        self._action_to_task_map = {
            0: 'idle',
            1: 'harvest_food',
            2: 'harvest_wood',
            3: 'train',
            4: 'buy_tool_wood',
            5: 'buy_tool_food',
            6: 'buy_ent',
            7: 'deliver_wood_to_factory', # For Tool Artisan
            8: 'go_get_tool_from_factory_wood', # For Seller
            9: 'go_get_tool_from_factory_food', # For Seller
            10: 'find_buyer_for_tool_wood', # For Seller
            11: 'find_buyer_for_tool_food', # For Seller
            12: 'build_tier2_device',
            13: 'seek_loan_for_tool', # Simplified general purpose loan
            14: 'trade_sell_food', # General sell
            15: 'trade_sell_wood', # General sell
            16: 'trade_buy_food', # General buy
            17: 'trade_buy_wood', # General buy
            18: 'start_project' # For Developer
            # Note: Emergency actions (e.g., seek_loan_for_food) are handled internally by the agent based on state, not direct RL actions.
            # The RL agent chooses a general goal, and the agent adapts if an emergency arises.
        }
        self.action_space = spaces.Discrete(len(self._action_to_task_map))

        # --- Define Observation Space ---
        # Includes agent state, market state, nearby environment state. Normalized where possible.
        max_wealth_norm = 500; max_skill_norm = 5; max_inv_norm = 50; max_tools_norm = 10; max_loans_norm = 5
        max_price_norm = 20; max_fac_stock_norm = 50; max_store_stock_norm = 50; max_dist_norm = Config.GRID_WIDTH + Config.GRID_HEIGHT; max_cap_norm = 200
        max_hunger_norm = Config.HUNGER_INTERVAL * 1.5 # How many steps of hunger to normalize over

        # Define low/high bounds for a Box space. Most are normalized [0, 1]. Wealth can be negative.
        # --- MODIFICATION START: Construct lists first ---        
        low_bounds_list = [
            -np.inf, 0, 0, 0, 0, # wealth(raw), skill(norm), food(norm), wood(norm), ent(norm)
            0, 0, 0, 0, 0, # has_t2(bool), tools_w(norm), tools_f(norm), sale_w(norm), sale_f(norm)
            0, 0, 0, 0, # pos_x(norm), pos_y(norm), happiness(0-1), hunger(norm)
            0, # default_cooldown(bool 0/1)
            0, 0, # loans_taken_count(norm), loans_given_count(norm)
            0, 0, # market_price_w(norm), market_price_f(norm)
            0, 0, 0, # fac_w_tools(norm), fac_f_tools(norm), fac_price(norm)
            0, 0, # store_ent_stock(norm), store_price(norm)
            0, 0, 0, 0, # dist_nearest_wood(norm), cap_nearest_wood(norm), dist_nearest_food(norm), cap_nearest_food(norm)
            0, 0, 0, # dist_train_center(norm), dist_factory(norm), dist_store(norm)
            0 # dist_nearest_agent(norm)
        ]
        # Add one-hot encoded role (length = len(Config.ROLES))
        low_bounds_list.extend([0] * len(Config.ROLES))
        # Add one-hot encoded archetype (length = len(Config.AGENT_ARCHETYPES))
        low_bounds_list.extend([0] * len(Config.AGENT_ARCHETYPES))

        high_bounds_list = [
            np.inf, 1, 1, 1, 1, # wealth(raw), skill(norm), food(norm), wood(norm), ent(norm)
            1, 1, 1, 1, 1, # has_t2(bool), tools_w(norm), tools_f(norm), sale_w(norm), sale_f(norm)
            1, 1, 1, 1, # pos_x(norm), pos_y(norm), happiness(0-1), hunger(norm)
            1, # default_cooldown(bool 0/1)
            1, 1, # loans_taken_count(norm), loans_given_count(norm)
            1, 1, # market_price_w(norm), market_price_f(norm)
            1, 1, 1, # fac_w_tools(norm), fac_f_tools(norm), fac_price(norm)
            1, 1, # store_ent_stock(norm), store_price(norm)
            1, 1, 1, 1, # dist_nearest_wood(norm), cap_nearest_wood(norm), dist_nearest_food(norm), cap_nearest_food(norm)
            1, 1, 1, # dist_train_center(norm), dist_factory(norm), dist_store(norm)
            1 # dist_nearest_agent(norm)
        ]
        # One-hot roles
        high_bounds_list.extend([1] * len(Config.ROLES))
        # One-hot archetypes
        high_bounds_list.extend([1] * len(Config.AGENT_ARCHETYPES))
        
        low_bounds = np.array(low_bounds_list, dtype=np.float32)
        high_bounds = np.array(high_bounds_list, dtype=np.float32)
        # --- MODIFICATION END ---

        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, shape=(low_bounds.shape[0],), dtype=np.float32)
        logging.info(f"AgentEnv: Observation space shape: {self.observation_space.shape}")

        # Pygame setup for rendering (optional) - initialized in render() if needed
        # REMOVED: self.window = None
        # REMOVED: self.clock = None
        # REMOVED: self.font = None
        # REMOVED: self.small_font = None
        # REMOVED: self.info_font = None


    def _normalize(self, value, max_val, clip_min=0, clip_max=1):
        """Helper to normalize and clip a value to [clip_min, clip_max]."""
        if max_val == 0: return float(clip_min) # Avoid division by zero, return min value
        # Ensure value is float before division
        normalized = float(value) / float(max_val)
        return np.clip(normalized, clip_min, clip_max)

    def _one_hot_encode(self, category: str, category_list: list) -> list:
        """ Helper to one-hot encode a category string based on a list. """
        encoding = [0.0] * len(category_list)
        try:
            index = category_list.index(category)
            encoding[index] = 1.0
        except ValueError:
            logging.warning(f"Category '{category}' not found in list {category_list} for one-hot encoding.")
        return encoding

    def _get_obs(self):
        """ Get observation for the RL agent. Returns a zero vector if agent is dead."""
        agent = self.rl_agent # Use the reference stored in __init__
        eco = self.economy

        if not agent or agent not in eco.agents: # Check if agent died or was removed
            # Return a zero observation consistent with the space shape and type
            logging.debug(f"RL Agent {self.rl_agent_id} is dead or removed. Returning zero observation.")
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        # --- Normalization Factors (redefine here for clarity) ---
        max_wealth_norm = 500; max_skill_norm = 5; max_inv_norm = 50; max_tools_norm = 10; max_loans_norm = 5
        max_price_norm = 20; max_fac_stock_norm = 50; max_store_stock_norm = 50; max_dist_norm = Config.GRID_WIDTH + Config.GRID_HEIGHT; max_cap_norm = 200
        max_hunger_norm = Config.HUNGER_INTERVAL * 1.5

        # --- Agent State ---
        wealth = agent.wealth # Raw wealth, not normalized here
        skill = self._normalize(agent.skill, max_skill_norm)
        food_inv = self._normalize(agent.inventory.get('food', 0), max_inv_norm)
        wood_inv = self._normalize(agent.inventory.get('wood', 0), max_inv_norm)
        ent_inv = self._normalize(agent.inventory.get('ent', 0), max_inv_norm)
        has_t2 = 1.0 if agent.inventory.get('has_tier2_device', False) else 0.0
        tools_w = self._normalize(agent.tools.get('wood', 0), max_tools_norm)
        tools_f = self._normalize(agent.tools.get('food', 0), max_tools_norm)
        sale_w = self._normalize(agent.inventory.get('tools_for_sale', {}).get('wood', 0), max_tools_norm)
        sale_f = self._normalize(agent.inventory.get('tools_for_sale', {}).get('food', 0), max_tools_norm)
        pos_x = self._normalize(agent.pos[0], Config.GRID_WIDTH)
        pos_y = self._normalize(agent.pos[1], Config.GRID_HEIGHT)
        happiness = np.clip(agent.happiness, 0, 1) # Happiness is assumed to be roughly 0-1
        hunger = self._normalize(agent.food_hunger, max_hunger_norm)
        default_cd = 1.0 if agent.default_cooldown > 0 else 0.0
        loans_taken_count = self._normalize(len(agent.loans_taken), max_loans_norm)
        loans_given_count = self._normalize(len(agent.loans_given), max_loans_norm)

        # --- Market State ---
        price_w_norm = self._normalize(eco.price_w, max_price_norm)
        price_f_norm = self._normalize(eco.price_f, max_price_norm)
        fac_w_tools_norm = self._normalize(eco.factory.wood_tools, max_fac_stock_norm)
        fac_f_tools_norm = self._normalize(eco.factory.food_tools, max_fac_stock_norm)
        fac_price_norm = self._normalize(eco.factory.price, max_price_norm * 2) # Factory price can exceed market price
        store_ent_norm = self._normalize(eco.store.ent_stock, max_store_stock_norm)
        store_price_norm = self._normalize(eco.store.price, max_price_norm * 2)

        # --- Environmental State (Nearest Nodes, Structures, Agent) ---
        dist_w, cap_w, dist_f, cap_f = max_dist_norm, 0, max_dist_norm, 0 # Defaults
        wood_nodes = [n for n in eco.grid.nodes if n.resource_type == 'wood' and n.capacity > 0]
        food_nodes = [n for n in eco.grid.nodes if n.resource_type == 'food' and n.capacity > 0]
        if wood_nodes:
            nearest_w = min(wood_nodes, key=lambda n: eco._get_distance(agent.pos, n.pos))
            dist_w = eco._get_distance(agent.pos, nearest_w.pos)
            cap_w = nearest_w.capacity
        if food_nodes:
            nearest_f = min(food_nodes, key=lambda n: eco._get_distance(agent.pos, n.pos))
            dist_f = eco._get_distance(agent.pos, nearest_f.pos)
            cap_f = nearest_f.capacity

        dist_w_norm = self._normalize(dist_w, max_dist_norm)
        cap_w_norm = self._normalize(cap_w, max_cap_norm)
        dist_f_norm = self._normalize(dist_f, max_dist_norm)
        cap_f_norm = self._normalize(cap_f, max_cap_norm)

        dist_train_norm = self._normalize(eco._get_distance(agent.pos, eco.train_center.pos), max_dist_norm)
        dist_factory_norm = self._normalize(eco._get_distance(agent.pos, eco.factory.pos), max_dist_norm)
        dist_store_norm = self._normalize(eco._get_distance(agent.pos, eco.store.pos), max_dist_norm)

        dist_agent_norm = 1.0 # Default if no other agents
        other_agents = [a for a in eco.agents if a.id != agent.id]
        if other_agents:
            nearest_a = min(other_agents, key=lambda a: eco._get_distance(agent.pos, a.pos))
            dist_agent = eco._get_distance(agent.pos, nearest_a.pos)
            dist_agent_norm = self._normalize(dist_agent, max_dist_norm)

        # --- Categorical Features (One-Hot Encoded) ---
        role_encoded = self._one_hot_encode(agent.role, Config.ROLES)
        archetype_encoded = self._one_hot_encode(agent.archetype, Config.AGENT_ARCHETYPES)

        # --- Assemble Observation Vector ---
        obs_list = [
            wealth, skill, food_inv, wood_inv, ent_inv,
            has_t2, tools_w, tools_f, sale_w, sale_f,
            pos_x, pos_y, happiness, hunger,
            default_cd,
            loans_taken_count, loans_given_count,
            price_w_norm, price_f_norm,
            fac_w_tools_norm, fac_f_tools_norm, fac_price_norm,
            store_ent_norm, store_price_norm,
            dist_w_norm, cap_w_norm, dist_f_norm, cap_f_norm,
            dist_train_norm, dist_factory_norm, dist_store_norm,
            dist_agent_norm
        ]
        obs_list.extend(role_encoded)
        obs_list.extend(archetype_encoded)

        obs = np.array(obs_list, dtype=np.float32)

        # Final check for shape consistency
        if obs.shape[0] != self.observation_space.shape[0]:
            logging.error(f"FATAL: Observation vector length ({obs.shape[0]}) does not match space definition ({self.observation_space.shape[0]}). Check _get_obs and space definition. Agent state: {agent.__dict__}")
            # Attempt to pad or truncate, though this indicates a critical error
            padded_obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            size_to_copy = min(obs.shape[0], self.observation_space.shape[0])
            padded_obs[:size_to_copy] = obs[:size_to_copy]
            # Raise error instead of returning potentially invalid observation?
            raise ValueError("Observation shape mismatch!")
            # return padded_obs

        return obs

    def _get_info(self):
        """ Return auxiliary info about the RL agent's state. """
        agent = self.rl_agent

        if not agent or agent not in self.economy.agents:
             # Provide default info for a dead agent
             return {
                 'status': 'dead', 'wealth': 0, 'food': 0, 'wood': 0, 'skill': 0,
                 'role': 'dead', 'archetype': 'unknown', 'current_action': None,
                 'target': None, 'loans_taken': 0, 'loans_given': 0, 'step': self.current_step,
                 'last_action_outcome': 'agent_dead', 'is_starving': True, 'happiness': 0,
                 'pos': None, 'emergency_food': True, 'default_cooldown': 0
             }

        # Get last known action outcome and starving status from the agent object
        # These should be updated by the agent's step method
        action_outcome = getattr(agent, 'last_action_outcome', None)
        is_starving = getattr(agent, 'last_starving_status', False)

        return {
            'status': 'alive',
            'wealth': agent.wealth,
            'food': agent.inventory.get('food', 0),
            'wood': agent.inventory.get('wood', 0),
            'skill': agent.skill,
            'role': agent.role,
            'archetype': agent.archetype,
            'current_action': agent.current_action_type, # Action intent set by RL or agent logic
            'target': agent.target,
            'loans_taken': len(agent.loans_taken),
            'loans_given': len(agent.loans_given),
            'step': self.current_step,
            'last_action_outcome': action_outcome, # Outcome of the *previous* step's action execution
            'is_starving': is_starving, # Starvation status at the end of the *previous* step
            'happiness': agent.happiness,
            'pos': agent.pos,
            'emergency_food': agent.emergency_food_seeking,
            'default_cooldown': agent.default_cooldown
        }

    def _calculate_reward(self, prev_info):
        """ Calculates reward based on the change in state from prev_info to current state. """
        current_info = self._get_info() # Get state *after* the step

        # --- Penalties/Rewards for Terminal States ---
        if current_info['status'] == 'dead':
             logging.debug(f"RL Agent {self.rl_agent_id} died. Assigning terminal penalty.")
             # Consider subtracting prev_info['wealth'] to penalize dying with wealth?
             return -100.0 # Large penalty for dying

        reward = 0.0

        # --- Survival Reward ---
        # Small positive reward for each step alive (can be removed if other rewards dominate)
        # reward += 0.01

        # --- Wealth Change Reward ---
        # Direct reward for increasing wealth
        wealth_change = current_info['wealth'] - prev_info['wealth']
        # Normalize or scale wealth change? Let's use a simple multiplier for now.
        reward += wealth_change * 0.1 # Tunable weight

        # --- Food Security / Starvation Penalty ---
        # Penalty for being in starving state (cumulative potentially)
        if current_info['is_starving']:
            reward -= 1.0 # Significant penalty per step starving
        # Reward for increasing food inventory, especially if low?
        food_change = current_info['food'] - prev_info['food']
        if prev_info['food'] < Config.FOOD_TARGET_LEVEL * 0.5: # If was low on food
            reward += food_change * 0.5 # Reward getting food when low
        elif food_change > 0: # General small reward for gaining food otherwise
            reward += food_change * 0.05

        # --- Skill Improvement Reward ---
        skill_change = current_info['skill'] - prev_info['skill']
        reward += skill_change * 2.0 # Reward gaining skill

        # --- Happiness / Entertainment ---
        happiness_change = current_info['happiness'] - prev_info['happiness']
        reward += happiness_change * 0.5 # Reward happiness increases (e.g., from 'ent')

        # --- Task Success/Failure Rewards (using last_action_outcome) ---
        outcome = current_info.get('last_action_outcome') or ''
        if outcome:
            # Positive outcomes
            if 'harvested_' in outcome: reward += 0.05
            if 'bought_tool' in outcome: reward += 0.2
            if 'delivered_wood' in outcome: reward += 0.3 # Artisan success
            if 'sold_tool' in outcome: reward += 0.5 # Seller success
            if 'built_t2' in outcome: reward += 1.0
            if 'trained' in outcome: reward += 0.1
            if 'got_loan' in outcome: reward += 0.1 # Getting a loan can be enabling
            if 'started_project' in outcome: reward += 0.2 # Developer success
            if 'paid_loan' in outcome: reward += 0.15 # Reward repaying loans
            if 'bought_ent' in outcome: reward += 0.05 # Small reward for ent if happiness is captured elsewhere

            # Negative outcomes
            if 'failed' in outcome: reward -= 0.05 # General failure penalty
            if 'loan_denied' in outcome: reward -= 0.1
            if 'defaulted' in outcome: reward -= 5.0 # Significant penalty for defaulting

        # --- Action Choice Penalties/Rewards ---
        # Penalty for being idle?
        current_action_str = current_info.get('current_action') or '' # Ensure it's a string
        if current_action_str.startswith('idle'):
             reward -= 0.02 # Small penalty to discourage idling

        # --- Loan Penalties ---
        # Penalty for holding debt?
        reward -= current_info['loans_taken'] * 0.01 # Small penalty per active loan taken

        # --- Reward Shaping based on Archetype? ---
        # Example: Give 'survivalist' slightly higher reward for food increases
        # if current_info['archetype'] == 'survivalist' and food_change > 0:
        #     reward += food_change * 0.05 # Extra bonus

        # Clip reward to prevent extreme values (optional)
        # reward = np.clip(reward, -10, 10)

        return reward


    def _map_action_to_agent_task(self, action_int):
        """ Maps the discrete action from RL agent to simulation task for the Agent class.
            Sets agent.current_action_type and potentially agent.target.
        """
        task_name = self._action_to_task_map.get(action_int, 'idle')

        agent = self.rl_agent
        if not agent or agent not in self.economy.agents:
            logging.error(f"RL Agent {self.rl_agent_id} not found during action mapping. Cannot set task.")
            return # Agent is likely dead or removed

        eco = self.economy
        agent.current_action_type = task_name # Set the high-level goal/task
        agent.target = None # Reset target, will be set below based on task
        agent.trade_partner = None # Reset interaction partner
        agent.trade_item_type = None
        agent.trade_is_buy = False
        agent.current_loan_purpose = None # Reset loan details
        agent.current_loan_amount_needed = 0.0

        # --- Determine Target based on Task ---
        # The agent's internal 'decide_target' might refine this later, but this sets the initial goal.
        try:
            if task_name == 'harvest_food':
                # Find nearest available food node
                nodes = [n for n in eco.grid.nodes if n.resource_type == 'food' and n.capacity > 0]
                if nodes: agent.target = min(nodes, key=lambda n: eco._get_distance(agent.pos, n.pos)).pos
                else: agent.current_action_type = 'idle_no_food_node' # Task impossible, revert to idle

            elif task_name == 'harvest_wood':
                # Find nearest available wood node
                nodes = [n for n in eco.grid.nodes if n.resource_type == 'wood' and n.capacity > 0]
                if nodes: agent.target = min(nodes, key=lambda n: eco._get_distance(agent.pos, n.pos)).pos
                else: agent.current_action_type = 'idle_no_wood_node'

            elif task_name == 'train':
                agent.target = eco.train_center.pos

            elif task_name.startswith('buy_tool_') or \
                 task_name.startswith('go_get_tool_') or \
                 task_name == 'deliver_wood_to_factory':
                agent.target = eco.factory.pos

            elif task_name == 'buy_ent':
                agent.target = eco.store.pos

            elif task_name == 'build_tier2_device':
                agent.target = agent.pos # Action happens at current location

            elif task_name == 'start_project':
                # Developer role needs this. Find *any* empty spot nearby (simplified).
                # Agent logic might have better placement strategy.
                found_spot = False
                for dx in range(-2, 3):
                     for dy in range(-2, 3):
                         if dx == 0 and dy == 0: continue
                         px, py = agent.pos[0] + dx, agent.pos[1] + dy
                         if 0 <= px < eco.grid.width and 0 <= py < eco.grid.height:
                             if (px,py) not in [Config.TRAIN_CENTER_POS, Config.TOOL_FACTORY_POS, Config.STORE_POS] and \
                                not eco.grid.get_node_at((px,py)) and \
                                not any(p.pos == (px,py) for p in eco.projects):
                                 agent.target = (px,py); found_spot = True; break
                     if found_spot: break
                if not found_spot: agent.current_action_type = 'idle_no_project_spot'

            elif task_name.startswith('find_buyer_for_tool_'):
                # Seller action. Target nearest agent initially. Agent logic refines partner.
                tool_type = task_name.split('_')[-1]
                agent.trade_item_type = f'tool_{tool_type}' # Inform agent logic
                agent.trade_is_buy = False
                others = [a for a in eco.agents if a != agent]
                if others: agent.target = min(others, key=lambda a: eco._get_distance(agent.pos, a.pos)).pos
                else: agent.current_action_type = 'idle_no_agents_to_sell_tool'

            elif task_name.startswith('trade_sell_') or task_name.startswith('trade_buy_'):
                # General trade. Target nearest agent initially. Agent logic refines partner.
                agent.trade_is_buy = 'buy' in task_name
                agent.trade_item_type = 'food' if 'food' in task_name else 'wood'
                others = [a for a in eco.agents if a != agent]
                if others: agent.target = min(others, key=lambda a: eco._get_distance(agent.pos, a.pos)).pos
                else: agent.current_action_type = 'idle_no_agents_to_trade'

            elif task_name == 'seek_loan_for_tool':
                # General loan seeking. Target nearest *potential* lender initially. Agent logic refines.
                agent.current_loan_purpose = 'tool' # Inform agent logic
                agent.current_loan_amount_needed = eco.factory.price * 1.1 if eco.factory.price > 0 else Config.COST_PER_SKILL # Need loan for tool or maybe training
                # Simplified initial target: nearest agent who isn't heavily indebted or in default cooldown
                lenders = [a for a in eco.agents if a != agent and a.default_cooldown == 0 and len(a.loans_taken) < 2]
                if lenders: agent.target = min(lenders, key=lambda a: eco._get_distance(agent.pos, a.pos)).pos
                else: agent.current_action_type = 'idle_no_potential_lenders'

            elif task_name == 'idle':
                agent.target = None # Explicitly no target for idle

            # Log the mapped action and preliminary target
            logging.debug(f"RL Agent {agent.id} mapped action {action_int} to task '{agent.current_action_type}' with initial target {agent.target}")

        except Exception as e:
             logging.error(f"Error during RL action mapping for task '{task_name}': {e}", exc_info=True)
             agent.current_action_type = 'idle_mapping_error' # Fallback to idle on error
             agent.target = None


    def step(self, action):
        # --- Pre-step checks ---
        agent = self.rl_agent
        if not agent or agent not in self.economy.agents:
             # RL agent died or was removed *before* this step call somehow
             logging.warning(f"RL Agent {self.rl_agent_id} not found at the beginning of step {self.current_step}.")
             obs = np.zeros(self.observation_space.shape, dtype=np.float32) # Zero obs for dead agent
             info = self._get_info() # Will report agent as dead
             # Return 0 reward, terminated=True, truncated=False, info
             return obs, 0.0, True, False, info

        # --- Store previous state info for reward calculation ---
        prev_info = self._get_info()

        # --- Set RL Agent's Intended Action ---
        # The RL action determines the agent's *goal* for the tick.
        # The agent's internal step logic will handle movement, execution, and potential
        # overrides due to emergencies (like starving).
        self._map_action_to_agent_task(action)

        # --- Step the Simulation ---
        # The Economy.step() iterates through all agents (including the RL one)
        # and calls their respective step() methods.
        # The Agent.step() method for the RL agent should respect self.current_action_type
        # set by _map_action_to_agent_task, rather than running its full decide_target() logic.
        # (This requires modification in Agent.step as previously noted).
        sim_status = self.economy.step() # This advances the simulation by one tick

        # --- Refresh RL agent reference after step (in case it died) ---
        self.rl_agent = next((a for a in self.economy.agents if a.id == self.rl_agent_id), None)

        # --- Post-step checks and return values ---
        terminated = False
        if not self.rl_agent: # RL agent died during this step
            terminated = True
            logging.info(f"RL Agent {self.rl_agent_id} died during step {self.current_step}.")
        elif sim_status == "game_over": # Simulation ended (e.g., all agents died)
            terminated = True
            logging.info(f"Simulation ended (game_over) during step {self.current_step}.")

        # Check truncation based on max_steps
        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        # Get new observation and info
        observation = self._get_obs() # Observation after the step
        info = self._get_info()       # Info after the step

        # Calculate reward based on state change during the step
        reward = self._calculate_reward(prev_info) # Compare prev_info with current state reflected in info

        # Render if requested
        if self.render_mode == "human":
            self._render_frame()

        # Log step details
        logging.debug(
            f"Step {self.current_step-1} -> {self.current_step} | "
            f"RL_ID: {self.rl_agent_id}, Action: {action} ({prev_info.get('current_action', 'N/A')}), "
            f"Outcome: {info.get('last_action_outcome', 'N/A')}, "
            f"Reward: {reward:.3f}, Wealth: {info.get('wealth', 0):.2f}, "
            f"Term: {terminated}, Trunc: {truncated}"
        )

        return observation, reward, terminated, truncated, info


    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Important for reproducibility via seeding

        # Re-initialize the entire economy simulation
        # Use stored initial parameters, but apply the new seed if provided
        current_seed = seed if seed is not None else Config.SEED
        self.economy = Economy(
            num_agents=self.sim_num_agents,
            num_resources=self.sim_num_resources,
            seed=current_seed,
            rl_agent_id=self.rl_agent_id # Make sure the RL agent is correctly identified
        )
        self.current_step = 0

        # Reset RL agent reference
        self.rl_agent = next((a for a in self.economy.agents if a.id == self.rl_agent_id), None)
        if not self.rl_agent:
             # This should not happen if Economy init is correct, but good practice to check
             raise ValueError(f"RL agent {self.rl_agent_id} not found after reset.")
        if not self.rl_agent.is_rl_controlled:
             logging.warning(f"RL agent {self.rl_agent_id} not marked as controlled after reset.")
             self.rl_agent.is_rl_controlled = True # Ensure it's set

        # Get initial observation and info
        observation = self._get_obs()
        info = self._get_info()

        # Reset rendering resources if they exist
        # Since resources are now local to render methods, reset logic here is not needed.
        # if self.window is not None:
        #      self._render_init() # Re-init pygame window for the new sim instance

        logging.info(f"Environment reset. RL Agent {self.rl_agent_id} initialized.")

        return observation, info


    def _render_init(self):
         """Initialize Pygame resources for rendering."""
         if self.render_mode == "human":
             pygame.init()
             pygame.display.init()
             # Increased height for info panel below grid
             window = pygame.display.set_mode((Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT + 150))
             pygame.display.set_caption(f"Agent Economy Gym Env (RL Agent: {self.rl_agent_id})")


    def _render_frame(self):
        """Render the current state of the environment. Manages Pygame resources locally."""
        if self.render_mode is None:
            gym.logger.warn("Cannot render without specifying render_mode during environment creation.")
            return

        # --- Initialize Pygame and Resources Locally ---
        # This block ensures Pygame is initialized only when needed for human rendering.
        window = None
        clock = None
        font = None
        small_font = None
        info_font = None

        if self.render_mode == "human":
             if not pygame.get_init(): # Initialize pygame if not already running
                  pygame.init()
                  pygame.display.init()
                  logging.info("Pygame initialized for rendering.")

             # Get the display surface (or create if needed - typically done once)
             # We assume a display surface exists if pygame is initialized.
             try:
                 # Attempt to get the current display surface. If none, create one.
                 window = pygame.display.get_surface()
                 if window is None:
                     window = pygame.display.set_mode((Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT + 150))
                     pygame.display.set_caption(f"Agent Economy Gym Env (RL Agent: {self.rl_agent_id})")
                     logging.info("Pygame window created.")
             except pygame.error as e: # Handle cases where display might not be available
                 logging.error(f"Pygame display error during rendering: {e}")
                 return # Cannot render

             clock = pygame.time.Clock() # Create clock for FPS control

             # Load fonts locally each time (or manage globally if performance critical)
             try:
                 font = pygame.font.SysFont(None, 20)
                 small_font = pygame.font.SysFont(None, 16)
                 info_font = pygame.font.SysFont(None, 18)
             except Exception as e:
                 logging.error(f"Pygame SysFont error: {e}. Using default font.")
                 font = pygame.font.Font(None, 20)
                 small_font = pygame.font.Font(None, 16)
                 info_font = pygame.font.Font(None, 18)

        screen = window # Use the locally obtained/created window surface
        eco = self.economy # Reference to the current economy instance

        # --- Colors --- MOVED TO TOP LEVEL
        # WHITE=(255, 255, 255); BLACK=(0, 0, 0); GREY=(200, 200, 200); LIGHT_GREY=(220, 220, 220)
        # RED=(255, 0, 0); BLUE=(0, 0, 255); GREEN=(0, 255, 0); YELLOW=(255, 255, 0); MAGENTA=(255,0,255)
        # Use role colors consistent with Economy.run_pygame
        role_colors = {
            'food_harvester': (34, 139, 34), 'wood_harvester': (139, 69, 19),
            'seller': (255, 165, 0), 'developer': (75, 0, 130),
            'idle': (128, 128, 128) # Add other idle states if needed
        }
        # Use archetype markers consistent with Economy.run_pygame
        arch_markers = {'survivalist': 'S', 'risk_taker': 'R', 'trader': 'T', 'tool_artisan': 'A'}

        # --- Drawing ---
        screen.fill(LIGHT_GREY)
        grid_area_height = Config.WINDOW_HEIGHT

        # Grid lines
        for x in range(0, Config.WINDOW_WIDTH + 1, Config.CELL_SIZE): pygame.draw.line(screen, GREY, (x, 0), (x, grid_area_height))
        for y in range(0, grid_area_height + 1, Config.CELL_SIZE): pygame.draw.line(screen, GREY, (0, y), (Config.WINDOW_WIDTH, y))

        # Structures (Train Center, Factory, Store)
        cs = Config.CELL_SIZE
        try:
             tx, ty = eco.train_center.pos; pygame.draw.rect(screen, YELLOW, (tx*cs, ty*cs, cs, cs)); screen.blit(small_font.render("T", True, BLACK), (tx*cs+5, ty*cs+5))
             fx, fy = eco.factory.pos; pygame.draw.rect(screen, (100,100,100), (fx*cs, fy*cs, cs, cs)); screen.blit(small_font.render("F", True, WHITE), (fx*cs+5, fy*cs+5))
             sx, sy = eco.store.pos; pygame.draw.rect(screen, (0,200,200), (sx*cs, sy*cs, cs, cs)); screen.blit(small_font.render("S", True, BLACK), (sx*cs+5, sy*cs+5))
        except AttributeError as e:
             logging.error(f"Error rendering structures: {e}. Economy object might be missing attributes.")

        # Resource Nodes
        for node in eco.grid.nodes:
            cx, cy = node.pos; col = (160,82,45) if node.resource_type == 'wood' else (60,179,113)
            pygame.draw.rect(screen, col, (cx*cs+2, cy*cs+2, cs-4, cs-4))
            # Use a helper to render text safely
            cap_text = f"{node.capacity:.0f}"
            _render_text_safe(screen, cap_text, (cx*cs+3, cy*cs + cs//2 - 5), small_font, BLACK)

        # Projects
        for proj in eco.projects:
            px, py = proj.pos; pygame.draw.rect(screen, MAGENTA, (px*cs, py*cs, cs, cs), 3)
            proj_text = f"P{proj.time_left}"
            _render_text_safe(screen, proj_text, (px*cs+2, py*cs+2), small_font, BLACK)

        # Agents
        for agent in eco.agents:
            ax, ay = agent.pos; px = ax*cs + cs//2; py = ay*cs + cs//2
            role_color = role_colors.get(agent.role, BLACK) # Default to black if role unknown

            # --- Visual Indicators ---
            # Highlight RL agent distinctly
            if agent.id == self.rl_agent_id:
                pygame.draw.circle(screen, BLUE, (px, py), cs//3 + 6, 4) # Thicker blue outline

            # Starvation indicator
            if agent.emergency_food_seeking:
                 pygame.draw.circle(screen, RED, (px, py), cs//3 + 4, 3) # Red outline for emergency

            # Default cooldown indicator (optional)
            # if agent.default_cooldown > 0: pygame.draw.circle(screen, (100, 100, 255), (px, py), cs//3 + 8, 2)

            # --- Agent Body ---
            pygame.draw.circle(screen, role_color, (px, py), cs//3)

            # --- Agent Label (ID + Archetype) ---
            agent_str = f"{agent.id}{arch_markers.get(agent.archetype, '?')}"
            text_color = WHITE if sum(role_color) < 384 else BLACK # Contrast text color
            _render_text_safe(screen, agent_str, (px, py), small_font, text_color, center=True)


        # --- Info Panel (Below Grid) ---
        info_y = Config.WINDOW_HEIGHT + 5
        line_h = 18 # Height per line of text
        col1_x = 5 # Left column X
        col2_x = Config.WINDOW_WIDTH // 2 # Right column X (approx)

        # General Sim Info (Top-Left)
        _render_text_safe(screen, f"Tick: {self.current_step}/{self.max_steps} | Agents: {len(eco.agents)}", (col1_x, info_y), font)
        _render_text_safe(screen, f"Prices: W ${eco.price_w:.2f} F ${eco.price_f:.2f}", (col1_x, info_y + line_h), font)
        # Add Factory/Store info if needed (can get long)
        # _render_text_safe(screen, f"Factory: WTools {eco.factory.wood_tools} ...", (col1_x, info_y + 2*line_h), self.font)

        # RL Agent Specific Info (Right Side)
        info = self._get_info() # Get latest info for the RL agent
        _render_text_safe(screen, f"--- RL Agent {self.rl_agent_id} ({info.get('archetype','?')}) ---", (col2_x, info_y), font, BLUE)
        if info['status'] == 'alive':
             action = info.get('current_action', 'N/A')
             outcome = info.get('last_action_outcome', 'N/A') or 'N/A' # Handle None
             starving = info.get('is_starving', False)
             happiness = info.get('happiness', 0)

             _render_text_safe(screen, f" Role: {info.get('role', 'N/A')}", (col2_x, info_y + line_h), info_font)
             _render_text_safe(screen, f" Wealth: ${info.get('wealth', 0):.2f} | Food: {info.get('food', 0):.1f} | Wood: {info.get('wood', 0):.1f}", (col2_x, info_y + 2*line_h), info_font)
             _render_text_safe(screen, f" Skill: {info.get('skill', 0):.2f} | Happy: {happiness:.2f}", (col2_x, info_y + 3*line_h), info_font)
             _render_text_safe(screen, f" Action: {action} | Target: {info.get('target', 'N/A')}", (col2_x, info_y + 4*line_h), info_font)
             _render_text_safe(screen, f" Last Outcome: {outcome}", (col2_x, info_y + 5*line_h), info_font)
             _render_text_safe(screen, f" Starving: {starving}", (col2_x, info_y + 6*line_h), info_font, RED if starving else BLACK)
             # Add loan info if needed
             # _render_text_safe(screen, f" Loans Tkn:{info['loans_taken']} Gvn:{info['loans_given']}", (col2_x, info_y + 7*line_h), self.info_font)
        else:
            _render_text_safe(screen, " Agent is DEAD", (col2_x, info_y + line_h), font, RED)


        # --- Finalize Frame ---
        if self.render_mode == "human":
            pygame.event.pump() # Process internal events
            pygame.display.flip() # Update the full display Surface to the screen
            if clock: clock.tick(self.metadata["render_fps"]) # Maintain frame rate using local clock
        else: # rgb_array
             # Return pixel data as numpy array (H, W, C)
             # Transpose from (W, H, C) returned by pixels3d to (H, W, C) expected by Gym
             return np.transpose(np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2))

    def render(self, mode='human'):
         """Public render method called by Gym."""
         # Use self.render_mode set during initialization
         render_mode_to_use = self.render_mode
         if render_mode_to_use == "rgb_array":
              # Note: For rgb_array, Pygame might not need a visible window,
              # but it often still requires pygame.init() and a surface.
              # The local management in _render_frame might need adjustment if
              # rgb_array rendering is frequently used without 'human'.
              # For now, assume _render_frame handles necessary setup.
              return self._render_frame() # Returns numpy array
         elif render_mode_to_use == "human":
              self._render_frame() # Updates display
              return True # Indicate success for human mode
         else:
              # Handle other modes or cases where render_mode is None
              super(AgentEconomyEnv, self).render() # Call parent render method if needed
              return

    def close(self):
        """Clean up resources (Pygame window)."""
        # Quit pygame only if it was initialized
        if pygame.get_init():
             pygame.display.quit()
             pygame.quit()
             logging.info("Pygame shut down.")

# --- End Gymnasium Environment ---