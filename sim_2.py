import pkgutil
# Monkey-patch for pkgutil.ImpImporter on Python 3.12+
if not hasattr(pkgutil, 'ImpImporter'):
    pkgutil.ImpImporter = type('ImpImporter', (), {})

import random
import logging
import sys
from typing import List, Optional, Tuple, Dict, Any, Union
import numpy as np
import pygame
import gymnasium as gym # NEW IMPORT
from gymnasium import spaces # NEW IMPORT
from stable_baselines3 import PPO  # Or A2C, DQN etc. # NEW IMPORT
from stable_baselines3.common.env_checker import check_env # NEW IMPORT
from stable_baselines3.common.vec_env import DummyVecEnv # NEW IMPORT
from stable_baselines3.common.vec_env import VecNormalize # NEW IMPORT
from gymnasium.envs.registration import register # NEW IMPORT
import functools # NEW IMPORT
import pickle # NEW IMPORT FOR LOADING STATS

from config import Config # MODIFIED IMPORT
from core_components import Grid, ResourceNode, TrainingCenter, ToolFactory, Store, Project, Loan # NEW IMPORT
from agent import Agent # NEW IMPORT
# from rl_env import AgentEconomyEnv # MOVED IMPORT

# Configuration constants
# CLASS Config REMOVED FROM HERE

# Logging setup
def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG, # Changed from logging.INFO
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# --- Environment Configuration (moved to top level) ---
RL_AGENT_ID = 0  # Choose which agent ID will be RL controlled
NUM_OTHER_AGENTS = 4 # Number of other (scripted) agents
TOTAL_AGENTS = NUM_OTHER_AGENTS + 1
NUM_RESOURCES_NODES = 10
MAX_STEPS_PER_EPISODE = 500
RENDER_MODE = None # "human" or None


# --- Helper function for RL observation normalization and action mapping ---
def _normalize_obs_for_pygame(value, max_val, clip_min=0, clip_max=1):
    if max_val == 0: return float(clip_min)
    normalized = float(value) / float(max_val)
    return np.clip(normalized, clip_min, clip_max)

def _one_hot_encode_for_pygame(category: str, category_list: list) -> list:
    encoding = [0.0] * len(category_list)
    try:
        index = category_list.index(category)
        encoding[index] = 1.0
    except ValueError:
        logging.warning(f"Category '{category}' not found in list {category_list} for one-hot encoding.")
    return encoding

ACTION_TO_TASK_MAP_PYGAME = { # Copied from rl_env.py
    0: 'idle', 1: 'harvest_food', 2: 'harvest_wood', 3: 'train',
    4: 'buy_tool_wood', 5: 'buy_tool_food', 6: 'buy_ent',
    7: 'deliver_wood_to_factory', 8: 'go_get_tool_from_factory_wood',
    9: 'go_get_tool_from_factory_food', 10: 'find_buyer_for_tool_wood',
    11: 'find_buyer_for_tool_food', 12: 'build_tier2_device',
    13: 'seek_loan_for_tool', 14: 'trade_sell_food', 15: 'trade_sell_wood',
    16: 'trade_buy_food', 17: 'trade_buy_wood', 18: 'start_project'
}

def _get_observation_for_pygame(agent: Agent, economy: 'Economy') -> np.ndarray:
    """Constructs observation vector for a given agent in the economy."""
    eco = economy # Alias for clarity

    max_wealth_norm = 500; max_skill_norm = 5; max_inv_norm = 50; max_tools_norm = 10; max_loans_norm = 5
    max_price_norm = 20; max_fac_stock_norm = 50; max_store_stock_norm = 50; max_dist_norm = Config.GRID_WIDTH + Config.GRID_HEIGHT; max_cap_norm = 200
    max_hunger_norm = Config.HUNGER_INTERVAL * 1.5

    wealth = agent.wealth
    skill = _normalize_obs_for_pygame(agent.skill, max_skill_norm)
    food_inv = _normalize_obs_for_pygame(agent.inventory.get('food', 0), max_inv_norm)
    wood_inv = _normalize_obs_for_pygame(agent.inventory.get('wood', 0), max_inv_norm)
    ent_inv = _normalize_obs_for_pygame(agent.inventory.get('ent', 0), max_inv_norm)
    has_t2 = 1.0 if agent.inventory.get('has_tier2_device', False) else 0.0
    tools_w = _normalize_obs_for_pygame(agent.tools.get('wood', 0), max_tools_norm)
    tools_f = _normalize_obs_for_pygame(agent.tools.get('food', 0), max_tools_norm)
    sale_w = _normalize_obs_for_pygame(agent.inventory.get('tools_for_sale', {}).get('wood', 0), max_tools_norm)
    sale_f = _normalize_obs_for_pygame(agent.inventory.get('tools_for_sale', {}).get('food', 0), max_tools_norm)
    pos_x = _normalize_obs_for_pygame(agent.pos[0], Config.GRID_WIDTH)
    pos_y = _normalize_obs_for_pygame(agent.pos[1], Config.GRID_HEIGHT)
    happiness = np.clip(agent.happiness, 0, 1)
    hunger = _normalize_obs_for_pygame(agent.food_hunger, max_hunger_norm)
    default_cd = 1.0 if agent.default_cooldown > 0 else 0.0
    loans_taken_count = _normalize_obs_for_pygame(len(agent.loans_taken), max_loans_norm)
    loans_given_count = _normalize_obs_for_pygame(len(agent.loans_given), max_loans_norm)

    price_w_norm = _normalize_obs_for_pygame(economy.price_w, max_price_norm)
    price_f_norm = _normalize_obs_for_pygame(economy.price_f, max_price_norm)
    fac_w_tools_norm = _normalize_obs_for_pygame(economy.factory.wood_tools, max_fac_stock_norm)
    fac_f_tools_norm = _normalize_obs_for_pygame(economy.factory.food_tools, max_fac_stock_norm)
    fac_price_norm = _normalize_obs_for_pygame(economy.factory.price, max_price_norm * 2)
    store_ent_norm = _normalize_obs_for_pygame(economy.store.ent_stock, max_store_stock_norm)
    store_price_norm = _normalize_obs_for_pygame(economy.store.price, max_price_norm * 2)

    dist_w, cap_w, dist_f, cap_f = max_dist_norm, 0, max_dist_norm, 0
    wood_nodes = [n for n in economy.grid.nodes if n.resource_type == 'wood' and n.capacity > 0]
    food_nodes = [n for n in economy.grid.nodes if n.resource_type == 'food' and n.capacity > 0]
    if wood_nodes:
        nearest_w = min(wood_nodes, key=lambda n: economy._get_distance(agent.pos, n.pos))
        dist_w = economy._get_distance(agent.pos, nearest_w.pos); cap_w = nearest_w.capacity
    if food_nodes:
        nearest_f = min(food_nodes, key=lambda n: economy._get_distance(agent.pos, n.pos))
        dist_f = economy._get_distance(agent.pos, nearest_f.pos); cap_f = nearest_f.capacity

    dist_w_norm = _normalize_obs_for_pygame(dist_w, max_dist_norm)
    cap_w_norm = _normalize_obs_for_pygame(cap_w, max_cap_norm)
    dist_f_norm = _normalize_obs_for_pygame(dist_f, max_dist_norm)
    cap_f_norm = _normalize_obs_for_pygame(cap_f, max_cap_norm)

    dist_train_norm = _normalize_obs_for_pygame(economy._get_distance(agent.pos, economy.train_center.pos), max_dist_norm)
    dist_factory_norm = _normalize_obs_for_pygame(economy._get_distance(agent.pos, economy.factory.pos), max_dist_norm)
    dist_store_norm = _normalize_obs_for_pygame(economy._get_distance(agent.pos, economy.store.pos), max_dist_norm)

    dist_agent_norm = 1.0
    other_agents = [a for a in economy.agents if a.id != agent.id]
    if other_agents:
        nearest_a = min(other_agents, key=lambda a: economy._get_distance(agent.pos, a.pos))
        dist_agent_norm = _normalize_obs_for_pygame(economy._get_distance(agent.pos, nearest_a.pos), max_dist_norm)

    role_encoded = _one_hot_encode_for_pygame(agent.role, Config.ROLES)
    archetype_encoded = _one_hot_encode_for_pygame(agent.archetype, Config.AGENT_ARCHETYPES)

    obs_list = [
        wealth, skill, food_inv, wood_inv, ent_inv, has_t2, tools_w, tools_f, sale_w, sale_f,
        pos_x, pos_y, happiness, hunger, default_cd, loans_taken_count, loans_given_count,
        price_w_norm, price_f_norm, fac_w_tools_norm, fac_f_tools_norm, fac_price_norm,
        store_ent_norm, store_price_norm, dist_w_norm, cap_w_norm, dist_f_norm, cap_f_norm,
        dist_train_norm, dist_factory_norm, dist_store_norm, dist_agent_norm
    ]
    obs_list.extend(role_encoded)
    obs_list.extend(archetype_encoded)
    return np.array(obs_list, dtype=np.float32)

def _map_action_to_agent_task_for_pygame(agent: Agent, economy: 'Economy', action_int: int):
    """Maps the discrete action to simulation task for the Agent class."""
    task_name = ACTION_TO_TASK_MAP_PYGAME.get(action_int, 'idle')
    eco = economy

    agent.current_action_type = task_name
    agent.target = None
    agent.trade_partner = None
    agent.trade_item_type = None
    agent.trade_is_buy = False
    agent.current_loan_purpose = None
    agent.current_loan_amount_needed = 0.0

    try:
        if task_name == 'harvest_food':
            nodes = [n for n in eco.grid.nodes if n.resource_type == 'food' and n.capacity > 0]
            if nodes: agent.target = min(nodes, key=lambda n: eco._get_distance(agent.pos, n.pos)).pos
            else: agent.current_action_type = 'idle_no_food_node'
        elif task_name == 'harvest_wood':
            nodes = [n for n in eco.grid.nodes if n.resource_type == 'wood' and n.capacity > 0]
            if nodes: agent.target = min(nodes, key=lambda n: eco._get_distance(agent.pos, n.pos)).pos
            else: agent.current_action_type = 'idle_no_wood_node'
        elif task_name == 'train':
            agent.target = eco.train_center.pos
        elif task_name.startswith('buy_tool_') or task_name.startswith('go_get_tool_') or task_name == 'deliver_wood_to_factory':
            agent.target = eco.factory.pos
        elif task_name == 'buy_ent':
            agent.target = eco.store.pos
        elif task_name == 'build_tier2_device':
            agent.target = agent.pos
        elif task_name == 'start_project':
            found_spot = False
            for _try in range(15):
                px, py = random.randrange(eco.grid.width), random.randrange(eco.grid.height)
                if (px,py) not in [Config.TRAIN_CENTER_POS, Config.TOOL_FACTORY_POS, Config.STORE_POS] and \
                   not eco.grid.get_node_at((px,py)) and not any(p.pos == (px,py) for p in eco.projects):
                    agent.target = (px,py); found_spot = True; break
            if not found_spot: agent.current_action_type = 'idle_no_project_spot'
        elif task_name.startswith('find_buyer_for_tool_'):
            tool_type = task_name.split('_')[-1]
            agent.trade_item_type = f'tool_{tool_type}'
            agent.trade_is_buy = False
            others = [a for a in eco.agents if a != agent]
            if others: agent.target = min(others, key=lambda a: eco._get_distance(agent.pos, a.pos)).pos
            else: agent.current_action_type = 'idle_no_agents_to_sell_tool'
        elif task_name.startswith('trade_sell_') or task_name.startswith('trade_buy_'):
            agent.trade_is_buy = 'buy' in task_name
            agent.trade_item_type = 'food' if 'food' in task_name else 'wood'
            others = [a for a in eco.agents if a != agent]
            if others: agent.target = min(others, key=lambda a: eco._get_distance(agent.pos, a.pos)).pos
            else: agent.current_action_type = 'idle_no_agents_to_trade'
        elif task_name == 'seek_loan_for_tool':
            agent.current_loan_purpose = 'tool'
            agent.current_loan_amount_needed = eco.factory.price * 1.1 if eco.factory.price > 0 else Config.COST_PER_SKILL
            lenders = [a for a in eco.agents if a != agent and a.default_cooldown == 0 and len(a.loans_taken) < 2]
            if lenders: agent.target = min(lenders, key=lambda a: eco._get_distance(agent.pos, a.pos)).pos
            else: agent.current_action_type = 'idle_no_potential_lenders'
        elif task_name == 'idle':
            agent.target = None
    except Exception as e:
        logging.error(f"Error mapping RL action '{task_name}' for agent {agent.id}: {e}", exc_info=True)
        agent.current_action_type = 'idle_mapping_error'; agent.target = None
# --- End Helper functions ---


# --- Helper function to create the environment (moved to top level)---
def make_env(num_agents_param: int, num_resources_param: int, rl_agent_id_param: int, max_steps_param: int, render_mode_param: Optional[str]): # REMAINS make_env
    env_args = {
        'num_agents': num_agents_param,
        'num_resources': num_resources_param,
        'rl_agent_id': rl_agent_id_param,
        'max_steps': max_steps_param,
        'render_mode': render_mode_param
    }
    return gym.make('AgentEconomy-v0', **env_args)

class Economy:
    def __init__(self, num_agents: int, num_resources: int, seed: Optional[int] = None, rl_agent_id: Optional[int] = None): # Added rl_agent_id
        if seed is not None: random.seed(seed); np.random.seed(seed)
        
        self.grid=Grid(Config.GRID_WIDTH,Config.GRID_HEIGHT)
        self.train_center=TrainingCenter(Config.TRAIN_CENTER_POS)
        self.factory=ToolFactory(Config.TOOL_FACTORY_POS)
        self.store=Store(Config.STORE_POS)
        self.projects:List[Project]=[] # Projects initiated by agents
        self.all_loans: List[Loan] = [] # Global list of all active loans
        Loan.next_loan_id = 0 # Reset loan IDs for new simulation

        for _ in range(num_resources):
            rt=random.choice(['wood','food'])
            while True: # Ensure nodes are not placed on special buildings
                nx, ny = random.randrange(self.grid.width), random.randrange(self.grid.height)
                if (nx,ny) not in [Config.TRAIN_CENTER_POS, Config.TOOL_FACTORY_POS, Config.STORE_POS]:
                    break
            n=ResourceNode(nx,ny,rt,random.uniform(70,180)) # Varied initial capacity
            self.grid.add_node(n)
        
        self.agents: List[Agent]=[] 
        for i in range(num_agents):
            # Standard Agent creation only now
            a=Agent(i,self.grid,self.train_center,self.factory,self.store,self.projects, self.agents)
            # **** MODIFICATION START ****
            if rl_agent_id is not None and i == rl_agent_id:
                 a.is_rl_controlled = True # This flag now signals Pygame loop to use policy
                 logging.info(f"Agent {i} marked as RL controlled for Pygame simulation.")
            # **** MODIFICATION END ****
            self.agents.append(a)
            
        self.price_w=Config.INITIAL_PRICE_WOOD
        self.price_f=Config.INITIAL_PRICE_FOOD
        
        # Initialize history
        self.history={
            'prices': [], 
            'agent_wealth': [[] for _ in range(num_agents)], 
            'agent_food': [[] for _ in range(num_agents)], 
            'agent_roles': [[] for _ in range(num_agents)],
            'agent_archetypes': [self.agents[i].archetype for i in range(num_agents)], # Store initial archetypes
            'loans_active_count': [], 
            'total_loan_volume_tick': [], # Sum of principal of new loans this tick
            'num_defaults_tick': [] # Number of new defaults this tick
        }
        # Initialize history lists for tick 0 to avoid index errors on first step
        current_tick = 0 # Represents state before first step
        for i in range(num_agents):
            self.history['agent_wealth'][i].append(self.agents[i].wealth)
            self.history['agent_food'][i].append(self.agents[i].inventory['food'])
            self.history['agent_roles'][i].append(self.agents[i].role)
        self.history['loans_active_count'].append(0)
        self.history['total_loan_volume_tick'].append(0)
        self.history['num_defaults_tick'].append(0)


    def general_p2p_trade(self): 
        # This is the general, somewhat random P2P market trade.
        # It can be made less frequent if targeted trades by agents are the primary mechanism.
        trades_this_tick_details=[] # For logging or more detailed history if needed
        if len(self.agents) < 2: return
        
        # Scale number of random trades, e.g., based on a fraction of agents
        num_random_trades = max(1, (len(self.agents) * Config.NUM_TRADES_PER_TICK) // 10)

        for _ in range(num_random_trades): 
            seller_agent = random.choice(self.agents)
            buyer_agent = random.choice(self.agents)
            if seller_agent is buyer_agent: continue

            # Buyer prioritizes food if very hungry, otherwise random available item from seller
            resource_to_trade = None
            if buyer_agent.inventory['food'] < buyer_agent._archetype_food_target() / 2 and seller_agent.inventory.get('food',0) > 1 :
                resource_to_trade = 'food'
            else: # Try to trade other goods
                available_from_seller = [item for item, quant in seller_agent.inventory.items() if isinstance(quant, (int,float)) and quant > 0 and item in ['wood', 'food', 'ent']]
                if available_from_seller:
                    resource_to_trade = random.choice(available_from_seller)
            
            if resource_to_trade and seller_agent.inventory.get(resource_to_trade, 0) > 0:
                market_price = self.price_f if resource_to_trade == 'food' else (self.price_w if resource_to_trade == 'wood' else self.store.price)
                if market_price <= 0: market_price = Config.MIN_PRICE # Ensure positive price
                
                # Simplified negotiation: Price around market price with some spread
                # Traders might try to get better deals even in random trades
                price_multiplier_buyer = 1.0
                price_multiplier_seller = 1.0
                if buyer_agent.archetype == 'trader': price_multiplier_buyer = random.uniform(0.85, 0.95)
                if seller_agent.archetype == 'trader': price_multiplier_seller = random.uniform(1.05, 1.15)

                buyer_max_pay = market_price * price_multiplier_buyer * random.uniform(0.9, 1.1) 
                seller_min_accept = market_price * price_multiplier_seller * random.uniform(0.9, 1.1)
                
                buyer_max_pay = max(Config.MIN_PRICE, buyer_max_pay)
                seller_min_accept = max(Config.MIN_PRICE, seller_min_accept)

                if buyer_agent.wealth >= seller_min_accept and buyer_max_pay >= seller_min_accept:
                    trade_price = (buyer_max_pay + seller_min_accept) / 2.0
                    trade_price = min(trade_price, buyer_agent.wealth) 
                    trade_price = max(Config.MIN_PRICE, trade_price)

                    # Execute trade
                    seller_agent.inventory[resource_to_trade] -= 1
                    seller_agent.wealth += trade_price
                    buyer_agent.inventory[resource_to_trade] = buyer_agent.inventory.get(resource_to_trade, 0) + 1
                    buyer_agent.wealth -= trade_price
                    
                    logging.debug(f"General P2P Trade: Buyer A:{buyer_agent.id} bought {resource_to_trade} from Seller A:{seller_agent.id} for ${trade_price:.2f}")
                    trades_this_tick_details.append({'buyer': buyer_agent.id, 'seller': seller_agent.id, 'item': resource_to_trade, 'price': round(trade_price,2)})
        
        if trades_this_tick_details: # Add to detailed trade history if needed
             # self.history['trades'].append(trades_this_tick_details) # Assuming 'trades' key exists if used
             logging.debug(f"General P2P trades completed: {len(trades_this_tick_details)} trades.")


    def step(self):
        current_tick = len(self.history['prices']) # This is the tick number we are currently processing (0-indexed for lists)

        # --- Price Adjustment ---
        # Demand proxy: sum of skills of harvesters + needs
        demand_w = sum(a.skill for a in self.agents if a.role == 'wood_harvester' or (a.role == 'developer' and a.inventory['wood'] < Config.PROJECT_WOOD_COST) or (not a.inventory['has_tier2_device'] and a.skill > Config.TIER2_MIN_SKILL_TO_BUILD and a.inventory['wood'] < Config.TIER2_WOOD_DEVICE_COST))
        supply_w = sum(n.capacity for n in self.grid.nodes if n.resource_type == 'wood') + sum(a.inventory['wood'] for a in self.agents if a.role == 'seller') # Seller stock contributes to supply
        
        base_food_need = len(self.agents) * (1.0/Config.HUNGER_INTERVAL) * 1.5 # Basic consumption, slightly amplified
        harvester_demand_f = sum(a.skill for a in self.agents if a.role == 'food_harvester')
        emergency_demand_f = sum(1 for a in self.agents if a.emergency_food_seeking) * 3 # Higher weight for emergency
        demand_f = base_food_need + harvester_demand_f + emergency_demand_f
        supply_f = sum(n.capacity for n in self.grid.nodes if n.resource_type == 'food') + sum(a.inventory['food'] for a in self.agents if a.role == 'seller')

        # Price adjustment factor to prevent extreme swings
        adj_factor = Config.PRICE_ADJUST_ALPHA

        self.price_w = max(Config.MIN_PRICE, self.price_w * (1 + adj_factor * (demand_w - supply_w) / max(supply_w, demand_w, 10.0)))
        self.price_f = max(Config.MIN_PRICE, self.price_f * (1 + adj_factor * (demand_f - supply_f) / max(supply_f, demand_f, 10.0)))
        
        logging.info(f"Tick {current_tick} Prices: W=${self.price_w:.2f}(D:{demand_w:.1f},S:{supply_w:.1f}), F=${self.price_f:.2f}(D:{demand_f:.1f},S:{supply_f:.1f})")
        self.history['prices'].append({'wood': self.price_w, 'food': self.price_f})

        # --- Pad History for Current Tick (before agent actions change states) ---
        # This ensures all history lists are of the same length as 'prices' before appending new data
        for i in range(len(self.history['agent_wealth'])): # Iterate over total number of initial agents
            if len(self.history['agent_wealth'][i]) == current_tick: # Agent i's lists needs padding for this new tick
                # Find agent by ID to get current state, or use None if dead
                agent_obj = next((a for a in self.agents if a.id == i), None)
                if agent_obj:
                    self.history['agent_wealth'][i].append(agent_obj.wealth)
                    self.history['agent_food'][i].append(agent_obj.inventory['food'])
                    self.history['agent_roles'][i].append(agent_obj.role)
                else: # Agent already died
                    self.history['agent_wealth'][i].append(None)
                    self.history['agent_food'][i].append(None)
                    self.history['agent_roles'][i].append("dead")
        # Pad loan history lists for the current tick before processing
        self.history['loans_active_count'].append(len(self.all_loans)) # Count from end of last tick
        self.history['total_loan_volume_tick'].append(0) # Will be summed up this tick
        self.history['num_defaults_tick'].append(0)      # Will be summed up this tick


        # --- Agent Actions ---
        survivors = []
        tick_defaults_counter = 0 # Reset for current tick
        new_loan_volume_this_tick_counter = 0 # Reset for current tick

        random.shuffle(self.agents) # Shuffle order of agent processing each tick

        for agent in list(self.agents): # Iterate over a copy in case agents die mid-loop (though shouldn't happen with current design)
            # Pass current_tick and self.all_loans (global list) to agent's step
            # The agent.step() method will now call _handle_loan_repayments_and_accruals internally, passing price_f
            agent_status_info = agent.step(self.price_w, self.price_f, current_tick, self.all_loans)
            status = agent_status_info['status'] if isinstance(agent_status_info, dict) else agent_status_info

            if status != 'dead':
                survivors.append(agent)
                # Update history for survivors (last element which was pre-padded or just added)
                self.history['agent_wealth'][agent.id][-1] = agent.wealth 
                self.history['agent_food'][agent.id][-1] = agent.inventory['food']
                self.history['agent_roles'][agent.id][-1] = agent.role
            else:
                logging.info(f"Agent {agent.id} ({agent.archetype[0]},{agent.role}) DIED. W:{agent.wealth:.1f} F:{agent.inventory['food']:.1f} H:{agent.food_hunger}")
                # Mark as dead in history (last element)
                self.history['agent_wealth'][agent.id][-1] = None
                self.history['agent_food'][agent.id][-1] = None
                self.history['agent_roles'][agent.id][-1] = "dead"
                
                # Handle outstanding loans of dead agent (e.g., lenders write them off, defaults are recorded)
                for loan in list(self.all_loans): # Iterate over copy for safe modification if needed
                    if loan.borrower_id == agent.id and not loan.is_defaulted:
                        loan.is_defaulted = True 
                        tick_defaults_counter +=1
                        logging.warning(f"Loan ID {loan.id} (Borrower {agent.id}) auto-defaulted: borrower died.")
        self.agents = survivors
        
        # --- Process Global Loan List ---
        active_loans_after_step = []
        for loan in list(self.all_loans): # Iterate copy as we might modify
            if loan.issue_tick == current_tick: # Newly issued this tick
                new_loan_volume_this_tick_counter += loan.principal
            
            if loan.is_defaulted and not any(a.id == loan.borrower_id for a in self.agents): # If borrower defaulted and is now dead
                pass

            if not loan.is_fully_repaid():
                # Prune very old defaulted loans (e.g., defaulted + duration + grace period passed)
                if loan.is_defaulted and current_tick > loan.due_tick + Config.LOAN_DURATION_TICKS: 
                    logging.debug(f"Pruning very old defaulted loan ID {loan.id}.")
                else:
                    active_loans_after_step.append(loan)
        self.all_loans = active_loans_after_step
        
        self.history['loans_active_count'][-1] = len(self.all_loans)
        self.history['total_loan_volume_tick'][-1] = new_loan_volume_this_tick_counter
        self.history['num_defaults_tick'][-1] = tick_defaults_counter


        # --- Update Projects ---
        for p in list(self.projects): # Iterate over a copy for safe removal
            p.time_left -= 1
            if p.time_left <= 0:
                if not self.grid.get_node_at(p.pos) and p.pos not in [Config.TRAIN_CENTER_POS, Config.TOOL_FACTORY_POS, Config.STORE_POS]:
                    new_node_capacity = random.uniform(80,120) * (1 + p.developer_id / (len(self.history['agent_wealth'])*2) ) 
                    new_node = ResourceNode(p.pos[0], p.pos[1], p.resource_type, new_node_capacity, p.developer_id) 
                    self.grid.add_node(new_node)
                    logging.info(f"Project by Agent {p.developer_id} completed at {p.pos}. New {p.resource_type} node (Cap:{new_node_capacity:.0f}).")
                else:
                    logging.warning(f"Project by {p.developer_id} at {p.pos} failed to complete, spot now occupied or invalid.")
                self.projects.remove(p)
        
        # --- Factory and Store Production ---
        for agent in self.agents: 
            if agent.inventory['wood'] > agent._archetype_food_target() + 10 : 
                wood_to_sell_to_infra = agent.inventory['wood'] - (agent._archetype_food_target() + 8) 
                wood_to_sell_to_infra = max(0, wood_to_sell_to_infra * 0.5) 

                if self._get_distance(agent.pos, self.factory.pos) < 3 and self.factory.wood_stock < 100 and wood_to_sell_to_infra > 1: 
                     transfer = min(wood_to_sell_to_infra, 10) 
                     agent.inventory['wood'] -= transfer; self.factory.wood_stock += transfer
                     agent.wealth += transfer * self.price_w * 0.7 
                     logging.debug(f"Agent {agent.id} sold {transfer:.1f} wood to factory. Factory stock: {self.factory.wood_stock:.0f}")
                elif self._get_distance(agent.pos, self.store.pos) < 3 and self.store.wood_stock < 50 and wood_to_sell_to_infra > 1: 
                     transfer = min(wood_to_sell_to_infra, 5) 
                     agent.inventory['wood'] -= transfer; self.store.wood_stock += transfer
                     agent.wealth += transfer * self.price_w * 0.7
                     logging.debug(f"Agent {agent.id} sold {transfer:.1f} wood to store. Store stock: {self.store.wood_stock:.0f}")

        self.factory.produce()
        self.store.produce()
        
        if not self.agents:
            logging.warning("All agents have died. Simulation ending.")
            return "game_over"
        return None


    def _get_distance(self, pos1: Tuple[int,int], pos2: Tuple[int,int]) -> int: 
        return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1])

    def run_pygame(self, steps: int, rl_policy=None, rl_agent_id_to_control=None, norm_mean=None, norm_var=None, norm_epsilon=1e-8):
        pygame.init()
        
        current_screen_width = Config.WINDOW_WIDTH
        current_screen_height = Config.WINDOW_HEIGHT + 120 
        
        screen = pygame.display.set_mode((current_screen_width, current_screen_height), pygame.RESIZABLE) 
        pygame.display.set_caption("Agent Economy Simulation (Interactive)")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 20) 
        small_font = pygame.font.SysFont(None, 16) 
        info_font = pygame.font.SysFont(None, 18)

        WHITE = (255, 255, 255); BLACK = (0, 0, 0); GREY = (200, 200, 200); LIGHT_GREY = (220, 220, 220)
        RED = (255, 0, 0); BLUE = (0, 0, 255); GREEN = (0, 255, 0); YELLOW = (255, 255, 0); MAGENTA = (255, 0, 255)

        role_colors = {
            'food_harvester': (34, 139, 34), 'wood_harvester': (139, 69, 19),
            'seller': (255, 165, 0), 'developer': (75, 0, 130),
            'idle': (128, 128, 128), 'idle_no_utilities': (128, 128, 128),
        }
        arch_markers = {'survivalist': 'S', 'risk_taker': 'R', 'trader': 'T', 'tool_artisan': 'A'}

        paused = False; selected_agent_id = None; running = True; sim_step_counter = 0
        info_start_y = Config.WINDOW_HEIGHT + 5 

        button_width = 80; button_height = 30; button_x = 200; button_y = info_start_y + 5 
        pause_button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
        button_color = (180, 180, 180); button_hover_color = (150, 150, 150); button_text_color = BLACK
        is_mouse_over_button = False

        while running and sim_step_counter < steps:
            is_mouse_over_button = False 
            mouse_pos_event = pygame.mouse.get_pos() 
            if pause_button_rect.collidepoint(mouse_pos_event):
                is_mouse_over_button = True

            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE: paused = not paused
                    elif event.key == pygame.K_ESCAPE: selected_agent_id = None
                elif event.type == pygame.VIDEORESIZE: 
                    current_screen_width = event.w; current_screen_height = event.h
                    screen = pygame.display.set_mode((current_screen_width, current_screen_height), pygame.RESIZABLE)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: 
                        mouse_pos_click = event.pos
                        clicked_on_agent = False
                        click_radius_sq = (Config.CELL_SIZE // 3 + 3)**2 
                        for agent_obj_click in self.agents:
                             ax, ay = agent_obj_click.pos
                             px_center = ax * Config.CELL_SIZE + Config.CELL_SIZE // 2
                             py_center = ay * Config.CELL_SIZE + Config.CELL_SIZE // 2
                             dist_sq = (mouse_pos_click[0] - px_center)**2 + (mouse_pos_click[1] - py_center)**2
                             if dist_sq <= click_radius_sq: 
                                 selected_agent_id = agent_obj_click.id; clicked_on_agent = True; break
                        if pause_button_rect.collidepoint(mouse_pos_click):
                            paused = not paused 

            if not paused:
                # --- RL Agent Action Prediction (if applicable) ---
                if rl_policy and rl_agent_id_to_control is not None:
                    rl_agent_obj = next((a for a in self.agents if a.id == rl_agent_id_to_control), None)
                    if rl_agent_obj and rl_agent_obj.is_rl_controlled: # Double check flag
                        raw_obs = _get_observation_for_pygame(rl_agent_obj, self)
                        
                        # Normalize observation
                        if norm_mean is not None and norm_var is not None:
                            # Ensure raw_obs is float for arithmetic operations
                            normalized_obs = (raw_obs.astype(np.float32) - norm_mean) / np.sqrt(norm_var + norm_epsilon)
                            # Clipping can be important if policy is sensitive to out-of-distribution values
                            # obs_low = np.array([-np.inf,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] + [0]*len(Config.ROLES) + [0]*len(Config.AGENT_ARCHETYPES), dtype=np.float32)
                            # obs_high = np.array([np.inf,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] + [1]*len(Config.ROLES) + [1]*len(Config.AGENT_ARCHETYPES), dtype=np.float32)
                            # For simplicity, we'll rely on the original normalization ranges or assume policy is robust.
                            # normalized_obs = np.clip(normalized_obs, obs_low, obs_high) # Complex due to wealth
                        else:
                            normalized_obs = raw_obs # No normalization if stats not available
                            logging.warning("RL agent acting without observation normalization.")

                        # Predict action using the loaded policy
                        action_int, _ = rl_policy.predict(normalized_obs.reshape(1, -1), deterministic=True)
                        action_int = action_int.item() # Get integer from array

                        # Map action to agent task
                        _map_action_to_agent_task_for_pygame(rl_agent_obj, self, action_int)
                        logging.debug(f"Pygame RL Agent {rl_agent_obj.id} took action: {ACTION_TO_TASK_MAP_PYGAME.get(action_int, 'unknown')}")
                # --- End RL Agent Action ---

                sim_status = self.step()
                if sim_status != "game_over" and self.agents:
                     sim_step_counter += 1
                elif sim_status == "game_over" or not self.agents:
                    logging.info(f"SIMULATION ENDED at tick {sim_step_counter}: All agents died or max steps reached.")
                    running = False 
            
            screen.fill(LIGHT_GREY) 
            grid_area_height = Config.WINDOW_HEIGHT 
            for x_coord in range(0, Config.WINDOW_WIDTH, Config.CELL_SIZE): pygame.draw.line(screen, GREY, (x_coord, 0), (x_coord, grid_area_height))
            for y_coord in range(0, grid_area_height, Config.CELL_SIZE): pygame.draw.line(screen, GREY, (0, y_coord), (Config.WINDOW_WIDTH, y_coord))

            tx, ty = self.train_center.pos; pygame.draw.rect(screen, YELLOW, (tx * Config.CELL_SIZE, ty * Config.CELL_SIZE, Config.CELL_SIZE, Config.CELL_SIZE)); screen.blit(small_font.render("T", True, BLACK), (tx * Config.CELL_SIZE + 5, ty * Config.CELL_SIZE + 5))
            fx, fy = self.factory.pos; pygame.draw.rect(screen, (100, 100, 100), (fx * Config.CELL_SIZE, fy * Config.CELL_SIZE, Config.CELL_SIZE, Config.CELL_SIZE)); screen.blit(small_font.render("F", True, WHITE), (fx * Config.CELL_SIZE + 5, fy * Config.CELL_SIZE + 5))
            sx, sy = self.store.pos; pygame.draw.rect(screen, (0, 200, 200), (sx * Config.CELL_SIZE, sy * Config.CELL_SIZE, Config.CELL_SIZE, Config.CELL_SIZE)); screen.blit(small_font.render("S", True, BLACK), (sx * Config.CELL_SIZE + 5, sy * Config.CELL_SIZE + 5))

            for node_obj in self.grid.nodes:
                cx, cy = node_obj.pos; col = (160, 82, 45) if node_obj.resource_type == 'wood' else (60, 179, 113)
                pygame.draw.rect(screen, col, (cx * Config.CELL_SIZE + 2, cy * Config.CELL_SIZE + 2, Config.CELL_SIZE - 4, Config.CELL_SIZE - 4))
                cap_text = small_font.render(f"{node_obj.capacity:.0f}", True, BLACK if sum(col) > 300 else WHITE)
                screen.blit(cap_text, (cx * Config.CELL_SIZE + 3, cy * Config.CELL_SIZE + Config.CELL_SIZE // 2 - 5))

            for proj_obj in self.projects:
                px, py = proj_obj.pos; pygame.draw.rect(screen, MAGENTA, (px * Config.CELL_SIZE, py * Config.CELL_SIZE, Config.CELL_SIZE, Config.CELL_SIZE), 3)
                time_text = small_font.render(f"P{proj_obj.time_left}", True, BLACK)
                screen.blit(time_text, (px * Config.CELL_SIZE + 2, py * Config.CELL_SIZE + 2))

            for agent_obj in self.agents:
                ax, ay = agent_obj.pos; px = ax * Config.CELL_SIZE + Config.CELL_SIZE // 2; py = ay * Config.CELL_SIZE + Config.CELL_SIZE // 2
                agent_role_color = role_colors.get(agent_obj.role, BLACK)
                if agent_obj.id == selected_agent_id: pygame.draw.circle(screen, YELLOW, (px, py), Config.CELL_SIZE // 3 + 5, 3) 
                if agent_obj.is_rl_controlled: # Highlight RL agent
                    pygame.draw.circle(screen, BLUE, (px,py), Config.CELL_SIZE // 3 + 7, 2) # Blue outline
                if agent_obj.emergency_food_seeking: pygame.draw.circle(screen, RED, (px, py), Config.CELL_SIZE // 3 + 4, 3)
                if agent_obj.default_cooldown > 0: pygame.draw.circle(screen, (100, 100, 255), (px, py), Config.CELL_SIZE // 3 + 6, 2)
                pygame.draw.circle(screen, agent_role_color, (px, py), Config.CELL_SIZE // 3) 
                agent_display_str = f"{agent_obj.id}{arch_markers.get(agent_obj.archetype, '?')}"
                agent_text_surface = small_font.render(agent_display_str, True, WHITE if sum(agent_role_color) < 384 else BLACK)
                screen.blit(agent_text_surface, (px - agent_text_surface.get_width() // 2, py - agent_text_surface.get_height() // 2))
            
            if paused:
                pause_text_str = "PAUSED (Space to Resume, Esc to Deselect)"
                text_width_pause, text_height_pause = font.size(pause_text_str)
                draw_text(screen, pause_text_str, (current_screen_width // 2 - text_width_pause // 2, info_start_y), font, RED)

            col1_x = 5; line_h = 20 
            draw_text(screen, f"Tick: {sim_step_counter}/{steps} Agents: {len(self.agents)}", (col1_x, info_start_y), font, BLACK)
            draw_text(screen, f"P_Wood: ${self.price_w:.2f} P_Food: ${self.price_f:.2f}", (col1_x, info_start_y + line_h), font, BLACK)
            draw_text(screen, f"Fac_WT:{self.factory.wood_tools} FT:{self.factory.food_tools} Wood:{self.factory.wood_stock:.0f}", (col1_x, info_start_y + 2*line_h), font, BLACK)
            draw_text(screen, f"Store_Ent:{self.store.ent_stock} Wood:{self.store.wood_stock:.0f}", (col1_x, info_start_y + 3*line_h), font, BLACK)
            
            num_active_loans = self.history['loans_active_count'][-1] if self.history['loans_active_count'] else 0
            vol_tick = self.history['total_loan_volume_tick'][-1] if self.history['total_loan_volume_tick'] else 0
            def_tick = self.history['num_defaults_tick'][-1] if self.history['num_defaults_tick'] else 0
            loan_stats_info = f"Loans Active: {num_active_loans} Vol(tick): ${vol_tick:.0f} Defaults(tick): {def_tick}"
            draw_text(screen, loan_stats_info, (col1_x, info_start_y + 4*line_h), font, BLACK)

            current_button_color = button_hover_color if is_mouse_over_button else button_color
            pygame.draw.rect(screen, current_button_color, pause_button_rect)
            button_text_str = "Resume" if paused else "Pause"
            text_surf_button = font.render(button_text_str, True, button_text_color)
            text_rect_button = text_surf_button.get_rect(center=pause_button_rect.center)
            screen.blit(text_surf_button, text_rect_button)

            if selected_agent_id is not None:
                agent_selected = next((a for a in self.agents if a.id == selected_agent_id), None)
                if agent_selected:
                    panel_x = Config.WINDOW_WIDTH // 2; panel_y = info_start_y
                    panel_width = current_screen_width - panel_x - 10; panel_height = current_screen_height - panel_y - 10 
                    line_y_panel = panel_y; line_height_panel = 18 
                    draw_text(screen, f"--- Agent {agent_selected.id} Details ---", (panel_x, line_y_panel), info_font, BLUE); line_y_panel += line_height_panel
                    draw_text(screen, f"Archetype: {agent_selected.archetype} {'(RL)' if agent_selected.is_rl_controlled else ''}", (panel_x, line_y_panel), info_font, BLACK); line_y_panel += line_height_panel
                    draw_text(screen, f"Role: {agent_selected.role}", (panel_x, line_y_panel), info_font, BLACK); line_y_panel += line_height_panel
                    draw_text(screen, f"Wealth: ${agent_selected.wealth:.2f}", (panel_x, line_y_panel), info_font, BLACK); line_y_panel += line_height_panel
                    draw_text(screen, f"Skill: {agent_selected.skill:.2f}", (panel_x, line_y_panel), info_font, BLACK); line_y_panel += line_height_panel
                    draw_text(screen, f"Happiness: {agent_selected.happiness:.2f}", (panel_x, line_y_panel), info_font, BLACK); line_y_panel += line_height_panel
                    line_y_panel += 5 
                    draw_text(screen, f"Inventory:", (panel_x, line_y_panel), info_font, BLUE); line_y_panel += line_height_panel
                    draw_text(screen, f"  Food: {agent_selected.inventory['food']:.1f}", (panel_x, line_y_panel), info_font, BLACK); line_y_panel += line_height_panel
                    draw_text(screen, f"  Wood: {agent_selected.inventory['wood']:.1f}", (panel_x, line_y_panel), info_font, BLACK); line_y_panel += line_height_panel
                    draw_text(screen, f"  Ent: {agent_selected.inventory['ent']:.1f}", (panel_x, line_y_panel), info_font, BLACK); line_y_panel += line_height_panel
                    draw_text(screen, f"  Tier2 Device: {agent_selected.inventory['has_tier2_device']}", (panel_x, line_y_panel), info_font, BLACK); line_y_panel += line_height_panel
                    line_y_panel += 5
                    draw_text(screen, f"Tools:", (panel_x, line_y_panel), info_font, BLUE); line_y_panel += line_height_panel
                    draw_text(screen, f"  Wood (Use): {agent_selected.tools['wood']}", (panel_x, line_y_panel), info_font, BLACK); line_y_panel += line_height_panel
                    draw_text(screen, f"  Food (Use): {agent_selected.tools['food']}", (panel_x, line_y_panel), info_font, BLACK); line_y_panel += line_height_panel
                    if agent_selected.inventory['tools_for_sale']['wood'] > 0 or agent_selected.inventory['tools_for_sale']['food'] > 0:
                         draw_text(screen, f"  Wood (Sale): {agent_selected.inventory['tools_for_sale']['wood']}", (panel_x, line_y_panel), info_font, BLACK); line_y_panel += line_height_panel
                         draw_text(screen, f"  Food (Sale): {agent_selected.inventory['tools_for_sale']['food']}", (panel_x, line_y_panel), info_font, BLACK); line_y_panel += line_height_panel
                    line_y_panel += 5
                    draw_text(screen, f"Decision (Last Action):", (panel_x, line_y_panel), info_font, BLUE); line_y_panel += line_height_panel
                    action_str_panel = agent_selected.current_action_type if agent_selected.current_action_type else "None"
                    # Utility for scripted agents is not directly stored for panel display
                    draw_text(screen, f"  Action Intent: {action_str_panel}", (panel_x, line_y_panel), info_font, BLACK); line_y_panel += line_height_panel
                    line_y_panel += 5
                    draw_text(screen, f"Loans:", (panel_x, line_y_panel), info_font, BLUE); line_y_panel += line_height_panel
                    def format_loan_str(loan): status = "DEF" if loan.is_defaulted else f"{loan.amount_repaid:.1f}/{loan.total_due:.1f}"; return f"ID:{loan.id} L:{loan.lender_id} B:{loan.borrower_id} P:{loan.principal:.1f} S:{status}"
                    if agent_selected.loans_taken:
                        for loan_taken in agent_selected.loans_taken[:2]: draw_text(screen, f"  Taken: {format_loan_str(loan_taken)}", (panel_x, line_y_panel), info_font, BLACK); line_y_panel += line_height_panel
                        if len(agent_selected.loans_taken) > 2: draw_text(screen, "    ...more", (panel_x, line_y_panel), info_font, BLACK); line_y_panel += line_height_panel
                    else: draw_text(screen, "  Taken: None", (panel_x, line_y_panel), info_font, BLACK); line_y_panel += line_height_panel
                    if agent_selected.loans_given:
                         for loan_given in agent_selected.loans_given[:2]: draw_text(screen, f"  Given: {format_loan_str(loan_given)}", (panel_x, line_y_panel), info_font, BLACK); line_y_panel += line_height_panel
                         if len(agent_selected.loans_given) > 2: draw_text(screen, "    ...more", (panel_x, line_y_panel), info_font, BLACK); line_y_panel += line_height_panel
                    else: draw_text(screen, "  Given: None", (panel_x, line_y_panel), info_font, BLACK); line_y_panel += line_height_panel
                else: selected_agent_id = None 
            
            pygame.display.flip()
            clock.tick(Config.FPS)

        pygame.quit()

    def plot_results(self):
        try:
            import matplotlib.pyplot as plt
            from collections import Counter
        except ImportError:
            logging.warning("Matplotlib not found. Skipping plot generation.")
            return

        logging.info("Plotting results - this may take a moment for many ticks/agents.")
        num_agents_initial = len(self.history['agent_archetypes'])
        num_ticks_completed = len(self.history['prices'])

        if num_ticks_completed == 0:
            logging.warning("No simulation steps completed. Cannot plot results.")
            return

        fig1, axs1 = plt.subplots(4, 2, figsize=(18, 22)) 
        fig1.suptitle("Simulation Time Series Data", fontsize=16)

        prices_w = [p['wood'] for p in self.history['prices']]
        prices_f = [p['food'] for p in self.history['prices']]
        axs1[0,0].plot(prices_w, label='Wood Price', color='saddlebrown')
        axs1[0,0].plot(prices_f, label='Food Price', color='green')
        axs1[0,0].set_title('Market Prices Over Time'); axs1[0,0].set_xlabel('Time (Ticks)'); axs1[0,0].set_ylabel('Price'); axs1[0,0].legend(); axs1[0,0].grid(True)

        total_wealth_over_time = []; total_food_over_time = []; active_agents_over_time = []
        for t_idx in range(num_ticks_completed):
            current_tick_wealth = sum(self.history['agent_wealth'][i][t_idx] for i in range(num_agents_initial) if self.history['agent_wealth'][i][t_idx] is not None)
            current_tick_food = sum(self.history['agent_food'][i][t_idx] for i in range(num_agents_initial) if self.history['agent_food'][i][t_idx] is not None)
            current_active_agents = sum(1 for i in range(num_agents_initial) if self.history['agent_roles'][i][t_idx] != "dead")
            total_wealth_over_time.append(current_tick_wealth)
            total_food_over_time.append(current_tick_food)
            active_agents_over_time.append(current_active_agents)

        axs1[0,1].plot(total_wealth_over_time, label='Total Agent Wealth', color='gold')
        axs1[0,1].set_title('Total Agent Wealth & Food Stock Over Time'); axs1[0,1].set_xlabel('Time (Ticks)'); axs1[0,1].set_ylabel('Total Wealth', color='gold'); axs1[0,1].legend(loc='upper left'); axs1[0,1].grid(True)
        ax_food_twin = axs1[0,1].twinx()
        ax_food_twin.plot(total_food_over_time, label='Total Agent Food Stock', color='lightgreen', linestyle='--')
        ax_food_twin.set_ylabel('Total Food Stock', color='lightgreen'); ax_food_twin.legend(loc='upper right')

        axs1[1,0].plot(active_agents_over_time, label='Number of Active Agents', color='blue')
        axs1[1,0].set_title('Active Agents Over Time'); axs1[1,0].set_xlabel('Time (Ticks)'); axs1[1,0].set_ylabel('Number of Agents'); axs1[1,0].legend(); axs1[1,0].grid(True)
        axs1[1,0].set_ylim(0, num_agents_initial + 1)

        role_counts_over_time = {role: [] for role in Config.ROLES}
        sample_ticks = range(0, num_ticks_completed, max(1, num_ticks_completed // 20)) 
        for t_idx in sample_ticks:
            current_roles = [self.history['agent_roles'][i][t_idx] for i in range(num_agents_initial) if self.history['agent_roles'][i][t_idx] != "dead"]
            role_counts_at_tick = Counter(current_roles)
            for role_key in Config.ROLES: role_counts_over_time[role_key].append(role_counts_at_tick.get(role_key, 0))
        for role_key in Config.ROLES: axs1[1,1].plot(sample_ticks, role_counts_over_time[role_key], label=f'{role_key}s')
        axs1[1,1].set_title('Agent Role Distribution (Sampled)'); axs1[1,1].set_xlabel('Time (Ticks)'); axs1[1,1].set_ylabel('Number of Agents'); axs1[1,1].legend(); axs1[1,1].grid(True)

        axs1[2,0].plot(self.history['loans_active_count'], label='Active Loans Count', color='purple')
        axs1[2,0].set_title('Active Loans Over Time'); axs1[2,0].set_xlabel('Time (Ticks)'); axs1[2,0].set_ylabel('Count'); axs1[2,0].legend(); axs1[2,0].grid(True)
        ax_loan_vol_twin = axs1[2,0].twinx()
        ax_loan_vol_twin.plot(self.history['total_loan_volume_tick'], label='New Loan Volume ($)', color='magenta', linestyle=':')
        ax_loan_vol_twin.set_ylabel('New Loan Volume ($)', color='magenta'); ax_loan_vol_twin.legend(loc='upper right')

        axs1[2,1].plot(self.history['num_defaults_tick'], label='Defaults per Tick', color='red')
        axs1[2,1].set_title('Loan Defaults per Tick'); axs1[2,1].set_xlabel('Time (Ticks)'); axs1[2,1].set_ylabel('Defaults Count'); axs1[2,1].legend(); axs1[2,1].grid(True)

        final_agents_data_list = [] 
        surviving_agent_ids = [agent.id for agent in self.agents] 
        for agent_id in surviving_agent_ids:
            agent_current = next((a for a in self.agents if a.id == agent_id), None)
            if agent_current:
                 final_agents_data_list.append({
                     'id': agent_current.id, 'archetype': agent_current.archetype, 'wealth': agent_current.wealth,
                     'food': agent_current.inventory['food'], 'wood': agent_current.inventory['wood'],
                     'wood_tools': agent_current.tools['wood'], 'food_tools': agent_current.tools['food'], 'role': agent_current.role
                 })

        if final_agents_data_list:
            wealth_by_archetype = {arch: [] for arch in Config.AGENT_ARCHETYPES}
            for agent_data in final_agents_data_list: wealth_by_archetype[agent_data['archetype']].append(agent_data['wealth'])
            arch_labels = list(wealth_by_archetype.keys())
            arch_wealth_data_for_plot = [wealth_by_archetype[arch] for arch in arch_labels if wealth_by_archetype[arch]]
            arch_labels_for_plot = [arch for arch in arch_labels if wealth_by_archetype[arch]]
            if arch_wealth_data_for_plot: axs1[3,0].boxplot(arch_wealth_data_for_plot, labels=arch_labels_for_plot, showmeans=True)
            axs1[3,0].set_title('Final Wealth Distribution by Archetype (Survivors)')
            axs1[3,0].set_ylabel('Wealth')

            survivor_counts_by_arch = Counter(agent_data['archetype'] for agent_data in final_agents_data_list)
            axs1[3,1].bar(survivor_counts_by_arch.keys(), survivor_counts_by_arch.values(), color=['skyblue', 'salmon', 'lightgreen'])
            axs1[3,1].set_title('Number of Survivors by Archetype')
            axs1[3,1].set_ylabel('Count')
        else:
            axs1[3,0].text(0.5, 0.5, "No agents survived.", horizontalalignment='center', verticalalignment='center', transform=axs1[3,0].transAxes)
            axs1[3,1].text(0.5, 0.5, "No agents survived.", horizontalalignment='center', verticalalignment='center', transform=axs1[3,1].transAxes)
        fig1.tight_layout(pad=2.0, rect=[0, 0, 1, 0.96]) 

        if final_agents_data_list:
            fig2, axs2 = plt.subplots(2, 3, figsize=(18, 10)) 
            fig2.suptitle("Final Agent Inventory and Stats (Survivors)", fontsize=16)
            agent_ids_str = [str(a['id']) for a in final_agents_data_list]
            wealth_values = [a['wealth'] for a in final_agents_data_list]
            axs2[0,0].bar(agent_ids_str, wealth_values, color='gold'); axs2[0,0].set_title('Wealth'); axs2[0,0].set_ylabel('Amount'); axs2[0,0].tick_params(axis='x', rotation=45)
            food_values = [a['food'] for a in final_agents_data_list]
            axs2[0,1].bar(agent_ids_str, food_values, color='limegreen'); axs2[0,1].set_title('Food'); axs2[0,1].tick_params(axis='x', rotation=45)
            wood_values = [a['wood'] for a in final_agents_data_list]
            axs2[0,2].bar(agent_ids_str, wood_values, color='saddlebrown'); axs2[0,2].set_title('Wood'); axs2[0,2].tick_params(axis='x', rotation=45)
            wood_tools_values = [a['wood_tools'] for a in final_agents_data_list]
            axs2[1,0].bar(agent_ids_str, wood_tools_values, color='dimgray'); axs2[1,0].set_title('Wood Tools'); axs2[1,0].set_ylabel('Count'); axs2[1,0].tick_params(axis='x', rotation=45)
            food_tools_values = [a['food_tools'] for a in final_agents_data_list]
            axs2[1,1].bar(agent_ids_str, food_tools_values, color='darkseagreen'); axs2[1,1].set_title('Food Tools'); axs2[1,1].tick_params(axis='x', rotation=45)
            if axs2.shape == (2,3): axs2[1,2].axis('off') 
            fig2.tight_layout(pad=2.0, rect=[0, 0, 1, 0.95])
        else: logging.info("No surviving agents to plot detailed final stats.")
        plt.show() 

def draw_text(surface, text, pos, font, color=(0,0,0)):
    text_surface = font.render(text, True, color)
    surface.blit(text_surface, pos)

# --- Helper function to create the environment (moved to top level)---
def _make_env_for_training(): # RENAMED local function
    env_args = {
        'num_agents': TOTAL_AGENTS, 'num_resources': NUM_RESOURCES_NODES,
        'rl_agent_id': RL_AGENT_ID, 'max_steps': MAX_STEPS_PER_EPISODE,
        'render_mode': RENDER_MODE
    }
    return gym.make('AgentEconomy-v0', **env_args)

if __name__=='__main__':
    from rl_env import AgentEconomyEnv 
    setup_logging()
    logging.info("Setting up RL Environment and Training...")

    # --- Register the custom environment ---
    try:
        gym.envs.registration.spec('AgentEconomy-v0')
        logging.info("Environment 'AgentEconomy-v0' already registered. Will use existing registration.")
    except gym.error.NameNotFound:
        register(
            id='AgentEconomy-v0',
            entry_point='rl_env:AgentEconomyEnv', 
            kwargs={
                'num_agents': TOTAL_AGENTS,
                'num_resources': NUM_RESOURCES_NODES,
                'rl_agent_id': RL_AGENT_ID,
                'max_steps': MAX_STEPS_PER_EPISODE,
                'render_mode': RENDER_MODE
            }
        )
        logging.info("Registered custom environment: AgentEconomy-v0")

    # --- Create and Check the Environment ---
    logging.info(f"Creating Gym environment 'AgentEconomy-v0' with RL Agent ID: {RL_AGENT_ID}")
    env = DummyVecEnv([_make_env_for_training]) # UPDATED call to use renamed local function
    logging.info("Environment wrapped in DummyVecEnv.")

    # --- Wrap with VecNormalize ---
    env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.99)
    logging.info("VecEnv wrapped in VecNormalize.")

    # --- Define and Train the RL Model (PPO) ---
    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log="./ppo_agent_economy_tensorboard/")
    
    logging.info("Starting PPO model training...")
    TOTAL_TRAINING_TIMESTEPS = 100000 
    model.learn(total_timesteps=TOTAL_TRAINING_TIMESTEPS, progress_bar=False)
    logging.info("Training complete.")

    # --- Save VecNormalize stats ---
    STATS_PATH = "vecnormalize_stats.pkl"
    env.save(STATS_PATH)
    logging.info(f"VecNormalize stats saved to {STATS_PATH}")

    # --- Save Only the Policy Network ---
    POLICY_SAVE_PATH = "ppo_agent_economy_policy.pth"
    model.policy.save(POLICY_SAVE_PATH)
    logging.info(f"Trained policy saved to {POLICY_SAVE_PATH}")
    
    env.close()
    logging.info("RL training script finished.")

    # --- RUN SIMULATION WITH TRAINED RL AGENT --- 
    print("\n--- Running Simulation with Loaded RL Agent --- ")
    logging.info("Setting up simulation with trained RL agent...")

    SIM_STEPS = 500
    sim_rl = Economy(num_agents=TOTAL_AGENTS, num_resources=NUM_RESOURCES_NODES, seed=Config.SEED, rl_agent_id=RL_AGENT_ID)

    norm_mean, norm_var = None, None
    norm_epsilon = 1e-8 
    try:
        with open(STATS_PATH, 'rb') as f:
            loaded_vec_normalize_obj = pickle.load(f) # Renamed variable
            norm_mean = loaded_vec_normalize_obj.obs_rms.mean
            norm_var = loaded_vec_normalize_obj.obs_rms.var
            norm_epsilon = loaded_vec_normalize_obj.epsilon 
            logging.info(f"Loaded VecNormalize stats from {STATS_PATH}")
    except FileNotFoundError: logging.error(f"Could not load normalization statistics from {STATS_PATH}."); sys.exit(1)
    except Exception as e: logging.error(f"Error loading normalization statistics: {e}"); sys.exit(1)

    try:
        # Use the globally defined make_env for policy loading structure
        dummy_env_for_policy_load = make_env(TOTAL_AGENTS, NUM_RESOURCES_NODES, RL_AGENT_ID, MAX_STEPS_PER_EPISODE, RENDER_MODE) # UNCHANGED - calls global make_env
        policy_model_struct = PPO("MlpPolicy", dummy_env_for_policy_load) 
        loaded_policy_obj = policy_model_struct.policy.load(POLICY_SAVE_PATH) # Renamed variable
        logging.info(f"Loaded trained policy from {POLICY_SAVE_PATH}")
        dummy_env_for_policy_load.close() 
    except FileNotFoundError: logging.error(f"Could not load trained policy from {POLICY_SAVE_PATH}."); sys.exit(1)
    except Exception as e: logging.error(f"Error loading policy: {e}"); sys.exit(1)

    sim_rl.run_pygame(steps=SIM_STEPS, 
                        rl_policy=loaded_policy_obj, 
                        rl_agent_id_to_control=RL_AGENT_ID, 
                        norm_mean=norm_mean, 
                        norm_var=norm_var, 
                        norm_epsilon=norm_epsilon)
    # sim_rl.plot_results()


    # If you still want to run the old simulation for comparison or visualization:
    # print("\nRunning original Pygame simulation (non-RL)...")