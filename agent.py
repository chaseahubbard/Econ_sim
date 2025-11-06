import random
import logging
import sys
from typing import List, Optional, Tuple, Dict, Any, Union
import numpy as np

# Assuming Config, Loan, Grid, ResourceNode, TrainingCenter, ToolFactory, Store, Project classes are defined elsewhere or in the same file.
from config import Config
from core_components import Grid, ResourceNode, TrainingCenter, ToolFactory, Store, Project, Loan # Import needed components

class Agent:
    def __init__(self, aid:int, grid:'Grid', train_center:'TrainingCenter', factory:'ToolFactory', store:'Store', projects:List['Project'], agents_ref:List['Agent']):
        self.id = aid
        self.grid = grid
        self.agents_ref = agents_ref # Reference to the list of all agents
        self.pos = (random.randrange(grid.width), random.randrange(grid.height))
        self.wealth = random.uniform(30,80) # Slightly higher start wealth
        self.skill = random.uniform(0.5,1.5) # General skill
        self.inventory = {'wood':0.0,'food':random.uniform(2,5),'ent':0.0, 'has_tier2_device': False, 'tools_for_sale': {'wood': 0, 'food': 0}} # Start with some food, added tools_for_sale
        self.food_hunger = 0 # Increments each step, resets when food is consumed
        self.tools = {'wood':0,'food':0} # Number of tools
        self.happiness = 1.0
        self.role = random.choice(Config.ROLES)
        self.archetype = random.choice(Config.AGENT_ARCHETYPES)
        self.role_eval_timer = random.randint(0, Config.ROLE_CHANGE_INTERVAL) # Stagger initial evals

        self.target:Optional[Tuple[int,int]] = None
        self.trade_partner:Optional['Agent'] = None # Also used for loan interactions
        self.current_action_type: Optional[str] = None
        self.trade_item_type: Optional[str] = None
        self.trade_is_buy: bool = False
        self.emergency_food_seeking: bool = False

        self.loans_taken: List['Loan'] = []
        self.loans_given: List['Loan'] = []
        self.loan_request_attempts = 0
        self.default_cooldown = 0 # Ticks remaining before being considered for loans after default
        self.current_loan_purpose: Optional[str] = None
        self.current_loan_amount_needed: float = 0.0

        self.train_center = train_center
        self.factory = factory
        self.store = store
        self.projects = projects # Global list of projects

        self.is_rl_controlled = False # NEW FLAG
        self.last_action_outcome = None # For Gym env info
        self.last_starving_status = False # For Gym env info

    def _get_distance(self, pos1: Tuple[int,int], pos2: Tuple[int,int]) -> int:
        return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1])

    def _archetype_food_target(self) -> float:
        base_target = Config.FOOD_TARGET_LEVEL
        if self.archetype == 'survivalist':
            return base_target * Config.SURVIVALIST_FOOD_TARGET_MULTIPLIER
        return base_target

    def _handle_loan_repayments_and_accruals(self, current_tick: int, price_f: float):
        if self.default_cooldown > 0:
            self.default_cooldown -=1

        for loan in list(self.loans_taken): # Iterate copy for safe removal
            loan.accrue_interest() # Accrue interest each tick

            if loan.is_fully_repaid():
                self.loans_taken.remove(loan)
                lender = next((a for a in self.agents_ref if a.id == loan.lender_id), None)
                if lender:
                    lender.loans_given = [l for l in lender.loans_given if l.id != loan.id]
                logging.debug(f"Agent {self.id} fully repaid Loan ID {loan.id} to Agent {loan.lender_id}.")
                continue

            if current_tick >= loan.due_tick and not loan.is_defaulted and not loan.is_fully_repaid():
                loan.is_defaulted = True
                self.default_cooldown = Config.LOAN_DEFAULT_PENALTY_TICKS
                logging.warning(f"Agent {self.id} DEFAULTED on Loan ID {loan.id} from Agent {loan.lender_id}. Owed: {loan.get_remaining_owed():.2f}")

            # Attempt repayment
            # Only pay if has surplus beyond food needs for X ticks + a small buffer
            food_needed_for_safety_value = self._archetype_food_target() * price_f * 1.2
            buffer_wealth = 5.0 # Small absolute buffer in addition to food safety

            if self.wealth > food_needed_for_safety_value + buffer_wealth:
                available_for_repayment = (self.wealth - (food_needed_for_safety_value + buffer_wealth)) * Config.LOAN_REPAYMENT_PRIORITY
                available_for_repayment = max(0, available_for_repayment)

                if available_for_repayment > Config.MIN_PRICE:
                    payment_amount = loan.make_payment(available_for_repayment)
                    if payment_amount > 0:
                        self.wealth -= payment_amount
                        lender = next((a for a in self.agents_ref if a.id == loan.lender_id), None)
                        if lender: # Lender might not exist if they died
                            lender.wealth += payment_amount
                        logging.debug(f"Agent {self.id} paid ${payment_amount:.2f} on Loan ID {loan.id} to Agent {loan.lender_id}. Owed: {loan.get_remaining_owed():.2f}")
                        if loan.is_fully_repaid(): # Double check after payment
                             self.loans_taken.remove(loan)
                             if lender: lender.loans_given = [l for l in lender.loans_given if l.id != loan.id]
                             logging.debug(f"Agent {self.id} fully repaid Loan ID {loan.id} (post-payment check).")

        # Lenders check their loans given (e.g. if a loan got repaid or defaulted, they might update their records)
        # This is mostly handled by borrower's actions informing the lender or global loan list.
        for loan in list(self.loans_given): # Iterate copy if removal is needed
            if loan.is_fully_repaid() or (loan.is_defaulted and current_tick > loan.due_tick + Config.LOAN_DURATION_TICKS): # Prune very old defaulted loans from lender's perspective too
                borrower = next((a for a in self.agents_ref if a.id == loan.borrower_id), None)
                # If borrower is dead and loan defaulted, lender might remove it earlier.
                if loan.is_defaulted and not borrower : # Borrower died and defaulted
                    self.loans_given.remove(loan)
                    logging.debug(f"Lender {self.id} removed defaulted loan {loan.id} (borrower {loan.borrower_id} died).")


    def evaluate_and_change_role(self, price_w: float, price_f: float):
        self.role_eval_timer -= 1
        if self.role_eval_timer > 0 and not self.emergency_food_seeking: # Allow emergency to trigger role eval
            return

        self.role_eval_timer = Config.ROLE_CHANGE_INTERVAL # Reset timer
        current_utility = -float('inf')
        best_role = self.role
        best_role_utility_val = -float('inf')
        utilities_debug = {}

        base_food_consumption_rate = 1.0 / Config.HUNGER_INTERVAL
        cost_of_living = base_food_consumption_rate * price_f

        for r_option in Config.ROLES:
            utility = 0
            # Base utilities for roles
            if r_option == 'food_harvester':
                utility = (self.skill + self.tools['food'] * Config.TOOL_EFFECT) * price_f
            elif r_option == 'wood_harvester':
                utility = (self.skill + self.tools['wood'] * Config.TOOL_EFFECT) * price_w
                if self.inventory['has_tier2_device']: utility *= Config.TIER2_WOOD_DEVICE_EFFECT_MULTIPLIER # More attractive if has T2
            elif r_option == 'seller': # Trader archetype strongly prefers this
                utility = self.skill * (price_w + price_f) * 0.05 + Config.SELLER_SALARY # Lower base, profit from trades
                if self.archetype == 'trader': utility *= 2.0
            elif r_option == 'developer':
                future_node_val_per_step = 0.05 * (price_w + price_f) / 2 # Avg price for generic node
                utility = (future_node_val_per_step - (Config.PROJECT_WOOD_COST * price_w / Config.PROJECT_TIME))
                if self.inventory['wood'] >= Config.PROJECT_WOOD_COST: utility *= 1.2
                if self.archetype == 'risk_taker': utility *= 1.5
                if self.archetype == 'survivalist': utility *= 0.3
            elif self.archetype == 'tool_artisan':
                if r_option == 'wood_harvester': utility *= 3.0 # Strongly prefers harvesting wood
                elif r_option == 'seller': utility *= 1.5 # Also considers selling, maybe tools
                elif r_option == 'food_harvester': utility *= 0.1 # Avoids food harvesting unless starving
                else: utility *= 0.5 # Generally less interested in other roles


            # Archetype and needs-based adjustments
            if self.archetype == 'survivalist':
                if r_option != 'food_harvester': utility *= 0.7
            elif self.archetype == 'risk_taker':
                if r_option == 'food_harvester': utility *= 0.8 # Prefers higher income potentially
            elif self.archetype == 'tool_artisan':
                if r_option == 'wood_harvester': utility *= 3.0 # Strongly prefers harvesting wood
                elif r_option == 'seller': utility *= 1.5 # Also considers selling, maybe tools
                elif r_option == 'food_harvester': utility *= 0.1 # Avoids food harvesting unless starving
                else: utility *= 0.5 # Generally less interested in other roles

            if self.inventory['food'] < self._archetype_food_target():
                if r_option == 'food_harvester': utility *= 1.5
                else: utility *= 0.8 # Penalize non-food roles if food is needed
            if self.inventory['food'] < 1 or self.emergency_food_seeking:
                if r_option == 'food_harvester': utility *= 3.0 # Strong push for food harvesting if critical
                else: utility *= 0.1 # Severely penalize other roles

            # Tool Artisan specific override when starving
            if self.archetype == 'tool_artisan' and (self.inventory['food'] < 1 or self.emergency_food_seeking):
                if r_option == 'food_harvester': utility = float('inf') # Absolute priority if starving
                elif r_option == 'wood_harvester': utility *= 0.01 # Massively penalize wood if starving

            utility -= cost_of_living

            utilities_debug[r_option] = utility
            if r_option == self.role:
                current_utility = utility

            if utility > best_role_utility_val:
                best_role_utility_val = utility
                best_role = r_option

        is_starving_and_can_switch_to_food_harvester = (
            (self.inventory['food'] < 1 or self.emergency_food_seeking) and \
            best_role == 'food_harvester' and \
            self.role != 'food_harvester'
        )

        should_change_role = (best_role != self.role and \
                             best_role_utility_val > current_utility * (1 + Config.ROLE_CHANGE_INERTIA)) \
                             or is_starving_and_can_switch_to_food_harvester

        if should_change_role:
            log_reason = f"(NewU: {best_role_utility_val:.2f} > CurrU: {current_utility:.2f}*Inertia)"
            if is_starving_and_can_switch_to_food_harvester:
                log_reason = "(Starving, FORCING switch to Food Harvester)"
            logging.debug(f"Agent {self.id} ({self.archetype[0]}) changing role from {self.role} to {best_role} {log_reason}")
            self.role = best_role
            self.current_action_type = None
            self.target = None

# Assuming other methods like _get_distance, _archetype_food_target, evaluate_and_change_role are defined
# Also assumes Config class and necessary imports are available

    def decide_target(self, price_w: float, price_f: float, current_tick: int):
        food_target_level = self._archetype_food_target()

        # --- EMERGENCY FOOD PROTOCOL ---
        # Trigger emergency if food is very low OR if food is below target and hunger is high
        if self.inventory['food'] < 1 or \
           (self.inventory['food'] < food_target_level and self.food_hunger >= Config.HUNGER_INTERVAL - 1):

            if not self.emergency_food_seeking: # Log only when entering emergency mode
                logging.debug(f"Agent {self.id} ({self.archetype[0]},{self.role}) ENTERING EMERGENCY food seeking. Food: {self.inventory['food']:.1f}, Hunger: {self.food_hunger}")
            self.emergency_food_seeking = True

            # Reset current action if it's not helping the emergency
            if self.current_action_type and not (\
                self.current_action_type.startswith("harvest_food") or \
                self.current_action_type.startswith("trade_buy_food") or \
                self.current_action_type.startswith("seek_loan_for_food")):
                self.current_action_type = None
                self.target = None

            # Try actions only if no current action is set for the emergency
            if self.current_action_type is None:
                # 1. Try to harvest food directly
                food_nodes = [n for n in self.grid.nodes if n.resource_type == 'food' and n.capacity > 0]
                if food_nodes:
                    food_nodes.sort(key=lambda n: self._get_distance(self.pos, n.pos))
                    self.target = food_nodes[0].pos
                    self.current_action_type = "harvest_food_emergency"
                    logging.debug(f"Agent {self.id} EMERGENCY: Targeting food node {self.target} for harvesting.")
                    return # Decided emergency action

                # 2. Try to buy food
                partners_with_food = [a for a in self.agents_ref if a is not self and a.inventory.get('food', 0) > 0]
                if partners_with_food and self.wealth >= price_f * 0.5 : # Need some money
                    partners_with_food.sort(key=lambda p: (not (p.role == 'seller' or p.archetype == 'trader'), self._get_distance(self.pos, p.pos)))
                    self.trade_partner = partners_with_food[0]
                    self.target = self.trade_partner.pos
                    self.current_action_type = "trade_buy_food_emergency"
                    self.trade_item_type = 'food'; self.trade_is_buy = True
                    logging.debug(f"Agent {self.id} EMERGENCY: Targeting agent {self.trade_partner.id} to buy food.")
                    return # Decided emergency action

                # 3. Try to seek loan for food if low on wealth and no default cooldown
                if self.wealth < price_f * 2 and self.default_cooldown == 0 and self.loan_request_attempts < Config.MAX_LOAN_REQUEST_ATTEMPTS:
                    logging.debug(f"Agent {self.id} EMERGENCY: Low wealth ({self.wealth:.1f}), trying to seek loan for food.")
                    # Set intent, target will be determined in main utility block if needed
                    self.current_action_type = "seek_loan_for_food"
                    self.current_loan_amount_needed = price_f * (food_target_level + 1)
                    self.current_loan_purpose = "emergency_food"
                    # Don't return yet, let main utility calculation proceed to find lender target

                # 4. If role is not food_harvester, force role evaluation
                if self.current_action_type is None and self.role != 'food_harvester':
                    logging.debug(f"Agent {self.id} EMERGENCY: No immediate food source. Current role {self.role}. Triggering role re-evaluation.")
                    self.role_eval_timer = 0 # Force evaluation
                    self.evaluate_and_change_role(price_w, price_f)
                    self.current_action_type = None # Let next main decision cycle handle target after role change
                    return # Exit decide_target for this tick after forcing role eval

                # 5. Last Resort (if still no action type set after all above checks)
                if self.current_action_type is None:
                    logging.warning(f"Agent {self.id} EMERGENCY: No food source found! Role: {self.role}. Wandering or trying to train.")
                    if self.role == 'food_harvester' and self.wealth >= Config.COST_PER_SKILL:
                        self.target = self.train_center.pos
                        self.current_action_type = "train_emergency_no_food_source"
                    else: # Wander
                        self.target = (random.randrange(self.grid.width), random.randrange(self.grid.height))
                        self.current_action_type = "idle_starving_critical_no_options"
                    return # Decided last resort action

        else: # Not in emergency or emergency resolved this tick
            if self.emergency_food_seeking: # Was in emergency, but conditions no longer met
                logging.debug(f"Agent {self.id} exiting emergency food mode.")
            self.emergency_food_seeking = False

        # If an emergency action was set above and returned, skip normal evaluation
        # If seek_loan_for_food was set, continue to calculate utilities to find target lender

        # If already decided on a non-emergency loan seeking action, keep it unless emergency overrides
        if self.current_action_type and self.current_action_type.startswith("seek_loan_") and self.target and not self.emergency_food_seeking:
             return # Stick with current loan seeking target

        # --- Non-Emergency Decision Making ---
        self.evaluate_and_change_role(price_w, price_f) # Normal role evaluation timing

        utilities = {}
        # Reset trade partner unless seeking loan (where it's the potential lender)
        if not (self.current_action_type and self.current_action_type.startswith("seek_loan_")):
            self.trade_partner = None
        self.trade_item_type = None # Always reset item type

        # --- Calculate Utilities ---
        harvest_efficiency_food = self.skill + self.tools['food'] * Config.TOOL_EFFECT
        harvest_efficiency_wood = self.skill + self.tools['wood'] * Config.TOOL_EFFECT
        if self.inventory['has_tier2_device']: harvest_efficiency_wood *= Config.TIER2_WOOD_DEVICE_EFFECT_MULTIPLIER

        # ** Role-Specific Primary Goals **
        if self.role == 'food_harvester':
            utilities['harvest_food'] = harvest_efficiency_food * price_f
            if self.inventory['food'] < food_target_level: utilities['harvest_food'] *= 1.5
            sell_food_surplus_threshold = food_target_level + (2 if self.archetype != 'survivalist' else 5)
            if self.inventory['food'] > sell_food_surplus_threshold:
                utilities['trade_sell_food'] = price_f * (self.inventory['food'] - sell_food_surplus_threshold) * (0.5 if self.archetype=='survivalist' else 1.0)

        elif self.role == 'wood_harvester':
            utilities['harvest_wood'] = harvest_efficiency_wood * price_w
            if self.inventory['wood'] < Config.WOOD_TARGET_LEVEL_WOOD_HARVESTER : utilities['harvest_wood'] *= 1.2
            if self.inventory['wood'] > Config.WOOD_TARGET_LEVEL_WOOD_HARVESTER + 3:
                utilities['trade_sell_wood'] = price_w * (self.inventory['wood'] - Config.WOOD_TARGET_LEVEL_WOOD_HARVESTER)

        elif self.role == 'seller':
            self.wealth += Config.SELLER_SALARY
            if self.inventory['food'] < Config.FOOD_TARGET_LEVEL_SELLER and self.wealth >= price_f:
                utilities['trade_buy_food_for_resale'] = (Config.FOOD_TARGET_LEVEL_SELLER - self.inventory['food']) * price_f * (0.9 if self.archetype == 'trader' else 0.7)
            if self.inventory['wood'] < Config.WOOD_TARGET_LEVEL_SELLER and self.wealth >= price_w:
                utilities['trade_buy_wood_for_resale'] = (Config.WOOD_TARGET_LEVEL_SELLER - self.inventory['wood']) * price_w * (0.9 if self.archetype == 'trader' else 0.7)
            if self.inventory['food'] > 0:
                utilities['trade_sell_food'] = self.inventory['food'] * price_f * (1.1 if self.archetype == 'trader' else 1.0)
            if self.inventory['wood'] > 0:
                utilities['trade_sell_wood'] = self.inventory['wood'] * price_w * (1.1 if self.archetype == 'trader' else 1.0)

        elif self.role == 'developer':
            if self.inventory['wood'] < Config.PROJECT_WOOD_COST:
                utilities['harvest_wood_for_project'] = harvest_efficiency_wood * price_w * Config.DEVELOPER_WOOD_PRIORITY_MULTIPLIER
                if self.wealth >= price_w * (Config.PROJECT_WOOD_COST - self.inventory['wood']):
                     utilities['trade_buy_wood_for_project'] = price_w * (Config.PROJECT_WOOD_COST - self.inventory['wood']) * Config.DEVELOPER_WOOD_PRIORITY_MULTIPLIER
            else:
                utilities['start_project'] = 100.0

        # --- Tool Artisan Specific Goals (Overrides general role goals if artisan) ---
        if self.archetype == 'tool_artisan' and not self.emergency_food_seeking:
            utilities = {} # Clear previous role-based utilities, focus on artisan tasks
            wood_needed_for_delivery = max(Config.WOOD_PER_TOOL * 2, 10.0) # Target having wood for a few tools
            # can_deliver_wood = self.inventory['wood'] >= Config.WOOD_PER_TOOL # Old condition

            # Only consider delivering wood if we have a decent amount (enough for >1 tool)
            if self.inventory['wood'] >= wood_needed_for_delivery:
                # High utility for delivering wood, increases with amount
                # Base utility higher now since we wait longer to deliver
                utilities['deliver_wood_to_factory'] = 75.0 + (self.inventory['wood'] - wood_needed_for_delivery) * price_w * 0.1

            # Always consider harvesting if below the target threshold
            if self.inventory['wood'] < wood_needed_for_delivery:
                 # High utility for harvesting wood if below target
                 utilities['harvest_wood_for_factory'] = harvest_efficiency_wood * price_w * 2.5
            else:
                 # Lower utility for harvesting if already has enough, but still positive
                 utilities['harvest_wood_for_factory'] = harvest_efficiency_wood * price_w * 0.5

            # Seller role is backup if wood harvesting isn't feasible or has excess non-wood items
            if self.role == 'seller':
                if self.inventory['food'] > self._archetype_food_target():
                    utilities['trade_sell_food_artisan'] = self.inventory['food'] * price_f * 0.8 # Less profit focus than trader
                # Avoid buying things for resale unless very rich
                if self.wealth > 100:
                   if self.inventory['food'] < Config.FOOD_TARGET_LEVEL_SELLER:
                       utilities['trade_buy_food_artisan_stock'] = (Config.FOOD_TARGET_LEVEL_SELLER - self.inventory['food']) * price_f * 0.2


        # ** Seller specific tool trading goals **
        if self.role == 'seller' and not self.emergency_food_seeking:
            has_tool_to_sell = self.inventory['tools_for_sale']['wood'] > 0 or self.inventory['tools_for_sale']['food'] > 0

            # Utility to fetch a tool if not currently holding one for sale
            if not has_tool_to_sell and self.wealth >= self.factory.price and self.factory.price > 0:
                if self.factory.wood_tools > 0:
                    # Higher utility if factory is well-stocked
                    utility_fetch_wood_tool = 25.0 + (self.factory.wood_tools * 0.5)
                    if self.archetype == 'trader': utility_fetch_wood_tool *= 1.5
                    utilities['go_get_tool_from_factory_wood'] = utility_fetch_wood_tool
                if self.factory.food_tools > 0:
                    utility_fetch_food_tool = 25.0 + (self.factory.food_tools * 0.5)
                    if self.archetype == 'trader': utility_fetch_food_tool *= 1.5
                    utilities['go_get_tool_from_factory_food'] = utility_fetch_food_tool

            # Utility to find a buyer if holding a tool for sale
            if has_tool_to_sell:
                # Which tool are we selling? Prioritize wood if holding both?
                tool_type = 'wood' if self.inventory['tools_for_sale']['wood'] > 0 else 'food'

                # Estimate potential profit (e.g., 20% markup on factory price)
                potential_profit = (self.factory.price * 1.2) - self.factory.price
                utility_find_buyer = 30.0 + potential_profit * 2.0 # Utility based on profit potential
                if self.archetype == 'trader': utility_find_buyer *= 1.8
                utilities[f'find_buyer_for_tool_{tool_type}'] = utility_find_buyer


        # ** Archetype Driven Goals / General Actions **
        # Build Tier 2 Device
        if not self.inventory['has_tier2_device'] and self.skill >= Config.TIER2_MIN_SKILL_TO_BUILD:
            base_utility_t2 = 20.0
            if self.archetype == 'risk_taker': base_utility_t2 *= 2.5
            elif self.archetype == 'survivalist': base_utility_t2 *= 0.1

            if self.inventory['wood'] >= Config.TIER2_WOOD_DEVICE_COST :
                utilities['build_tier2_device'] = base_utility_t2
            elif (self.archetype == 'risk_taker' or (self.archetype == 'trader' and self.role == 'wood_harvester')):
                 needed_wood_for_t2 = Config.TIER2_WOOD_DEVICE_COST - self.inventory['wood']
                 utilities['harvest_wood_for_tier2'] = harvest_efficiency_wood * price_w * 1.2
                 if self.wealth > needed_wood_for_t2 * price_w * 0.8 :
                     utilities['trade_buy_wood_for_tier2'] = needed_wood_for_t2 * price_w * 1.1

        # Buy food if role is not food harvester and food is low (but not emergency yet)
        # Tool Artisan should only do this if really necessary, handled by emergency mostly
        if self.role != 'food_harvester' and self.archetype != 'tool_artisan' and self.inventory['food'] < food_target_level and self.wealth > price_f:
            utilities['trade_buy_food'] = (food_target_level - self.inventory['food']) * price_f * 1.2

        # ** Investment & Loan Seeking (Non-Emergency) **
        # Only calculate if not already seeking a loan from emergency state
        if not (self.current_action_type and self.current_action_type == "seek_loan_for_food"):
            needs_loan = False
            loan_utility_boost = 0
            # Check conditions only if not on cooldown and hasn't failed too many times recently
            if self.default_cooldown == 0 and self.loan_request_attempts < Config.MAX_LOAN_REQUEST_ATTEMPTS:
                # Tool loan check
                can_invest_in_tool = self.wealth < self.factory.price and self.factory.price > 0 and \
                                     (self.archetype == 'risk_taker' or (self.archetype == 'trader' and self.role != 'seller'))
                # T2 wood loan check
                can_invest_in_t2_wood = not self.inventory['has_tier2_device'] and self.skill >= Config.TIER2_MIN_SKILL_TO_BUILD and \
                                        self.inventory['wood'] < Config.TIER2_WOOD_DEVICE_COST and \
                                        self.wealth < (Config.TIER2_WOOD_DEVICE_COST - self.inventory['wood']) * price_w and \
                                        (self.archetype == 'risk_taker' or (self.archetype == 'trader' and self.role == 'wood_harvester'))

                if can_invest_in_tool:
                    needs_loan = True
                    self.current_loan_purpose = "tool"
                    self.current_loan_amount_needed = self.factory.price * 1.05
                    loan_utility_boost = self.factory.price * 0.15 * (2.0 if self.archetype == 'risk_taker' else 1.0)
                    utilities['seek_loan_for_investment'] = loan_utility_boost # Add base utility first
                elif can_invest_in_t2_wood: # Prioritize T2 loan if both are true? Check utility comparison.
                    needed_for_t2_wood = (Config.TIER2_WOOD_DEVICE_COST - self.inventory['wood']) * price_w
                    needs_loan = True
                    self.current_loan_purpose = "tier2_device_wood"
                    self.current_loan_amount_needed = needed_for_t2_wood * 1.05
                    loan_utility_boost_t2 = needed_for_t2_wood * 0.15 * (2.5 if self.archetype == 'risk_taker' else 1.5)
                    # If tool loan was also possible, choose higher utility loan
                    if 'seek_loan_for_investment' in utilities:
                        if loan_utility_boost_t2 > utilities['seek_loan_for_investment']:
                             utilities['seek_loan_for_investment'] = loan_utility_boost_t2
                        else: # Keep tool loan utility, reset T2 purpose/amount
                             needs_loan = True # Keep flag true, but details are for tool
                             self.current_loan_purpose = "tool"
                             self.current_loan_amount_needed = self.factory.price * 1.05
                    else:
                         utilities['seek_loan_for_investment'] = loan_utility_boost_t2

        # ** Common Actions **
        # Training
        if self.wealth >= Config.COST_PER_SKILL:
            train_util = (Config.SKILL_GAIN_PER_TRAIN / Config.COST_PER_SKILL) * (3.0 - self.skill)
            if self.archetype == 'risk_taker': train_util *= 1.2
            if self.archetype == 'survivalist': train_util *= 0.8
            if self.archetype == 'tool_artisan': train_util *= 0.3 # Less interested in general training
            utilities['train'] = train_util

        # Buy Tools (Revised Utility)
        if self.wealth >= self.factory.price and self.factory.price > 0:
            planning_horizon = 25 # Estimate benefit over N ticks

            if self.factory.wood_tools > 0:
                gain_w_per_tick = Config.TOOL_EFFECT * price_w
                total_gain_w = gain_w_per_tick * planning_horizon
                util_buy_w = total_gain_w / self.factory.price if self.factory.price > 0 else 0
                if self.archetype == 'risk_taker': util_buy_w *= 2.5
                elif self.archetype == 'trader' and self.role == 'wood_harvester': util_buy_w *= 1.3
                elif self.archetype == 'survivalist': util_buy_w *= 0.3
                elif self.archetype == 'tool_artisan': util_buy_w *= 0.05 # Very low desire to buy tools
                if util_buy_w > 0: utilities['buy_tool_wood'] = util_buy_w

            if self.factory.food_tools > 0:
                gain_f_per_tick = Config.TOOL_EFFECT * price_f
                total_gain_f = gain_f_per_tick * planning_horizon
                util_buy_f = total_gain_f / self.factory.price if self.factory.price > 0 else 0
                if self.archetype == 'risk_taker': util_buy_f *= 2.5
                elif self.archetype == 'trader' and self.role == 'food_harvester': util_buy_f *= 1.3
                elif self.archetype == 'survivalist': util_buy_f *= 0.3
                elif self.archetype == 'tool_artisan': util_buy_f *= 0.05 # Very low desire to buy tools
                if util_buy_f > 0: utilities['buy_tool_food'] = util_buy_f

        # Buy Entertainment
        if self.wealth >= self.store.price and self.store.ent_stock > 0 and self.store.price > 0:
            ent_util = Config.ENT_EFFECT * (3.0 - self.happiness)
            if self.archetype == 'survivalist' and self.wealth < 50 : ent_util *= 0.2
            if self.archetype == 'tool_artisan': ent_util *= 0.1 # Low priority for ent
            utilities['buy_ent'] = ent_util


        # --- Final Decision Logic ---
        if not utilities:
            self.current_action_type = "idle_no_utilities"
            self.target = (random.randrange(self.grid.width), random.randrange(self.grid.height)) # Wander
            # Store idle action info
            # self.last_chosen_action = self.current_action_type
            # self.last_max_utility = 0 # Or some other indicator for idle utility
            return

        # Choose action with highest utility
        # Note: If seek_loan_for_food was set in emergency block, it might not have a utility value here.
        # We need to handle this potential conflict if the agent decides something else is better now.
        # If emergency loan seeking was triggered, prioritize it unless a VERY high utility non-loan action is found.
        emergency_loan_seeking = self.current_action_type == "seek_loan_for_food"

        chosen_action = max(utilities, key=utilities.get)
        max_utility = utilities[chosen_action]

        # --- Store chosen action and utility ---
        # self.last_chosen_action = chosen_action # Store the best option based on raw utility
        # self.last_max_utility = max_utility
        # --- END Store ---

        if emergency_loan_seeking and chosen_action != "seek_loan_for_food":
             # If a non-loan action has significantly higher utility, maybe switch off emergency loan seeking? Risky.
             # For safety, let's stick with emergency loan seeking if it was triggered.
             # However, if 'seek_loan_for_food' wasn't added to utilities (because conditions changed), we need chosen_action.
             if 'seek_loan_for_food' not in utilities:
                 # Emergency condition might have slightly changed, proceed with best calculated utility
                 self.current_action_type = chosen_action
                 logging.debug(f"A:{self.id} EMERGENCY loan seek condition changed, choosing best normal utility: {chosen_action}")
             else:
                 # Stick with emergency loan seeking if it's still in utilities or was the original intent.
                 self.current_action_type = "seek_loan_for_food"
                 # Target setting below will handle finding lender
                 # Update stored action if overridden by emergency
                 # self.last_chosen_action = self.current_action_type
                 # Utility for emergency loan seeking might not be in the dict, use a placeholder or lookup
                 # self.last_max_utility = utilities.get("seek_loan_for_food", float('inf')) # Indicate high priority
        elif self.current_action_type and self.current_action_type.startswith("find_buyer_for_tool_"): # If we decided to find buyer, stick to it
            pass # Keep the action, target setting below will handle it
        else:
             # Normal operation or non-emergency loan seeking
             self.current_action_type = chosen_action

        # --- Set Target based on chosen_action ---
        self.target = None # Reset target, set based on action

        action_key_final = self.current_action_type # Use the potentially overridden action type

        if action_key_final.startswith('harvest_'):
            res_type = 'food' if 'food' in action_key_final else 'wood'
            # Handle artisan specific harvest key
            if action_key_final == 'harvest_wood_for_factory': res_type = 'wood'
            nodes = [n for n in self.grid.nodes if n.resource_type == res_type and n.capacity > 0]
            if nodes: self.target = min(nodes, key=lambda n: self._get_distance(self.pos, n.pos)).pos
            else: self.current_action_type = f"idle_no_{res_type}_node"

        elif action_key_final == 'train': self.target = self.train_center.pos
        elif action_key_final.startswith('buy_tool_'): self.target = self.factory.pos
        elif action_key_final == 'buy_ent': self.target = self.store.pos
        elif action_key_final == 'build_tier2_device': self.target = self.pos
        elif action_key_final == 'deliver_wood_to_factory': self.target = self.factory.pos # Target factory for delivery
        elif action_key_final.startswith('go_get_tool_from_factory_'): self.target = self.factory.pos # Target factory to pick up tool
        elif action_key_final.startswith('find_buyer_for_tool_'):
            tool_type = 'wood' if 'wood' in action_key_final else 'food'
            # Find potential buyers: agents who could use the tool and might afford it
            potential_buyers = []
            for agent in self.agents_ref:
                if agent is not self and agent.wealth > self.factory.price * 1.1: # Basic affordability check
                    # Higher need if they are a harvester of that type with few tools
                    needs_tool = False
                    if tool_type == 'wood' and agent.role == 'wood_harvester' and agent.tools['wood'] < 2:
                        needs_tool = True
                    elif tool_type == 'food' and agent.role == 'food_harvester' and agent.tools['food'] < 2:
                        needs_tool = True
                    # Risk takers might buy speculatively
                    if agent.archetype == 'risk_taker': needs_tool = True

                    if needs_tool:
                        potential_buyers.append(agent)

            if potential_buyers:
                potential_buyers.sort(key=lambda b: (b.archetype == 'trader', self._get_distance(self.pos, b.pos))) # Prefer non-traders first? Then distance.
                self.trade_partner = potential_buyers[0]
                self.target = self.trade_partner.pos
                # Set trade item type for the trade logic
                self.trade_item_type = f'tool_{tool_type}'
                self.trade_is_buy = False # Seller perspective
                logging.debug(f"A:{self.id} (Seller) finding buyer for {tool_type} tool, targeting A:{self.trade_partner.id}")
            else:
                self.current_action_type = "idle_no_tool_buyer_found" # No suitable buyer found
                self.target = None

        elif action_key_final.startswith('trade_'):
            self.trade_is_buy = 'buy' in action_key_final
            self.trade_item_type = 'food' if 'food' in action_key_final else 'wood'

            potential_partners = []
            # ... (Logic for finding trade partners based on buy/sell need, preferring sellers/traders etc. - same as before) ...
            if self.trade_is_buy:
                potential_partners = [a for a in self.agents_ref if a is not self and a.inventory.get(self.trade_item_type, 0) > 0]
                if potential_partners: potential_partners.sort(key=lambda p: (not (p.role == 'seller' or p.archetype == 'trader'), self._get_distance(self.pos, p.pos)))
            else:
                potential_partners = [a for a in self.agents_ref if a is not self]
                if potential_partners: potential_partners.sort(key=lambda p: ( not ( (p.role == 'seller' or p.archetype == 'trader') and p.inventory.get(self.trade_item_type, 0) < (Config.FOOD_TARGET_LEVEL_SELLER if self.trade_item_type == 'food' else Config.WOOD_TARGET_LEVEL_SELLER) ), p.inventory.get(self.trade_item_type, 0) >= (p._archetype_food_target() if self.trade_item_type == 'food' else 5), self._get_distance(self.pos, p.pos) ))

            if potential_partners: self.trade_partner = potential_partners[0]; self.target = self.trade_partner.pos
            else: self.current_action_type = "idle_no_trade_partner"

        elif action_key_final.startswith("seek_loan_"):
            # Increment attempts ONLY when setting the target, prevents double counting if decision changes
            if self.target is None: self.loan_request_attempts +=1

            potential_lenders = [
                a for a in self.agents_ref if a is not self and a.default_cooldown == 0 and
                len(a.loans_given) < 5 and # Avoid lenders with too many loans out
                (
                    # Trader condition: enough wealth beyond minimum + loan amount
                    (a.archetype == 'trader' and a.wealth > Config.TRADER_LOAN_MIN_WEALTH_TO_LEND + self.current_loan_amount_needed) or \
                    # Non-trader condition: much wealthier than loan amount, not in debt themselves
                    (a.wealth > self.current_loan_amount_needed * 3.0 and len(a.loans_taken) == 0)
                )
            ]
            if potential_lenders:
                potential_lenders.sort(key=lambda l: (l.archetype != 'trader', self._get_distance(self.pos, l.pos))) # Prefer traders
                self.trade_partner = potential_lenders[0] # Use trade_partner for loan interaction target
                self.target = self.trade_partner.pos
                logging.debug(f"Agent {self.id} aiming to seek loan for {self.current_loan_purpose} (Amt:{self.current_loan_amount_needed:.1f}), targeting Agent {self.trade_partner.id} ({self.trade_partner.archetype[0]})")
            else:
                # No suitable lender found this tick
                logging.debug(f"Agent {self.id} could not find a suitable lender this tick for {self.current_loan_purpose} (Amt:{self.current_loan_amount_needed:.1f}). Attempts left: {Config.MAX_LOAN_REQUEST_ATTEMPTS - self.loan_request_attempts}")
                self.current_action_type = "idle_no_lender_found"; self.target = None; self.trade_partner = None
                if self.loan_request_attempts >= Config.MAX_LOAN_REQUEST_ATTEMPTS:
                    # Reset attempts after cooldown period (e.g., half loan duration?)
                    # For now, just log that attempts are exhausted. Agent needs another strategy.
                    logging.warning(f"Agent {self.id} exhausted loan attempts for {self.current_loan_purpose}.")

        elif action_key_final == 'start_project':
            # Find an empty spot for the project
            found_spot = False
            for _try in range(15): # Try more spots
                px, py = random.randrange(self.grid.width), random.randrange(self.grid.height)
                # Check if spot is valid (not special building, not existing node, not existing project)
                if (px,py) not in [Config.TRAIN_CENTER_POS, Config.TOOL_FACTORY_POS, Config.STORE_POS] and \
                   not self.grid.get_node_at((px,py)) and not any(p.pos == (px,py) for p in self.projects):
                    self.target = (px,py)
                    found_spot = True
                    break
            if not found_spot:
                self.current_action_type = "idle_no_project_spot"

        # Final check: if action decided but no target found/set, revert to idle
        if self.target is None and not (self.current_action_type and self.current_action_type.startswith("idle_")):
            logging.debug(f"Agent {self.id} chose action {self.current_action_type} but no target could be set. Idling.")
            self.current_action_type = "idle_target_not_found"
            self.target = (random.randrange(self.grid.width), random.randrange(self.grid.height)) # Wander
            # Store idle action info
            # self.last_chosen_action = self.current_action_type
            # self.last_max_utility = 0

    def step(self, price_w: float, price_f: float, current_tick: int, all_loans_ref: List['Loan']):
        # Reset status trackers for this step
        self.last_action_outcome = None
        self.last_starving_status = False
        status_info = {'status': 'alive', 'action_outcome': None, 'is_starving': False} # Local status for internal logic
        action_taken_this_step = False

        self._handle_loan_repayments_and_accruals(current_tick, price_f)

        # --- Hunger & Survival ---
        passive_food_cost = 1.0 / Config.HUNGER_INTERVAL # Constant metabolic cost per tick
        self.food_hunger += 1 # Hunger always increases

        if self.inventory['food'] >= passive_food_cost:
            self.inventory['food'] -= passive_food_cost
            self.happiness = max(0, self.happiness - 0.01) # Small constant happiness drain
            # Reset hunger if food level is okay? No, let it accumulate for penalty timing.
        elif self.inventory['food'] > 0:
            # Consume the remaining fraction
            self.inventory['food'] = 0
            self.happiness = max(0, self.happiness - 0.01)
        # Else: food is already 0, do nothing regarding consumption

        # Check for starvation penalties if out of food AND hunger threshold reached
        if self.inventory['food'] == 0 and self.food_hunger >= Config.HUNGER_INTERVAL:
             self.wealth -= 0.5 # Penalty for starving
             self.happiness = max(0, self.happiness - 0.1) # Larger happiness hit for starving
             status_info['is_starving'] = True # Mark as starving for internal logic / RL reward
             self.last_starving_status = True # Mark for RL info dict
             logging.warning(f"Agent {self.id} ({self.archetype[0]},{self.role}) STARVING! F:0 H:{self.food_hunger} W:{self.wealth:.1f}")
             self.food_hunger = 0 # Reset hunger penalty timer after applying penalty
             if self.wealth < Config.AGENT_DEATH_WEALTH_THRESHOLD :
                 status_info['status'] = 'dead_starvation'
                 return status_info # Agent died
        elif self.inventory['food'] > 0:
             # If agent has food, reset hunger timer periodically maybe?
             # Or reset it when they eat? Let's reset if they are NOT starving this tick.
             # Resetting here prevents rapid penalties if they fluctuate near 0 food.
             if self.food_hunger >= Config.HUNGER_INTERVAL: # Reset if they have food and interval passed
                 self.food_hunger = 0

        # --- RL Agent Control Check ---
        if self.is_rl_controlled:
            # If RL agent is controlling, we assume the RL environment (AgentEconomyEnv)
            # has already called _map_action_to_agent_task() or similar
            # to set self.current_action_type, self.target, self.trade_partner etc.
            # We should skip the Agent's internal decide_target logic.
            pass # RL controls the target/action
        else:
            # --- Normal Action Decision ---
            # If starving and current action is not helpful for food, force re-evaluation
            if self.emergency_food_seeking and self.current_action_type and \
               not (self.current_action_type.startswith("harvest_food") or \
                    self.current_action_type.startswith("trade_buy_food") or \
                    self.current_action_type.startswith("seek_loan_for_food")):
                logging.debug(f"Agent {self.id} EMERGENCY, current action {self.current_action_type} not food related. Re-evaluating.")
                self.current_action_type = None

            if self.current_action_type is None or self.target is None or self.current_action_type.startswith("idle_") :
                self.decide_target(price_w, price_f, current_tick)


        # --- Movement ---
        if self.target and self.pos != self.target:
            x, y = self.pos; tx, ty = self.target
            dx = int(np.sign(tx - x)); dy = int(np.sign(ty - y))
            # Prefer diagonal, then straight
            if dx != 0 and dy != 0: self.pos = (x + dx, y + dy)
            elif dx != 0: self.pos = (x + dx, y)
            elif dy != 0: self.pos = (x, y + dy)

        # --- Perform Action at Target ---
        if self.pos == self.target and self.current_action_type and not self.current_action_type.startswith("idle_"):
            action_key = self.current_action_type
            outcome_detail = None # Store specific outcome string

            # --- Refactor action blocks to set outcome_detail ---
            if action_key.startswith('harvest_food'): # Covers emergency harvest too
                node = self.grid.get_node_at(self.pos)
                if node and node.resource_type == 'food' and node.capacity > 0:
                    eff_harvest = self.skill + self.tools['food'] * Config.TOOL_EFFECT
                    amount = node.harvest(eff_harvest)
                    self.inventory['food'] += amount; self.skill += 0.005 * amount
                    if node.fee_receiver is not None and node.fee_receiver != self.id: # Pay fee
                        fee = amount * Config.HARVEST_FEE * price_f
                        self.wealth -= fee
                        receiver = next((a for a in self.agents_ref if a.id == node.fee_receiver), None)
                        if receiver: receiver.wealth += fee
                    logging.debug(f"A:{self.id} harvested {amount:.1f} food. Total F:{self.inventory['food']:.1f}")
                    action_taken_this_step = True
                    if 'emergency' in action_key: self.emergency_food_seeking = False # Harvested food in emergency
                    outcome_detail = f"harvested_food_{amount:.1f}"

            elif action_key.startswith('harvest_wood'): # Covers variants like _for_project, _for_factory, _for_tier2
                node = self.grid.get_node_at(self.pos)
                if node and node.resource_type == 'wood' and node.capacity > 0:
                    eff_harvest = self.skill + self.tools['wood'] * Config.TOOL_EFFECT
                    if self.inventory['has_tier2_device']: eff_harvest *= Config.TIER2_WOOD_DEVICE_EFFECT_MULTIPLIER
                    amount = node.harvest(eff_harvest)
                    self.inventory['wood'] += amount; self.skill += 0.005 * amount
                    if node.fee_receiver is not None and node.fee_receiver != self.id: # Pay fee
                        fee = amount * Config.HARVEST_FEE * price_w
                        self.wealth -= fee
                        receiver = next((a for a in self.agents_ref if a.id == node.fee_receiver), None)
                        if receiver: receiver.wealth += fee
                    logging.debug(f"A:{self.id} harvested {amount:.1f} wood. Total W:{self.inventory['wood']:.1f}")
                    action_taken_this_step = True
                    outcome_detail = f"harvested_wood_{amount:.1f}"

            elif action_key == 'train' and self.pos == self.train_center.pos:
                if self.wealth >= Config.COST_PER_SKILL:
                    self.wealth -= Config.COST_PER_SKILL; self.skill += Config.SKILL_GAIN_PER_TRAIN
                    logging.debug(f"A:{self.id} trained. Skill:{self.skill:.2f}")
                    action_taken_this_step = True
                    outcome_detail = "trained"
                else:
                    outcome_detail = "failed_train_wealth"

            elif action_key.startswith('buy_tool_') and self.pos == self.factory.pos:
                tool_type = 'wood' if 'wood' in action_key else 'food'
                if self.wealth >= self.factory.price and self.factory.price > 0:
                    if tool_type == 'wood' and self.factory.wood_tools > 0:
                        self.factory.wood_tools -= 1; self.tools['wood'] += 1; self.wealth -= self.factory.price; action_taken_this_step = True
                    elif tool_type == 'food' and self.factory.food_tools > 0:
                        self.factory.food_tools -= 1; self.tools['food'] += 1; self.wealth -= self.factory.price; action_taken_this_step = True
                    if action_taken_this_step:
                        logging.debug(f"A:{self.id} bought {tool_type} tool.")
                        self.factory.produce()
                        outcome_detail = f"bought_tool_{tool_type}"
                    else:
                        outcome_detail = f"failed_buy_tool_{tool_type}_stock"
                else:
                    outcome_detail = f"failed_buy_tool_{tool_type}_wealth"

            elif action_key == 'buy_ent' and self.pos == self.store.pos:
                if self.store.ent_stock > 0 and self.wealth >= self.store.price and self.store.price > 0:
                    self.store.ent_stock -= 1; self.wealth -= self.store.price; self.happiness += Config.ENT_EFFECT; action_taken_this_step = True
                    if action_taken_this_step:
                        logging.debug(f"A:{self.id} bought ent. Happiness:{self.happiness:.2f}")
                        self.store.produce()
                        outcome_detail = "bought_ent"
                    else:
                         outcome_detail = "failed_buy_ent_stock"
                else:
                     outcome_detail = "failed_buy_ent_wealth"

            elif action_key == 'build_tier2_device': # Already at self.pos, target was self.pos
                if not self.inventory['has_tier2_device'] and self.inventory['wood'] >= Config.TIER2_WOOD_DEVICE_COST:
                    self.inventory['wood'] -= Config.TIER2_WOOD_DEVICE_COST
                    self.inventory['has_tier2_device'] = True
                    logging.info(f"Agent {self.id} ({self.archetype[0]}) BUILT Tier 2 Wood Device!")
                    action_taken_this_step = True
                    outcome_detail = "built_t2"
                else:
                    logging.warning(f"Agent {self.id} failed to build T2 device (not enough wood or already has) when at target. Wood: {self.inventory['wood']}")
                    outcome_detail = "failed_build_t2"

            elif action_key.startswith("seek_loan_") and self.trade_partner and self.pos == self.trade_partner.pos:
                lender = self.trade_partner # trade_partner is the potential lender
                amount_to_borrow = self.current_loan_amount_needed
                purpose_of_loan = self.current_loan_purpose

                can_lend = False
                interest_rate = Config.TRADER_LOAN_INTEREST_RATE_BASE
                if lender.archetype == 'trader' and \
                   lender.wealth > Config.TRADER_LOAN_MIN_WEALTH_TO_LEND + amount_to_borrow and \
                   len(lender.loans_given) < 5 : # Trader has enough wealth and not too many active loans
                    can_lend = True
                    # Dynamic interest based on borrower's situation (e.g. existing debt, default history)
                    risk_premium = self.default_cooldown / (Config.LOAN_DEFAULT_PENALTY_TICKS * 2) if Config.LOAN_DEFAULT_PENALTY_TICKS > 0 else 0 # Max 0.5 if just defaulted
                    risk_premium += len(self.loans_taken) * 0.001 # Small premium per existing loan
                    interest_rate = Config.TRADER_LOAN_INTEREST_RATE_BASE + risk_premium
                    if self.wealth < amount_to_borrow * 0.2: interest_rate += 0.002 # Very low wealth borrower
                elif lender.wealth > amount_to_borrow * 3 and len(lender.loans_taken) == 0 and len(lender.loans_given) < 2: # Non-trader, wealthy, not in debt, few loans given
                    can_lend = True
                    interest_rate = Config.TRADER_LOAN_INTEREST_RATE_BASE * 1.5 + (self.default_cooldown / (Config.LOAN_DEFAULT_PENALTY_TICKS*2) if Config.LOAN_DEFAULT_PENALTY_TICKS >0 else 0)

                interest_rate = np.clip(interest_rate, Config.LOAN_INTEREST_MIN, Config.LOAN_INTEREST_MAX)

                if can_lend and amount_to_borrow > 0:
                    new_loan = Loan(lender.id, self.id, amount_to_borrow, interest_rate, current_tick, Config.LOAN_DURATION_TICKS)
                    all_loans_ref.append(new_loan) # Add to global list of all loans in the economy

                    lender.wealth -= amount_to_borrow; lender.loans_given.append(new_loan)
                    self.wealth += amount_to_borrow; self.loans_taken.append(new_loan)
                    self.loan_request_attempts = 0 # Reset attempts on successful loan

                    logging.info(f"LOAN GRANTED: L:{lender.id}({lender.archetype[0]}) to B:{self.id}({self.archetype[0]}). Amt:{amount_to_borrow:.1f} Purp:{purpose_of_loan}. {new_loan}")
                    action_taken_this_step = True
                    if purpose_of_loan == "emergency_food": self.emergency_food_seeking = False # Loan for food might resolve emergency
                    outcome_detail = f"got_loan_{purpose_of_loan}"
                else:
                    logging.debug(f"LOAN DENIED: Lender {lender.id}({lender.archetype[0]}) to Borrower {self.id}. (L.Wealth:{lender.wealth:.1f}, Amt.Needed:{amount_to_borrow:.1f})")
                    outcome_detail = f"loan_denied_{purpose_of_loan}"

                self.trade_partner = None # Clear interaction partner

            elif action_key and action_key.startswith('trade_') and self.trade_partner and self.pos == self.trade_partner.pos:
                partner = self.trade_partner
                item = self.trade_item_type
                base_market_price = price_f if item == 'food' else price_w

                # Price negotiation based on archetypes
                final_trade_price = base_market_price
                if self.archetype == 'trader':
                    final_trade_price *= (random.uniform(0.85,0.95) if self.trade_is_buy else random.uniform(1.05,1.15))
                elif partner.archetype == 'trader':
                    final_trade_price *= (random.uniform(1.05,1.15) if self.trade_is_buy else random.uniform(0.85,0.95))
                final_trade_price = max(Config.MIN_PRICE, final_trade_price)

                # Execute trade
                if self.trade_is_buy: # Self is Buyer
                    # Emergency buy food has higher willingness to pay
                    if action_key == "trade_buy_food_emergency": final_trade_price = base_market_price * random.uniform(1.1, 1.3)
                    final_trade_price = min(self.wealth, final_trade_price) # Can't pay more than I have

                    partner_min_sell_food = 1 if partner.archetype != 'survivalist' else partner._archetype_food_target() * 0.5
                    if partner.inventory.get(item, 0) > (partner_min_sell_food if item == 'food' else 0) and self.wealth >= final_trade_price:
                        partner.inventory[item] -= 1; partner.wealth += final_trade_price
                        self.inventory[item] = self.inventory.get(item, 0) + 1; self.wealth -= final_trade_price
                        logging.debug(f"A:{self.id} BOUGHT {item} from A:{partner.id} for ${final_trade_price:.2f}")
                        action_taken_this_step = True
                        outcome_detail = f"bought_{item}_from_{partner.id}"
                        if action_key == "trade_buy_food_emergency": self.emergency_food_seeking = False
                    else:
                        outcome_detail = f"failed_buy_{item}_from_{partner.id}"
                else: # Self is Seller
                    # --- Check if selling a tool ---
                    is_selling_tool = item and item.startswith('tool_')
                    if is_selling_tool:
                        tool_type = item.split('_')[1] # 'wood' or 'food'
                        if self.inventory['tools_for_sale'][tool_type] > 0:
                            # Dynamic pricing for tools - seller tries to mark up factory price
                            sell_price = self.factory.price * random.uniform(1.1, 1.4) # 10-40% markup
                            if self.archetype == 'trader': sell_price *= random.uniform(1.1, 1.3) # Traders push higher

                            # Buyer willingness check (crude): willing to pay up to X% more than factory price based on need/wealth
                            buyer_willingness_factor = 1.1
                            if partner.archetype == 'risk_taker': buyer_willingness_factor = 1.3
                            # Simplified: If buyer role matches tool and few tools, more willing
                            if (tool_type == 'wood' and partner.role == 'wood_harvester' and partner.tools['wood'] < 1) or \
                               (tool_type == 'food' and partner.role == 'food_harvester' and partner.tools['food'] < 1):
                                buyer_willingness_factor = 1.4

                            buyer_max_pay = self.factory.price * buyer_willingness_factor
                            final_trade_price = min(sell_price, buyer_max_pay, partner.wealth) # Buyer pays minimum of seller ask, their max, their wealth
                            final_trade_price = max(Config.MIN_PRICE, final_trade_price)

                            # Check if buyer accepts the final price
                            if partner.wealth >= final_trade_price and final_trade_price > self.factory.price * 0.9: # Buyer accepts if they can afford and price isn't below factory cost
                                self.inventory['tools_for_sale'][tool_type] -= 1
                                self.wealth += final_trade_price
                                partner.tools[tool_type] += 1 # Add to buyer's usable tools
                                partner.wealth -= final_trade_price
                                logging.debug(f"A:{self.id} (Seller) SOLD {tool_type} tool to A:{partner.id} for ${final_trade_price:.2f} (Fac Price: {self.factory.price:.2f})")
                                action_taken_this_step = True
                                outcome_detail = f"sold_{tool_type}_to_{partner.id}"
                            else:
                                logging.debug(f"A:{self.id} (Seller) failed tool sale to A:{partner.id}. Offer: ${final_trade_price:.2f}, Buyer Wealth: {partner.wealth:.2f}, Buyer Max: {buyer_max_pay:.2f}")
                                outcome_detail = f"failed_sell_{tool_type}_to_{partner.id}"
                        else:
                           logging.warning(f"A:{self.id} (Seller) tried to sell {tool_type} tool to A:{partner.id} but had none in sale inventory.")
                           outcome_detail = f"failed_sell_{tool_type}_to_{partner.id}_no_stock"
                    # --- End tool selling check ---
                    else: # Selling regular resource (wood/food)
                        final_trade_price = min(partner.wealth, final_trade_price) # Partner can't pay more than they have
                        if self.inventory.get(item, 0) > 0 and partner.wealth >= final_trade_price:
                            self.inventory[item] -= 1; self.wealth += final_trade_price
                            partner.inventory[item] = partner.inventory.get(item, 0) + 1; partner.wealth -= final_trade_price
                            logging.debug(f"A:{self.id} SOLD {item} to A:{partner.id} for ${final_trade_price:.2f}")
                            action_taken_this_step = True
                            outcome_detail = f"sold_{item}_to_{partner.id}"
                        else:
                             outcome_detail = f"failed_sell_{item}_to_{partner.id}"
                self.trade_partner = None # Clear interaction partner

            elif action_key == 'start_project' and self.role == 'developer':
                if self.inventory['wood'] >= Config.PROJECT_WOOD_COST:
                    if not self.grid.get_node_at(self.pos) and not any(p.pos == self.pos for p in self.projects): # Check if spot is valid
                        self.inventory['wood'] -= Config.PROJECT_WOOD_COST
                        project_res_type = random.choice(['wood', 'food'])
                        new_project = Project(self.pos, project_res_type, self.id)
                        self.projects.append(new_project) # Economy handles adding to global list from agent's list? NO! Economy must add it.
                        logging.info(f"Agent {self.id} (Developer) started a {project_res_type} project at {self.pos}")
                        action_taken_this_step = True
                        outcome_detail = f"started_project_{project_res_type}"
                    else:
                        logging.warning(f"Agent {self.id} (Dev) project site {self.pos} occupied/invalid.")
                        outcome_detail = "failed_start_project_occupied"
                else:
                    logging.warning(f"Agent {self.id} (Dev) at project site {self.pos} but lacks wood.")
                    outcome_detail = "failed_start_project_no_wood"

            # --- NEW: Handle Tool Artisan delivering wood to factory ---
            elif action_key == 'deliver_wood_to_factory' and self.pos == self.factory.pos:
                if self.inventory['wood'] >= Config.WOOD_PER_TOOL: # Check again just in case
                    # Deliver slightly more than base cost if available, up to a limit
                    amount_to_deliver = min(self.inventory['wood'], Config.WOOD_PER_TOOL * 3)
                    self.inventory['wood'] -= amount_to_deliver
                    self.factory.wood_stock += amount_to_deliver
                    # Pay the artisan for the wood at a slight premium over market?
                    payment = amount_to_deliver * price_w * 1.05 # Changed from self.price_w
                    self.wealth += payment
                    logging.debug(f"A:{self.id} (Artisan) delivered {amount_to_deliver:.1f} wood to factory. Fac stock: {self.factory.wood_stock:.1f}. Earned: ${payment:.2f}")
                    action_taken_this_step = True
                    outcome_detail = "delivered_wood_to_factory"
                    self.factory.produce() # Trigger factory production after delivery
                else:
                    logging.warning(f"A:{self.id} (Artisan) at factory {self.pos} but lacked wood ({self.inventory['wood']:.1f}) for delivery.")
                    outcome_detail = "failed_deliver_wood_to_factory"
            # --- END NEW ---

            # --- NEW: Handle Seller getting tool from factory ---
            elif action_key.startswith('go_get_tool_from_factory_') and self.pos == self.factory.pos:
                tool_type = 'wood' if 'wood' in action_key else 'food'
                can_get = False
                inventory_ref = tool_type # 'wood' or 'food'
                tool_stock_ref = f"{tool_type}_tools" # e.g., 'wood_tools' or 'food_tools'

                if tool_type == 'wood' and self.factory.wood_tools > 0:
                    can_get = True
                elif tool_type == 'food' and self.factory.food_tools > 0:
                    can_get = True

                if can_get and self.wealth >= self.factory.price:
                    setattr(self.factory, tool_stock_ref, getattr(self.factory, tool_stock_ref) - 1) # Decrease factory stock
                    self.wealth -= self.factory.price
                    self.inventory['tools_for_sale'][inventory_ref] += 1 # Add to seller's sale inventory
                    logging.debug(f"A:{self.id} (Seller) picked up {tool_type} tool from factory for ${self.factory.price:.2f}. For Sale: W:{self.inventory['tools_for_sale']['wood']} F:{self.inventory['tools_for_sale']['food']}")
                    action_taken_this_step = True
                    outcome_detail = f"got_tool_{tool_type}_from_factory"
                else:
                    logging.warning(f"A:{self.id} (Seller) at factory {self.pos} failed to get {tool_type} tool (Stock: {getattr(self.factory, tool_stock_ref, 0)}, Price: {self.factory.price:.2f}, Wealth: {self.wealth:.2f})")
                    outcome_detail = f"failed_get_tool_{tool_type}_from_factory"
            # --- END NEW ---

            self.last_action_outcome = outcome_detail # Store detailed outcome

            # If an action was taken or target is no longer valid, reset for next decision cycle
            if action_taken_this_step:
                self.current_action_type = None
                self.target = None

        status_info['action_outcome'] = self.last_action_outcome # Pass outcome back

        # If agent is idle or finished task (and not in an emergency that dictates next step), it should re-evaluate next tick
        # This is handled at the start of the non-RL decision block now.

        return status_info # Return final status for this step
