from typing import Optional

# Configuration constants
class Config:
    SEED: Optional[int] = None
    GRID_WIDTH = 25
    GRID_HEIGHT = 15
    CELL_SIZE = 40
    WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
    WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE
    FPS = 5

    INITIAL_PRICE_WOOD = 1.0
    INITIAL_PRICE_FOOD = 1.0
    PRICE_ADJUST_ALPHA = 0.1
    MIN_PRICE = 0.1 # Minimum price for any trade

    TRAIN_CENTER_POS = (0, 0)
    COST_PER_SKILL = 5.0
    SKILL_GAIN_PER_TRAIN = 0.5

    HUNGER_INTERVAL = 50 # Agent eats every 5 steps
    FOOD_TARGET_LEVEL = 4 # Agents try to keep at least this much food (Increased from 3)
    WOOD_TARGET_LEVEL_WOOD_HARVESTER = 10 
    WOOD_TARGET_LEVEL_SELLER = 5 
    FOOD_TARGET_LEVEL_SELLER = 5 
    AGENT_DEATH_WEALTH_THRESHOLD = -5 # Agent dies if wealth drops below this (was implicit 0)


    TOOL_FACTORY_POS = (GRID_WIDTH - 1, GRID_HEIGHT - 1)
    WOOD_PER_TOOL = 5
    TOOL_MARKUP = 5.0
    TOOL_EFFECT = 0.5

    STORE_POS = (GRID_WIDTH//2, GRID_HEIGHT//2)
    WOOD_PER_ENT = 50
    ENT_MARKUP = 1.5
    ENT_EFFECT = 0.2

    SELLER_SALARY = 0.1 
    SELLER_COMMISSION = 0.05 

    NUM_TRADES_PER_TICK = 3 
    PROJECT_WOOD_COST = 10
    PROJECT_TIME = 5
    HARVEST_FEE = 0.2 

    ROLES = ['food_harvester', 'wood_harvester', 'seller', 'developer']
    ROLE_CHANGE_INTERVAL = 10 
    ROLE_CHANGE_INERTIA = 0.15 
    DEVELOPER_WOOD_PRIORITY_MULTIPLIER = 1.5
    AGENT_ARCHETYPES = ['survivalist', 'risk_taker', 'trader', 'tool_artisan'] # Added 'tool_artisan'
    
    # Archetype specific params (examples, can be tuned)
    SURVIVALIST_FOOD_TARGET_MULTIPLIER = 1.5 # Hoards more food
    RISKTAKER_INVESTMENT_THRESHOLD = 0.3 # Willing to spend X% of wealth on investment
    TRADER_LOAN_INTEREST_RATE_BASE = 0.01 # 1% per tick base
    TRADER_LOAN_MIN_WEALTH_TO_LEND = 50 # Min wealth a trader needs to offer loans
    LOAN_DEFAULT_PENALTY_TICKS = 50 # Cooldown before defaulted agent can easily get new loans

    # Tier 2 Wood Device
    TIER2_WOOD_DEVICE_COST = 30 # Wood cost
    TIER2_WOOD_DEVICE_EFFECT_MULTIPLIER = 10.0 # e.g., doubles skill-based wood harvesting part
    TIER2_MIN_SKILL_TO_BUILD = 1.0 # Min skill to consider building this

    # Loan System
    MAX_LOAN_REQUEST_ATTEMPTS = 3
    LOAN_DURATION_TICKS = 50 # How long a loan lasts
    LOAN_INTEREST_MIN = 0.005 # 0.5% per tick
    LOAN_INTEREST_MAX = 0.05  # 5% per tick
    LOAN_REPAYMENT_PRIORITY = 0.3 # Fraction of available wealth to use for repayment 