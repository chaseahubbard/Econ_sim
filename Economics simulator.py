import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
class Skill:
    def __init__(self, name, level=0, experience=0):
        self.name = name
        self.level = level
        self.experience = experience
        self.next_level_exp = 100  # Base experience needed for next level

    def gain_experience(self, amount):
        self.experience += amount
        while self.experience >= self.next_level_exp:
            self.level_up()

    def level_up(self):
        self.level += 1
        self.experience -= self.next_level_exp
        self.next_level_exp = int(self.next_level_exp * 1.5)  # Increase exp needed for next level

class Machine:
    def __init__(self, name, position, skill_boost, resource_consumption):
        self.name = name
        self.position = np.array(position)
        self.skill_boost = skill_boost  # Dictionary of skills and their boost amounts
        self.resource_consumption = resource_consumption  # Dictionary of resources needed per use
        self.owner = None
        self.in_use = False
        self.maintenance_cost = sum(resource_consumption.values()) * 2  # Base maintenance cost

    def can_operate(self, agent):
        """Check if agent has enough resources to operate the machine"""
        return all(agent.inventory.get(resource, 0) >= amount 
                  for resource, amount in self.resource_consumption.items())

    def consume_resources(self, agent):
        """Consume resources needed for operation"""
        for resource, amount in self.resource_consumption.items():
            agent.inventory[resource] = agent.inventory.get(resource, 0) - amount

class Job:
    def __init__(self, name, position, reward, required_skills, training_job=False):
        self.name = name
        self.position = np.array(position)
        self.reward = reward  # Can be positive (paying job) or negative (training)
        self.required_skills = required_skills  # Dictionary of skill names and minimum levels
        self.training_job = training_job  # Whether this is a training job
        self.experience_reward = {skill: 20 for skill in required_skills}  # Base experience gain
        self.completion_time = 5  # Base time to complete job
        self.machine_compatible = True  # Whether machines can help with this job


class DynamicAI:
    def __init__(self, market):
        self.market = market
        self.goods_list = ['good1', 'good2', 'good3']
        self.trading_radius = 15  # Agents can trade within 15 units of distance
        
    def run_market_cycle(self):
        # Agents update their internal demand
        for agent in self.market.agents:
            agent.update_internal_demand(self.goods_list)
        # Agents attempt to trade with nearby agents
        self.execute_trades()

    def update_internal_demand(self):
        # Define a scaling factor for desired inventory
        scaling_factor = 10

        # Calculate internal demand for each agent
        for agent in self.market.agents:
            agent.update_internal_demand(self.goods_list)
            agent.desired_inventory = {}
            agent.internal_demand = {}
            for good in self.goods_list:
                agent.desired_inventory[good] = agent.values_1.get(good, 0) * scaling_factor
                agent.internal_demand[good] = agent.desired_inventory[good] - agent.inventory.get(good, 0)

    def execute_trades(self):
        # Buyers initiate the trading process
        for buyer in self.market.agents:
            for good in self.goods_list:
                if buyer.internal_demand.get(good, 0) > 0:
                    # Buyer wants to buy this good
                    nearby_agents = buyer.find_nearby_agents(self.market.agents, radius=self.trading_radius)
                    potential_sellers = [a for a in nearby_agents if a != buyer and a.inventory.get(good, 0) > 0]
                    if potential_sellers:
                        buyer_max_price = buyer.values_1.get(good, 0)  # Buyer's maximum acceptable price
                        # Start the auction at buyer's maximum acceptable price
                        current_price = buyer_max_price
                        # Sellers will undercut each other until they reach their minimum acceptable price
                        sellers_in_auction = potential_sellers.copy()
                        selected_seller = None  # Initialize selected_seller
                        while True:
                            # Each seller decides if they are willing to offer at the current price
                            willing_sellers = []
                            for seller in sellers_in_auction:
                                seller_min_price = seller.values_1.get(good, 0)  # Seller's minimum acceptable price
                                if current_price >= seller_min_price:
                                    willing_sellers.append(seller)
                            if not willing_sellers:
                                # No sellers willing to sell at this price
                                break
                            if len(willing_sellers) == 1:
                                # Only one seller willing to sell at this price
                                selected_seller = willing_sellers[0]
                                break
                            else:
                                # Multiple sellers willing to sell, reduce price slightly
                                current_price -= 0.01
                                # Ensure price does not go below the lowest seller's minimum acceptable price
                                min_seller_price = min(s.values_1.get(good, 0) for s in willing_sellers)
                                if current_price < min_seller_price:
                                    current_price = min_seller_price
                            # Check if price cannot be reduced further
                            if current_price <= 0:
                                break
                        if selected_seller is None:
                            # No seller was selected, move to next good or buyer
                            continue
                        # Determine the maximum quantity buyer can afford at this price
                        if current_price == 0:
                            continue  # Prevent division by zero
                        max_affordable_quantity = buyer.wealth // current_price
                        # Determine the quantity to trade
                        quantity = min(
                            buyer.internal_demand[good],
                            selected_seller.inventory[good],
                            max_affordable_quantity
                        )
                        quantity = int(quantity)
                        if quantity >= 1:
                            total_price = current_price * quantity
                            # Execute trade
                            buyer.spend(total_price)
                            selected_seller.earn(total_price)
                            selected_seller.inventory[good] -= quantity
                            buyer.inventory[good] = buyer.inventory.get(good, 0) + quantity
                            # Update internal demand
                            selected_seller.internal_demand[good] += quantity  # Seller's demand decreases
                            buyer.internal_demand[good] -= quantity
                            # Print transaction details
                            print(f"\nTransaction executed between {selected_seller.name} and {buyer.name}:")
                            print(f"{selected_seller.name} sold {quantity} units of {good} to {buyer.name} at {current_price:.2f} per unit.")
                            print("Current Market State:")
                            print(self.market)
                        else:
                            continue  # Quantity less than 1, skip
                    else:
                        # No potential sellers
                        continue

    def execute_trade(self, buyer, seller, good, price):
        # Determine the maximum quantity buyer can afford at this price
        if price == 0:
            return  # Prevent division by zero
        max_affordable_quantity = buyer.wealth // price
        # Determine the quantity to trade
        quantity = min(
            buyer.internal_demand[good],
            seller.inventory[good],
            max_affordable_quantity
        )
        quantity = int(quantity)
        if quantity >= 1:
            total_price = price * quantity
            # Execute trade
            buyer.spend(total_price)
            seller.earn(total_price)
            seller.inventory[good] -= quantity
            buyer.inventory[good] = buyer.inventory.get(good, 0) + quantity
            # Update internal demand
            seller.internal_demand[good] += quantity  # Seller's demand decreases
            buyer.internal_demand[good] -= quantity
            # Print transaction details
            print(f"\nTransaction executed between {seller.name} and {buyer.name}:")
            print(f"{seller.name} sold {quantity} units of {good} to {buyer.name} at {price:.2f} per unit.")
            print("Current Market State:")
            print(self.market)

class Good:
    def __init__(self, name, position):
        self.name = name
        self.position = np.array(position)

class Task:
    def __init__(self, name, position, reward, min_ability):
        self.name = name
        self.position = np.array(position)
        self.reward = reward
        self.min_ability = min_ability

class EconomicAgent:
    def __init__(self, name, initial_wealth, position, values_1, luxury_pref=0.2):
        # Basic agent properties
        self.name = name
        self.wealth = initial_wealth
        self.position = np.array(position)
        self.values_1 = values_1
        self.luxury_pref = luxury_pref
        
        # Initialize inventory and demands
        self.inventory = {'good1': 0, 'good2': 0, 'good3': 0}
        self.desired_inventory = {good: 0 for good in ['good1', 'good2', 'good3']}
        self.internal_demand = {good: 0 for good in ['good1', 'good2', 'good3']}
        
        # Movement and collection properties
        self.target = None
        self.collecting = False
        self.collecting_steps = 0
        
        # Initialize skills
        self.skills = {
            'manufacturing': Skill('manufacturing'),
            'technology': Skill('technology'),
            'research': Skill('research'),
            'maintenance': Skill('maintenance'),
            'operations': Skill('operations')
        }
        
        # Job and machine properties
        self.machines = []  # Machines owned by this agent
        self.current_job = None
        self.job_progress = 0
        self.experience_multiplier = 1.0

    
    def earn(self, amount):
        self.wealth += amount

    def spend(self, amount):
        if amount > self.wealth:
            amount = self.wealth
        self.wealth -= amount

    def update_internal_demand(self, goods_list, good_types, scaling_factor=10):
            for good in goods_list:
                # Adjust desired inventory based on the type of good
                priority = scaling_factor * (1 + self.luxury_pref if good_types[good] == 'luxury' else 1)
                self.desired_inventory[good] = self.values_1.get(good, 0) * priority
                self.internal_demand[good] = self.desired_inventory[good] - self.inventory.get(good, 0)

    def perceived_value(self, good, leader_price):
        # Agents perceive the leader's price with some variation
        perception_error = random.uniform(-0.1, 0.1)  # +/- 10% error in perception
        return leader_price * (1 + perception_error)

    def pick_up_good(self, good, quantity=1):
        self.inventory[good] += quantity

    def trade_good(self, other, good, quantity):
        if self.inventory[good] >= quantity:
            self.inventory[good] -= quantity
            other.inventory[good] += quantity

    def move_towards(self, target_position):
        direction = target_position - self.position
        distance = np.linalg.norm(direction)
        if distance > 0:
            direction_x = direction[0] / distance
            direction_y = direction[1] / distance  # Normalize the direction
            self.position[0] += direction_x * 0.1  # Move a fraction towards the target
            self.position[1] += direction_y * 0.1

    def choose_target(self, agents, goods, tasks):
        goods_targets = [(g.position, g.name) for g in goods]
        agent_targets = [(a.position, 'agent') for a in agents if a != self and any(a.inventory.values())]
        task_targets = [(t.position, t.name) for t in tasks if self.abilities.get(t.name, 0) >= t.min_ability]

        # Determine if agent should prioritize tasks due to low wealth
        if self.wealth < 20 or any(self.internal_demand.get(good, 0) > 0 and self.wealth < self.values_1.get(good, 0) for good in self.internal_demand):
            if task_targets:
                self.target = min(task_targets, key=lambda t: np.linalg.norm(t[0] - self.position))
                return

        if goods_targets:
            # Prefer goods over agents and tasks
            self.target = min(goods_targets, key=lambda t: np.linalg.norm(t[0] - self.position))
        elif task_targets:
            self.target = min(task_targets, key=lambda t: np.linalg.norm(t[0] - self.position))
        elif agent_targets:
            self.target = min(agent_targets, key=lambda t: np.linalg.norm(t[0] - self.position))

    def perform_task(self, tasks):
        for task in tasks:
            if np.array_equal(self.position, task.position) and self.abilities.get(task.name, 0) >= task.min_ability:
                self.earn(task.reward)
                print(f"{self.name} performed {task.name} and earned {task.reward} wealth.")

    def buy_price(self, goods):
        buy_price = self.values_1[goods]
        return buy_price 

    def sell_price(self, goods):
        sell_price = self.values_1[goods] + 1
        return sell_price

    def find_nearby_agents(self, agents, radius):
        nearby_agents = []
        for agent in agents:
            if agent != self:
                distance = np.linalg.norm(agent.position - self.position)
                if distance <= radius:
                    nearby_agents.append(agent)
        return nearby_agents
    def can_perform_job(self, job):
        """Check if agent meets skill requirements for a job"""
        return all(self.skills[skill].level >= level 
                  for skill, level in job.required_skills.items())

    def perform_job(self, job):
        """Attempt to perform a job with potential machine assistance"""
        if not self.can_perform_job(job):
            return False

        # Calculate effective skill level including machine boosts
        effective_skills = self.calculate_effective_skills()
        completion_speed = 1.0

        # Check for available machines that can help
        usable_machines = [m for m in self.machines 
                          if not m.in_use and m.can_operate(self)]
        
        # Use machines if available and beneficial
        for machine in usable_machines:
            if job.machine_compatible:
                if machine.can_operate(self):
                    machine.in_use = True
                    machine.consume_resources(self)
                    completion_speed += 0.5
                    self.experience_multiplier += 0.2

        # Progress the job
        self.job_progress += completion_speed
        if self.job_progress >= job.completion_time:
            # Complete the job
            if job.reward > 0:  # Paying job
                self.earn(job.reward)
            else:  # Training job
                self.spend(abs(job.reward))

            # Grant experience
            for skill, exp in job.experience_reward.items():
                if skill in self.skills:
                    self.skills[skill].gain_experience(
                        exp * self.experience_multiplier
                    )

            # Reset job progress and machine use
            self.job_progress = 0
            for machine in self.machines:
                machine.in_use = False
            
            return True

        return False

    def calculate_effective_skills(self):
        """Calculate total skill levels including machine boosts"""
        effective_skills = {name: skill.level 
                          for name, skill in self.skills.items()}
        
        for machine in self.machines:
            if not machine.in_use:
                for skill, boost in machine.skill_boost.items():
                    if skill in effective_skills:
                        effective_skills[skill] += boost
        
        return effective_skills

    def maintain_machines(self):
        """Pay maintenance costs for owned machines"""
        total_maintenance = sum(m.maintenance_cost for m in self.machines)
        self.spend(total_maintenance)

    def __str__(self):
        return f"{self.name} has wealth: {self.wealth:.2f}, inventory: {self.inventory}, at position {self.position}"

class Market:
    def __init__(self):
        self.agents = []
        self.goods = []
        self.tasks = []

    def add_agent(self, agent):
        self.agents.append(agent)

    def add_good(self, good):
        self.goods.append(good)

    def add_task(self, task):
        self.tasks.append(task)

    def __str__(self):
        return "\n".join(str(agent) for agent in self.agents)

class Economy:
    def __init__(self):
        self.market = Market()
        self.good_types = {
            'good1': 'necessary',
            'good2': 'necessary',
            'good3': 'luxury'
        }
        self.jobs = []
        self.machines = []
        self.training_jobs = []
        print("Market Open")
        self.ai = DynamicAI(self.market)
        
    def add_machine(self, machine):
            self.machines.append(machine)
            
    def simulate(self, steps):
        for _ in range(steps):
            self.move_agents()
            self.acquire_goods()
            self.process_jobs()
            self.maintain_machines()
            self.ai.run_market_cycle()
            
    def process_jobs(self):
        for agent in self.market.agents:
            if agent.current_job:
                job_completed = agent.perform_job(agent.current_job)
                if job_completed:
                    agent.current_job = None
            else:
                # Choose new job based on skills and wealth
                available_jobs = self.jobs + self.training_jobs
                suitable_jobs = [j for j in available_jobs 
                               if agent.can_perform_job(j)]
                
                if suitable_jobs:
                    if agent.wealth < 50:  # Low on money, prioritize paying jobs
                        paying_jobs = [j for j in suitable_jobs 
                                     if j.reward > 0]
                        if paying_jobs:
                            agent.current_job = random.choice(paying_jobs)
                    else:  # Can afford training
                        agent.current_job = random.choice(suitable_jobs)
                        
    def maintain_machines(self):
        for agent in self.market.agents:
            agent.maintain_machines()

    def add_agent(self, name, initial_wealth, position, values_1, initial_inventory=None):
        agent = EconomicAgent(name, initial_wealth, position, values_1)
        if initial_inventory:
            agent.inventory.update(initial_inventory)
        self.market.add_agent(agent)

    def add_good(self, name, position, good_type):
        good = Good(name, position)
        self.market.add_good(good)
        self.good_types[name] = good_type  # Assign type (luxury or necessary)

    def add_task(self, name, position, reward, min_ability):
        task = Task(name, position, reward, min_ability)
        self.market.add_task(task)

    def add_job(self, name, position, reward, required_skills, training_job=False):
        job = Job(name, position, reward, required_skills,training_job)
        self.jobs.append(job)

    
    def move_agents(self):
        for agent in self.market.agents:
            agent.choose_target(self.market.agents, self.market.goods, self.jobs)
            if agent.target:
                agent.move_towards(agent.target[0])

    def acquire_goods(self):
        for agent in self.market.agents:
            if agent.target and agent.target[1] != 'agent':
                good = next((g for g in self.market.goods if g.name == agent.target[1] and np.array_equal(g.position, agent.target[0])), None)
                if good:
                    if not agent.collecting:
                        agent.collecting = True
                        agent.collecting_steps = 5  # Reduced steps to collect for quicker simulation
                    else:
                        agent.collecting_steps -= 1
                        if agent.collecting_steps <= 0:
                            agent.pick_up_good(good.name)
                            self.market.goods.remove(good)
                            agent.collecting = False
                            print(f"{agent.name} acquired {good.name}")

    def plot_market(self, ax):
        ax.clear()
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])

        # For demonstration, we'll use simple markers instead of images
        for agent in self.market.agents:
            x, y = agent.position
            size = max(agent.wealth / 10, 5)  # Size proportional to wealth
            if agent.collecting:
                # Shake the agent position a bit
                x += random.uniform(-0.2, 0.2)
                y += random.uniform(-0.2, 0.2)
            ax.plot(x, y, 'bo', markersize=size)
            ax.text(x, y, f'{agent.name}', size=10, zorder=1, color='k')

        for good in self.market.goods:
            x, y = good.position
            ax.plot(x, y, 'rs', markersize=10)
            ax.text(x, y, f'{good.name}', size=10, zorder=1, color='r')

        for task in self.market.tasks:
            x, y = task.position
            ax.plot(x, y, 'g^', markersize=10)
            ax.text(x, y, f'{task.name}', size=10, zorder=1, color='g')

def update(frame, economy, ax):
    economy.simulate(1)
    economy.plot_market(ax)




if __name__ == "__main__":
    economy = Economy()

    # Add agents
    economy.add_agent("Alice", 100, (0, 0), {'good1': 10, 'good2': 5, 'good3': 1})
    economy.add_agent("Bob", 50, (5, 5), {'good1': 2, 'good2': 80, 'good3': 4})
    economy.add_agent("Charlie", 75, (-5, -5), {'good1': 6, 'good2': 3, 'good3': 7})
    print(economy)
    # Function to generate random positions within the environment
    def random_position():
        return (random.uniform(-8, 8), random.uniform(-8, 8))

     # Add goods with classifications
    for _ in range(5):
        economy.add_good('good1', random_position(), 'necessary')
    for _ in range(7):
        economy.add_good('good2', random_position(), 'necessary')
    for _ in range(10):
        economy.add_good('good3', random_position(), 'luxury')

        
    machine1 = Machine("AutoFabricator", (1, 1),
                      skill_boost={'manufacturing': 2, 'technology': 1},
                      resource_consumption={'good1': 2, 'good2': 1})
    
    machine2 = Machine("ResearchAI", (-1, -1),
                      skill_boost={'research': 3, 'technology': 2},
                      resource_consumption={'good2': 2, 'good3': 1})
    
    economy.add_machine(machine1)
    economy.add_machine(machine2)
    
    # Add regular jobs
    economy.add_job('factory_work', (2, 2),
                   reward=20,
                   required_skills={'manufacturing': 1, 'operations': 1})
    
    economy.add_job('research_project', (-2, -2),
                   reward=35,
                   required_skills={'research': 2, 'technology': 1})
    
    # Add training jobs (negative reward means payment required)
    economy.add_job('manufacturing_training', (3, 3),
                   reward=-10,
                   required_skills={'manufacturing': 0},
                   training_job=True)
    
    economy.add_job('tech_workshop', (-3, -3),
                   reward=-15,
                   required_skills={'technology': 0},
                   training_job=True)
 


    print(economy.market)

    fig, ax = plt.subplots(figsize=(12, 8))  # Increased size of the subplot

    ani = FuncAnimation(fig, update, fargs=(economy, ax), interval=10)  # Update every 10 milliseconds
    plt.show()
    print('\nFinal Results of the Simulation')
    print(economy.market)
