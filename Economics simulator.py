import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class DynamicAI:
    def __init__(self, market):
        self.market = market
        self.goods_list = ['good1', 'good2', 'good3']
        self.trading_radius = 15  # Agents can trade within 15 units of distance

    def run_market_cycle(self):
        # Agents update their internal demand
        self.update_internal_demand()
        # Agents attempt to trade with nearby agents
        self.execute_trades()

    def update_internal_demand(self):
        # Define a scaling factor for desired inventory
        scaling_factor = 10

        # Calculate internal demand for each agent
        for agent in self.market.agents:
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




class Good:
    def __init__(self, name, position):
        self.name = name
        self.position = np.array(position)

class EconomicAgent:
    def __init__(self, name, initial_wealth, position, values_1):
        self.name = name
        self.wealth = initial_wealth
        self.position = np.array(position)
        self.inventory = {'good1': 0, 'good2': 0, 'good3': 0}
        self.values_1 = values_1  # values_1 for goods as a dictionary
        self.target = None  # Current target for movement or acquisition
        self.collecting = False  # Indicates if the agent is collecting a good
        self.collecting_steps = 0  # Counter for collection steps

    def earn(self, amount):
        self.wealth += amount

    def spend(self, amount):
        if amount > self.wealth:
            amount = self.wealth
        self.wealth -= amount

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

    def choose_target(self, agents, goods):
        goods_targets = [(g.position, g.name) for g in goods]
        agent_targets = [(a.position, 'agent') for a in agents if a != self and any(a.inventory.values())]

        if goods_targets:
            # Prefer goods over agents
            self.target = min(goods_targets, key=lambda t: np.linalg.norm(t[0] - self.position))
        elif agent_targets:
            self.target = min(agent_targets, key=lambda t: np.linalg.norm(t[0] - self.position))

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

    def __str__(self):
        return f"{self.name} has wealth: {self.wealth:.2f}, inventory: {self.inventory}, at position {self.position}"

class Market:
    def __init__(self):
        self.agents = []
        self.goods = []

    def add_agent(self, agent):
        self.agents.append(agent)

    def add_good(self, good):
        self.goods.append(good)

    def __str__(self):
        return "\n".join(str(agent) for agent in self.agents)

class Economy:
    def __init__(self):
        self.market = Market()
        print("Market Open")
        self.ai = DynamicAI(self.market)

    def add_agent(self, name, initial_wealth, position, values_1, initial_inventory=None):
        agent = EconomicAgent(name, initial_wealth, position, values_1)
        if initial_inventory:
            agent.inventory.update(initial_inventory)
        self.market.add_agent(agent)


    def add_good(self, name, position):
        good = Good(name, position)
        self.market.add_good(good)

    def simulate(self, steps):
        for _ in range(steps):
            self.move_agents()
            self.acquire_goods()
            self.ai.run_market_cycle()  # Now trading happens after acquiring goods

    def move_agents(self):
        for agent in self.market.agents:
            agent.choose_target(self.market.agents, self.market.goods)
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

def update(frame, economy, ax):
    economy.simulate(1)
    economy.plot_market(ax)

if __name__ == "__main__":
    economy = Economy()
    economy.add_agent("Alice", 100, (0, 0), {'good1': 10, 'good2': 5, 'good3': 1}, {'good1': 15,'good2': 10,'good3': 1})
    economy.add_agent("Bob", 50, (5, 5), {'good1': 2, 'good2': 80, 'good3': 4}, {'good1': 5,'good2': 1,'good3': 7})
    economy.add_agent("Charlie", 75, (-5, -5), {'good1': 6, 'good2': 3, 'good3': 7}, {'good1': 1,'good2': 5,'good3': 13})

    # Function to generate random positions within the environment
    def random_position():
        return (random.uniform(-8, 8), random.uniform(-8, 8))

    # Add 5 units of good1
    for _ in range(5):
        economy.add_good('good1', random_position())

    # Add 7 units of good2
    for _ in range(7):
        economy.add_good('good2', random_position())

    # Add 10 units of good3
    for _ in range(10):
        economy.add_good('good3', random_position())
    print(economy.market)

    fig, ax = plt.subplots(figsize=(12, 8))  # Increased size of the subplot

    ani = FuncAnimation(fig, update, fargs=(economy, ax), interval=10)  # Update every 10 milliseconds
    plt.show()
    print('\nFinal Results of the Simulation')
    print(economy.market)
