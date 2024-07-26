import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.animation import FuncAnimation

class Good:
    def __init__(self, name, position):
        self.name = name
        self.position = np.array(position)

class EconomicAgent:
    def __init__(self, name, initial_wealth, position, values):
        self.name = name
        self.wealth = initial_wealth
        self.position = np.array(position)
        self.inventory = {'good1': 0, 'good2': 0, 'good3': 0}
        self.values = values  # Values for goods as a dictionary
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

    def __str__(self):
        return f"{self.name} has wealth: {self.wealth}, inventory: {self.inventory}, at position {self.position}"

class Market:
    def __init__(self):
        self.agents = []
        self.goods = []

    def add_agent(self, agent):
        self.agents.append(agent)

    def add_good(self, good):
        self.goods.append(good)

    def simulate_transaction(self):
        if len(self.agents) < 2:
            return
        agent1, agent2 = random.sample(self.agents, 2)

        available_goods = {}
        for inventory in agent1.inventory.keys():
            if agent1.inventory[inventory] > 0:
                available_goods[inventory] = agent1.inventory[inventory]
        
        print(available_goods)
        if len(available_goods) > 1:
            good = random.choice(available_goods.keys())
            quantity = random.randint(1, available_goods[good])

            value_agent1 = agent1.values[good] * quantity
            value_agent2 = agent2.values[good] * quantity
            
            if agent2.wealth >= value_agent1 and agent1.inventory[good] >= quantity:
                agent2.spend(value_agent1)
                agent1.earn(value_agent1)
                agent1.trade_good(agent2, good, quantity)

    def __str__(self):
        return "\n".join(str(agent) for agent in self.agents)

class Economy:
    def __init__(self):
        self.market = Market()

    def add_agent(self, name, initial_wealth, position, values):
        agent = EconomicAgent(name, initial_wealth, position, values)
        self.market.add_agent(agent)

    def add_good(self, name, position):
        good = Good(name, position)
        self.market.add_good(good)

    def simulate(self, steps):
        for _ in range(steps):
            self.market.simulate_transaction()
            self.move_agents()
            self.acquire_goods()

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
                        agent.collecting_steps = 10  # Number of steps to collect the good
                    else:
                        agent.collecting_steps -= 1
                        if agent.collecting_steps <= 0:
                            agent.pick_up_good(good.name)
                            self.market.goods.remove(good)
                            agent.collecting = False

    def plot_market(self, ax):
        ax.clear()
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])

        agent_img = mpimg.imread('C:\Code\SP12463-350x350.png')
        good_img = mpimg.imread('C:\Code\pngtree-good-girl-lettering-line-art-simple-png-image_3774110.jpg')

        for agent in self.market.agents:
            x, y = agent.position
            size = agent.wealth / 10  # Size proportional to wealth
            if agent.collecting:
                # Shake the agent image
                x += random.uniform(-0.2, 0.2)
                y += random.uniform(-0.2, 0.2)
            ax.imshow(agent_img, extent=[x-0.5, x+0.5, y-0.5, y+0.5], aspect='auto')
            ax.text(x, y, f'{agent.name}', size=10, zorder=1, color='k')

        for good in self.market.goods:
            x, y = good.position
            ax.imshow(good_img, extent=[x-0.5, x+0.5, y-0.5, y+0.5], aspect='auto')
            ax.text(x, y, f'{good.name}', size=10, zorder=1, color='r')

def update(frame, economy, ax):
    economy.simulate(1)
    economy.plot_market(ax)

if __name__ == "__main__":
    economy = Economy()
    economy.add_agent("Alice", 100, (0, 0), {'good1': 10, 'good2': 5, 'good3': 1})
    economy.add_agent("Bob", 50, (5, 5), {'good1': 2, 'good2': 8, 'good3': 4})
    economy.add_agent("Charlie", 75, (-5, -5), {'good1': 6, 'good2': 3, 'good3': 7})
    economy.add_good('good1', (3, 3))
    economy.add_good('good2', (-3, -3))
    economy.add_good('good3', (6, -6))
    print(economy.market)


    fig, ax = plt.subplots(figsize=(12, 8))  # Increased size of the subplot
    ani = FuncAnimation(fig, update, fargs=(economy, ax), interval=100)  # Update every 1 seconds
    plt.show()
    print('\nResults of the Simulation')
    print(economy.market)