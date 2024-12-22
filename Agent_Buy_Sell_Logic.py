import random
import numpy as np

class DynamicAI:
    def __init__(self, market):
        self.market = market

    def update_market_forces(self):
        # Calculate internal demand for each agent and market supply/demand
        total_supply = {'good1': 0, 'good2': 0, 'good3': 0}
        total_demand = {'good1': 0, 'good2': 0, 'good3': 0}

        # Define a scaling factor for desired inventory
        scaling_factor = 10

        # Calculate internal demand for each agent
        for agent in self.market.agents:
            agent.desired_inventory = {}
            agent.internal_demand = {}
            for good in agent.inventory:
                agent.desired_inventory[good] = agent.values_1[good] * scaling_factor
                agent.internal_demand[good] = agent.desired_inventory[good] - agent.inventory[good]

        # Calculate total supply and demand in the market
        for good in total_supply.keys():
            total_supply[good] = sum(max(-agent.internal_demand[good], 0) for agent in self.market.agents)
            total_demand[good] = sum(max(agent.internal_demand[good], 0) for agent in self.market.agents)

        # Calculate price adjustment factors based on market conditions
        self.price_adjustment_factor = {}
        for good in total_supply.keys():
            total = total_demand[good] + total_supply[good]
            if total > 0:
                self.price_adjustment_factor[good] = (total_demand[good] - total_supply[good]) / total
            else:
                self.price_adjustment_factor[good] = 0  # No adjustment if no activity

    def execute_trades(self):
        # Agents set their ask and bid prices based on internal values and market conditions
        for agent in self.market.agents:
            agent.ask_price = {}
            agent.bid_price = {}
            for good in agent.inventory:
                padj = self.price_adjustment_factor[good]
                if agent.internal_demand[good] < 0:
                    # Agent wants to sell
                    agent.ask_price[good] = agent.values_1[good] * (1 + 0.5 * padj)
                elif agent.internal_demand[good] > 0:
                    # Agent wants to buy
                    agent.bid_price[good] = agent.values_1[good] * (1 - 0.5 * padj)

        # Execute trades
        for good in ['good1', 'good2', 'good3']:
            # Sellers for this good
            sellers = [agent for agent in self.market.agents if agent.internal_demand[good] < 0]
            # Buyers for this good
            buyers = [agent for agent in self.market.agents if agent.internal_demand[good] > 0]

            # Sort sellers by lowest ask price
            sellers.sort(key=lambda x: x.ask_price.get(good, float('inf')))
            # Sort buyers by highest bid price
            buyers.sort(key=lambda x: x.bid_price.get(good, 0), reverse=True)

            # Match buyers and sellers
            while sellers and buyers:
                seller = sellers[0]
                buyer = buyers[0]

                ask_price = seller.ask_price[good]
                bid_price = buyer.bid_price[good]

                if bid_price >= ask_price:
                    # Agree on the transaction price (could be midpoint or ask_price)
                    transaction_price = ask_price

                    # Determine the quantity to trade
                    quantity = min(
                        -seller.internal_demand[good],
                        buyer.internal_demand[good],
                        seller.inventory[good]
                    )

                    total_price = transaction_price * quantity

                    if buyer.wealth >= total_price:
                        # Execute trade
                        buyer.spend(total_price)
                        seller.earn(total_price)
                        seller.inventory[good] -= quantity
                        buyer.inventory[good] += quantity

                        # Update internal demand
                        seller.internal_demand[good] += quantity
                        buyer.internal_demand[good] -= quantity

                        # Update inventories
                        if seller.internal_demand[good] >= 0:
                            sellers.pop(0)  # Seller no longer wants to sell this good
                        if buyer.internal_demand[good] <= 0:
                            buyers.pop(0)  # Buyer no longer wants to buy this good
                    else:
                        # Buyer cannot afford, remove buyer
                        buyers.pop(0)
                else:
                    # No agreement on price, break the loop
                    break

    def run_market_cycle(self):
        self.update_market_forces()
        self.execute_trades()
