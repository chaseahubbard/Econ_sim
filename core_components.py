from typing import List, Optional, Tuple, Dict
import logging
import random
from config import Config # Import Config

class Grid:
    def __init__(self, width:int, height:int):
        self.width = width
        self.height = height
        self.nodes: List['ResourceNode'] = []
    def add_node(self, node:'ResourceNode'):
        self.nodes.append(node)
    def get_node_at(self, pos: Tuple[int,int]) -> Optional['ResourceNode']:
        for node in self.nodes:
            if node.pos == pos:
                return node
        return None

class ResourceNode:
    def __init__(self, x:int, y:int, resource_type:str, capacity:float, fee_receiver:Optional[int]=None):
        self.pos = (x,y)
        self.resource_type = resource_type
        self.capacity = capacity
        self.fee_receiver = fee_receiver 
    def harvest(self, amount:float) -> float:
        taken = min(amount, self.capacity)
        self.capacity -= taken
        return taken

class TrainingCenter:
    def __init__(self, pos:Tuple[int,int]):
        self.pos = pos

class ToolFactory:
    def __init__(self, pos:Tuple[int,int]):
        self.pos = pos
        self.wood_stock = 0 
        self.wood_tools = 0 
        self.food_tools = 0 
        self.price = 0.0
    def produce(self):
        producible_tools = int(self.wood_stock // Config.WOOD_PER_TOOL)
        if producible_tools > 0:
            new_wood_tools = producible_tools // 2
            new_food_tools = producible_tools - new_wood_tools
            
            self.wood_tools += new_wood_tools
            self.food_tools += new_food_tools
            self.wood_stock -= producible_tools * Config.WOOD_PER_TOOL
            # Assume INITIAL_PRICE_WOOD is accessible via Config
            self.price = Config.WOOD_PER_TOOL * Config.INITIAL_PRICE_WOOD * Config.TOOL_MARKUP 
            logging.info(f"Factory produced {new_wood_tools} wood tools, {new_food_tools} food tools. Stock: W{self.wood_tools}, F{self.food_tools}. Price: {self.price}")

class Store:
    def __init__(self, pos:Tuple[int,int]):
        self.pos = pos
        self.wood_stock = 0
        self.ent_stock = 0
        self.price = 0.0
    def produce(self):
        total = int(self.wood_stock // Config.WOOD_PER_ENT)
        if total>0:
            self.ent_stock += total
            self.wood_stock -= total*Config.WOOD_PER_ENT
            # Assume INITIAL_PRICE_WOOD is accessible via Config
            self.price = Config.WOOD_PER_ENT*Config.INITIAL_PRICE_WOOD * Config.ENT_MARKUP

class Project:
    def __init__(self, pos:Tuple[int,int], resource_type:str, developer_id:int):
        self.pos = pos
        self.resource_type = resource_type
        self.developer_id = developer_id
        self.time_left = Config.PROJECT_TIME

class Loan:
    next_loan_id = 0
    def __init__(self, lender_id: int, borrower_id: int, principal: float, interest_rate: float, issue_tick: int, duration_ticks: int):
        self.id = Loan.next_loan_id
        Loan.next_loan_id += 1
        self.lender_id = lender_id
        self.borrower_id = borrower_id
        self.principal = principal
        self.interest_rate = interest_rate # Simple interest rate for the duration
        self.issue_tick = issue_tick
        self.duration_ticks = duration_ticks
        self.due_tick = issue_tick + duration_ticks
        self.total_due = principal * (1 + interest_rate * duration_ticks) # Total amount to be repaid
        self.amount_repaid = 0.0
        self.is_defaulted = False

    def accrue_interest(self): # Placeholder - simple interest calculated at creation
        pass # In a real compound interest model, interest would accrue here

    def make_payment(self, amount: float) -> float:
        payable = self.get_remaining_owed()
        payment = min(amount, payable)
        self.amount_repaid += payment
        return payment

    def is_fully_repaid(self) -> bool:
        return self.amount_repaid >= self.total_due
        
    def get_remaining_owed(self) -> float:
        return max(0, self.total_due - self.amount_repaid)

    def __repr__(self):
        return f"Loan(ID:{self.id}, L:{self.lender_id}->B:{self.borrower_id}, P:{self.principal:.1f}, R:{self.interest_rate*100:.1f}%, Due:{self.due_tick}, Repaid:{self.amount_repaid:.1f}/{self.total_due:.1f}, Default:{self.is_defaulted})" 