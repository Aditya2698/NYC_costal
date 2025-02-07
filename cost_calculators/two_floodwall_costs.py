import numpy as np
from values_slr import slr
from values_surge import surge

class TwoFloodwallCosts:
    """Cost calculator for two floodwall protection system"""
    
    def __init__(self, params):
        """
        Initialize with borough-specific parameters
        
        Args:
            params: Parameter object containing:
                - exposure_value: Value of exposed assets ($/m width)
                - vulnerability_factor: Vulnerability factor
                - slope: Slope of city
                - city_height: Height of city (m)
                - floodwall_params: Dictionary containing:
                    - b1: Bottom height of first floodwall (m)
                    - t1: Top height of first floodwall (m)
                    - b2: Bottom height of second floodwall (m)
                    - t2: Top height of second floodwall (m)
                - discount_factor: Discount factor (default 0.97)
                    Note: Discount factor will be applied at environment level
        """
        self.params = params
        
        # Calculate derived values
        self.cf = -self.params.vulnerability_factor * self.params.exposure_value
        self.vol_z = 0.5 * self.params.city_height * (1/self.params.slope) * self.params.city_height
        
        # Floodwall parameters and heights
        self.b1 = params.floodwall_params['b1']
        self.t1 = params.floodwall_params['t1']
        self.b2 = params.floodwall_params['b2']
        self.t2 = params.floodwall_params['t2']
        
        # Calculate floodwall heights
        self.h1 = self.t1 - self.b1  # Height of first floodwall
        self.h2 = self.t2 - self.b2  # Height of second floodwall

    def calculate_flood_damage(self, state, system_state):
        """
        Calculate flood damage costs based on water level and system state
        
        Args:
            state: tuple (slr_state, surge_state)
            system_state: int (0-3) representing system configuration:
                0: No floodwalls
                1: Only lower floodwall (F1)
                2: Only higher floodwall (F2)
                3: Both floodwalls (F1, F2)
        
        Returns:
            Dictionary containing monetary and carbon costs
        """
        # Convert states to meters
        slr_value = slr(state[0]) * 0.02  # 2cm discretization to meters
        surge_value = surge(state[1]) * 0.1  # 10cm discretization to meters
        total_height = slr_value + surge_value
        
        # Calculate flooded volume based on system state
        if system_state == 0:  # No floodwalls
            vol_f = 0.5 * (1/self.params.slope) * total_height**2
            
        elif system_state == 1:  # Only F1
            if (total_height > self.b1) and (total_height <= self.t1):
                area = 0.5 * self.b1 * self.b1 * (1/self.params.slope) + \
                       (total_height - self.b1) * self.b1 * (1/self.params.slope)
            else:
                area = 0.5 * (1/self.params.slope) * total_height**2
            vol_f = area
            
        elif system_state == 2:  # Only F2
            if (total_height > self.b2) and (total_height <= self.t2):
                area = 0.5 * self.b2 * self.b2 * (1/self.params.slope) + \
                       (total_height - self.b2) * self.b2 * (1/self.params.slope)
            else:
                area = 0.5 * (1/self.params.slope) * total_height**2
            vol_f = area
            
        else:  # Both floodwalls (system state 3)
            if total_height <= self.b1:
                area = 0.5 * (1/self.params.slope) * total_height**2
            elif total_height > self.b1 and total_height <= self.t1:
                area = 0.5 * (1/self.params.slope) * self.b1**2 + \
                       (total_height - self.b1) * self.b1 * (1/self.params.slope)
            elif total_height > self.t1 and total_height <= self.b2:
                area = 0.5 * (1/self.params.slope) * total_height**2
            elif total_height > self.b2 and total_height <= self.t2:
                area = 0.5 * (1/self.params.slope) * self.b2**2 + \
                       (total_height - self.b2) * self.b2 * (1/self.params.slope)
            else:
                area = 0.5 * (1/self.params.slope) * total_height**2
            vol_f = area

        # Calculate monetary flood damage
        c_flood = self.cf * vol_f / self.vol_z
        
        # Calculate carbon costs from flood damage
        c_flood_carbon = self.calculate_flood_carbon_cost(c_flood)
        
        return {
            'monetary': c_flood,
            'carbon': c_flood_carbon
        }

    def calculate_construction_cost(self, action):
        """
        Calculate construction costs for given action
        
        Args:
            action: int (0-2) representing:
                0: Do nothing
                1: Construct F1
                2: Construct F2
        """
        base_cost_per_meter = -1.38e+04 / 1.5  # Base cost for 1.5m height
        
        if action == 0:  # Do nothing
            monetary = 0
            carbon = 0
        elif action == 1:  # Construct F1
            monetary = base_cost_per_meter * self.h1
            carbon = self.calculate_construction_carbon(self.h1)
        elif action == 2:  # Construct F2
            monetary = base_cost_per_meter * self.h2
            carbon = self.calculate_construction_carbon(self.h2)
            
        return {
            'monetary': monetary,
            'carbon': carbon
        }

    def calculate_maintenance_cost(self, system_state):
        """Calculate annual maintenance costs based on system state"""
        if system_state == 0:
            monetary = 0
            carbon = 0
        else:  # Single floodwall (state 1 or 2)
            monetary = -100
            carbon = self.calculate_maintenance_carbon()
            
        return {
            'monetary': monetary,
            'carbon': carbon
        }

    def calculate_flood_carbon_cost(self, flood_damage):
        """Calculate carbon costs from flood damage"""
        flood_cost_2007 = (flood_damage / 1000000) * 0.77
        fd_ghg = 445.1 * flood_cost_2007
        return fd_ghg  # Will be multiplied by SCC later

    def calculate_construction_carbon(self, height):
        """Calculate carbon costs from construction"""
        cost_2007 = (8000 * height) * 0.89
        cost_million = cost_2007 / 1000000
        return cost_million * 243  # Will be multiplied by SCC later

    def calculate_maintenance_carbon(self):
        """Calculate carbon costs from maintenance"""
        cost_2007 = 100 * 0.89
        cost_million = cost_2007 / 1000000
        return cost_million * 385  # Will be multiplied by SCC later 