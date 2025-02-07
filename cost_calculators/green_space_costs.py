import numpy as np
from values_slr import slr
from values_surge import surge

class GreenSpaceCosts:
    """Cost calculator for green space and floodwall protection system"""
    
    def __init__(self, params):
        """
        Initialize with borough-specific parameters
        
        Args:
            params: Parameter object containing:
                - exposure_value: Value of exposed assets ($/m width)
                - vulnerability_factor: Vulnerability factor
                - slope: Slope of city
                - city_height: Height of city (m)
                - seawall_height: Initial seawall height (m)
                - green_space_params: Dictionary containing:
                    - b_green: Bottom height of green space (m)
                    - t_green: Top height of green space (m)
                - floodwall_params: Dictionary containing:
                    - b2: Bottom height of floodwall (m)
                    - t2: Top height of floodwall (m)
                - discount_factor: Discount factor (default 0.97)
                    Note: Discount factor will be applied at environment level
        """
        self.params = params
        self.discount_factor = getattr(params, 'discount_factor', 0.97)
        
        # Calculate derived values
        self.cf = -self.params.vulnerability_factor * self.params.exposure_value
        self.vol_z = 0.5 * self.params.city_height * (1/self.params.slope) * self.params.city_height
        
        # Seawall height
        self.seawall_h = params.seawall_height
        
        # Green space parameters
        self.b_green = params.green_space_params['b_green']
        self.t_green = params.green_space_params['t_green']
        self.l_green = np.sqrt(((self.t_green - self.b_green)/self.params.slope)**2 + 
                              (self.t_green - self.b_green)**2)  # Length of green space
        
        # Floodwall parameters
        self.b2 = params.floodwall_params['b2']
        self.t2 = params.floodwall_params['t2']
        self.h2 = self.t2 - self.b2  # Height of floodwall

    def calculate_flood_damage(self, state, system_state):
        """
        Calculate flood damage costs based on water level and system state
        
        Args:
            state: tuple (slr_state, surge_state)
            system_state: int (0-3) representing system configuration:
                0: No protection
                1: Only green space
                2: Only floodwall
                3: Both green space and floodwall
        """
        # Convert states to meters
        slr_value = slr(state[0]) * 0.02  # 2cm discretization to meters
        surge_value = surge(state[1]) * 0.1  # 10cm discretization to meters
        total_height = slr_value + surge_value  # meters
            
        # Calculate flooded area based on system state
        if system_state == 0:  # No protection
            # Adjust water level for seawall
            if total_height <= self.seawall_h:
                total_height = 0
            else:
                total_height = total_height - self.seawall_h
                
            # Calculate flooded area
            area = 0.5 * (1/self.params.slope) * total_height**2
            
        elif system_state == 1:  # Only green space
            if total_height <= self.seawall_h:
                total_height = 0
                area = 0
            elif total_height > self.seawall_h and total_height <= self.b_green:
                f_damage = 1.0
                total_height = total_height - self.seawall_h
                area = 0.5 * f_damage * (1/self.params.slope) * total_height**2
            elif total_height > self.b_green and total_height <= self.t_green:
                f_damage1 = 1.00
                a1 = 0.5 * (1/self.params.slope) * (self.b_green - self.seawall_h)**2
                f_damage2 = 0.80
                a2 = 0.5 * (1/self.params.slope) * (total_height - self.seawall_h)**2 - a1
                area = a1 * f_damage1 + a2 * f_damage2
            else:  # total_height > t_green
                f_damage1 = 1.00
                a1 = 0.5 * (1/self.params.slope) * (self.b_green - self.seawall_h)**2
                f_damage2 = 0.80
                a2 = 0.5 * (1/self.params.slope) * (self.t_green - self.seawall_h)**2 - a1
                a3 = 0.5 * (1/self.params.slope) * (total_height - self.seawall_h)**2 - a1 - a2
                area = a1 * f_damage1 + a2 * f_damage2 + a3 * f_damage1
                
        elif system_state == 2:  # Only floodwall
            if total_height <= self.seawall_h:
                total_height = 0
                area = 0
            elif total_height > self.b2 and total_height <= self.t2:
                area = 0.5 * (self.b2 - self.seawall_h) * (self.b2 - self.seawall_h) * (1/self.params.slope) + \
                       (total_height - self.b2) * (self.b2 - self.seawall_h) * (1/self.params.slope)
            else:
                area = 0.5 * (1/self.params.slope) * (total_height - self.seawall_h)**2
            
        else:  # Both green space and floodwall
            if total_height <= self.seawall_h:
                total_height = 0
                area = 0
            elif total_height > self.seawall_h and total_height <= self.b_green:
                f_damage = 1.0
                total_height = total_height - self.seawall_h
                area = 0.5 * f_damage * (1/self.params.slope) * total_height**2
            elif total_height > self.b_green and total_height <= self.t_green:
                f_damage1 = 1.00
                a1 = 0.5 * (1/self.params.slope) * (self.b_green - self.seawall_h)**2
                f_damage2 = 0.80
                a2 = 0.5 * (1/self.params.slope) * (total_height - self.seawall_h)**2 - a1
                area = a1 * f_damage1 + a2 * f_damage2
            elif total_height > self.t_green and total_height <= self.b2:
                f_damage1 = 1.00
                a1 = 0.5 * (1/self.params.slope) * (self.b_green - self.seawall_h)**2
                f_damage2 = 0.80
                a2 = 0.5 * (1/self.params.slope) * (self.t_green - self.seawall_h)**2 - a1
                a3 = 0.5 * (1/self.params.slope) * (total_height - self.seawall_h)**2 - a1 - a2
                area = a1 * f_damage1 + a2 * f_damage2 + a3 * f_damage1
            elif total_height > self.b2 and total_height <= self.t2:
                f_damage1 = 1.00
                a1 = 0.5 * (1/self.params.slope) * (self.b_green - self.seawall_h)**2
                f_damage2 = 0.80
                a2 = 0.5 * (1/self.params.slope) * (self.t_green - self.seawall_h)**2 - a1
                a3 = 0.5 * (1/self.params.slope) * (self.b2 - self.seawall_h)**2 - a1 - a2
                a4 = (total_height - self.b2) * (self.b2 - self.seawall_h) * (1/self.params.slope)
                area = a1 * f_damage1 + a2 * f_damage2 + a3 * f_damage1 + a4 * f_damage1
            else:  # total_height > t2
                f_damage1 = 1.00
                a1 = 0.5 * (1/self.params.slope) * (self.b_green - self.seawall_h)**2
                f_damage2 = 0.80
                a2 = 0.5 * (1/self.params.slope) * (self.t_green - self.seawall_h)**2 - a1
                a3 = 0.5 * (1/self.params.slope) * (self.b2 - self.seawall_h)**2 - a1 - a2
                a4 = 0.5 * (1/self.params.slope) * (total_height - self.seawall_h)**2 - a1 - a2 - a3
                area = a1 * f_damage1 + a2 * f_damage2 + a3 * f_damage1 + a4 * f_damage1

        # Calculate flood damage
        c_flood = self.cf * area / self.vol_z
        c_flood_carbon = self.calculate_flood_carbon_cost(c_flood)
        
        return {
            'monetary': c_flood,
            'carbon': c_flood_carbon  # Will be multiplied by SCC in environment
        }

    def calculate_construction_cost(self, action):
        """
        Calculate construction costs for given action
        
        Args:
            action: int (0-2) representing:
                0: Do nothing
                1: Construct green space
                2: Construct floodwall
        """
        if action == 0:  # Do nothing
            monetary = 0
            carbon = 0
        elif action == 1:  # Construct green space
            cg = -25  # Cost per m²
            monetary = cg * self.l_green
            carbon = self.calculate_construction_carbon_green()
        elif action == 2:  # Construct floodwall
            base_cost_per_meter = -1.38e+04 / 1.5
            monetary = base_cost_per_meter * self.h2
            carbon = self.calculate_construction_carbon_floodwall(self.h2)
            
        return {
            'monetary': monetary,
            'carbon': carbon  # Will be multiplied by SCC in environment
        }

    def calculate_maintenance_cost(self, system_state):
        """Calculate annual maintenance costs based on system state"""
        if system_state == 0:  # No protection
            monetary = 0
            carbon = 0
        elif system_state == 1:  # Only green space
            cm_nbs_l = -2.7  # Maintenance cost per m²
            monetary = cm_nbs_l * self.l_green
            carbon = self.calculate_maintenance_carbon_green()
        elif system_state == 2:  # Only floodwall
            monetary = -100
            carbon = self.calculate_maintenance_carbon_floodwall()
        else:  # Both
            cm_nbs_l = -2.7
            monetary = cm_nbs_l * self.l_green - 100
            carbon = (self.calculate_maintenance_carbon_green() + 
                     self.calculate_maintenance_carbon_floodwall())
            
        return {
            'monetary': monetary,
            'carbon': carbon  # Will be multiplied by SCC in environment
        }

    def calculate_carbon_absorption(self, system_state):
        """Calculate carbon absorption by green infrastructure"""
        if system_state in [1, 3]:  # States with green space
            carbon_up_green = 0.17/1000  # ton CO2 per m²
            return carbon_up_green * self.l_green  # Will be multiplied by SCC in environment
        else:
            return 0

    def calculate_flood_carbon_cost(self, flood_damage):
        """Calculate carbon costs from flood damage"""
        flood_cost_2007 = (flood_damage / 1000000) * 0.77
        fd_ghg = 445.1 * flood_cost_2007
        return fd_ghg  # Will be multiplied by SCC in environment

    def calculate_construction_carbon_green(self):
        """Calculate carbon costs from green space construction"""
        carbon_green_c = 1.6/1000  # ton CO2 per m²
        return -(carbon_green_c) * self.l_green  # Will be multiplied by SCC in environment

    def calculate_construction_carbon_floodwall(self, height):
        """Calculate carbon costs from floodwall construction"""
        cost_2007 = (8000 * height) * 0.89
        cost_million = cost_2007 / 1000000
        return cost_million * 243  # Will be multiplied by SCC in environment

    def calculate_maintenance_carbon_green(self):
        """Calculate carbon costs from green space maintenance"""
        carbon_green_m = 0.09/1000  # ton CO2 per m²
        return -(carbon_green_m) * self.l_green  # Will be multiplied by SCC in environment

    def calculate_maintenance_carbon_floodwall(self):
        """Calculate carbon costs from floodwall maintenance"""
        cost_2007 = 100 * 0.89
        cost_million = cost_2007 / 1000000
        return cost_million * 385  # Will be multiplied by SCC in environment 