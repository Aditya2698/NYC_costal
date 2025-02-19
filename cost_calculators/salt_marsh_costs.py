import numpy as np
from values_slr import slr
from values_surge import surge

def calculate_wave_attenuation(total_height):
    """
    Calculate wave attenuation factor for salt marsh based on water depth
    
    Args:
        total_height: Total water depth in meters
        
    Returns:
        Wave attenuation factor (0-1)
    """
    if total_height <= 0.1:
        return 0.60  # 60% attenuation for depth < 0.1m
    else:
        return (6.3398 * (total_height)**(-0.974)) / 100  # Convert percentage to fraction

class SaltMarshCosts:
    """Cost calculator for salt marsh and floodwall protection system"""
    
    def __init__(self, params):
        """
        Initialize with borough-specific parameters
        
        Args:
            params: Parameter object containing:
                - exposure_value: Value of exposed assets ($/m width)
                - vulnerability_factor: Vulnerability factor
                - slope: Slope of city
                - city_height: Height of city (m)
                - salt_marsh_params: Dictionary containing:
                    - width: Width of salt marsh (m)
                - floodwall_params: Dictionary containing:
                    - b2: Bottom height of floodwall (m)
                    - t2: Top height of floodwall (m)
                - discount_factor: Discount factor (default 0.97)
                    Note: Discount factor will be applied at environment level
        """
        self.params = params
        
        # Calculate derived values
        self.cf = -self.params.vulnerability_factor * self.params.exposure_value
        self.vol_z = 0.5 * self.params.city_height * (1/self.params.slope) * self.params.city_height
        
        # Salt marsh parameters
        self.marsh_width = params.salt_marsh_params['width']  # Width before wedge starts
        
        # Floodwall parameters
        self.b2 = params.floodwall_params['b2']  # Bottom height (1.8m in MATLAB)
        self.t2 = params.floodwall_params['t2']  # Top height (3.3m in MATLAB)
        self.h2 = self.t2 - self.b2  # Height of floodwall

    def calculate_flood_damage(self, water_levels, system_state):
        """
        Calculate flood damage costs based on water level and system state
        
        Args:
            water_levels: tuple (slr_value, surge_value) in meters
            system_state: int (0-3) representing system configuration:
                0: No protection
                1: Only salt marsh
                2: Only floodwall
                3: Both salt marsh and floodwall
        """
        slr_value, surge_value = water_levels
        total_height = slr_value + surge_value
        
        # Calculate flooded area based on system state
        if system_state == 0:  # No protection
            area = 0.5 * (1/self.params.slope) * total_height**2
            
        elif system_state == 1:  # Only salt marsh
            # Apply wave attenuation
            wa_factor = calculate_wave_attenuation(total_height)
            height_reduction = wa_factor * total_height
            # Cap reduction at surge height
            if height_reduction >= surge_value:
                height_reduction = surge_value
            reduced_height = total_height - height_reduction
            area = 0.5 * (1/self.params.slope) * reduced_height**2
            
        elif system_state == 2:  # Only floodwall
            if total_height > self.b2 and total_height <= self.t2:
                area = 0.5 * self.b2 * self.b2 * (1/self.params.slope) + \
                       (total_height - self.b2) * self.b2 * (1/self.params.slope)
            else:
                area = 0.5 * (1/self.params.slope) * total_height**2
                
        else:  # Both salt marsh and floodwall
            # Apply wave attenuation first
            wa_factor = calculate_wave_attenuation(total_height)
            height_reduction = wa_factor * total_height
            # Cap reduction at surge height
            if height_reduction >= surge_value:
                height_reduction = surge_value
            reduced_height = total_height - height_reduction
            
            # Then apply floodwall protection
            if reduced_height > self.b2 and reduced_height <= self.t2:
                area = 0.5 * self.b2 * self.b2 * (1/self.params.slope) + \
                       (reduced_height - self.b2) * self.b2 * (1/self.params.slope)
            else:
                area = 0.5 * (1/self.params.slope) * reduced_height**2

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
                1: Construct salt marsh
                2: Construct floodwall
        """
        if action == 0:  # Do nothing
            monetary = 0
            carbon = 0
        elif action == 1:  # Construct salt marsh
            cc_marsh = -1000  # Cost per m
            monetary = cc_marsh * self.marsh_width
            carbon = self.calculate_construction_carbon_marsh()
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
        elif system_state == 1:  # Only salt marsh
            monetary = 0  # No maintenance cost for salt marsh
            carbon = self.calculate_maintenance_carbon_marsh()
        elif system_state == 2:  # Only floodwall
            monetary = -100
            carbon = self.calculate_maintenance_carbon_floodwall()
        else:  # Both
            monetary = -100  # Only floodwall maintenance
            carbon = (self.calculate_maintenance_carbon_marsh() + 
                     self.calculate_maintenance_carbon_floodwall())
            
        return {
            'monetary': monetary,
            'carbon': carbon  # Will be multiplied by SCC in environment
        }

    def calculate_carbon_absorption(self, system_state):
        """Calculate carbon absorption by salt marsh"""
        if system_state in [1, 3]:  # States with salt marsh
            carbon_up_marsh = 0.44/1000  # ton CO2 per m²
            return carbon_up_marsh * self.marsh_width  # Will be multiplied by SCC in environment
        return 0

    def calculate_flood_carbon_cost(self, flood_damage):
        """Calculate carbon costs from flood damage"""
        flood_cost_2007 = (flood_damage / 1000000) * 0.77
        fd_ghg = 445.1 * flood_cost_2007
        return fd_ghg  # Will be multiplied by SCC in environment

    def calculate_construction_carbon_marsh(self):
        """Calculate carbon costs from salt marsh construction"""
        carbon_marsh_c = 1.0/1000  # ton CO2 per m²
        return -(carbon_marsh_c) * self.marsh_width  # Will be multiplied by SCC in environment

    def calculate_construction_carbon_floodwall(self, height):
        """Calculate carbon costs from floodwall construction"""
        cost_2007 = (8000 * height) * 0.89
        cost_million = cost_2007 / 1000000
        return cost_million * 243  # Will be multiplied by SCC in environment

    def calculate_maintenance_carbon_marsh(self):
        """Calculate carbon costs from salt marsh maintenance"""
        carbon_marsh_m = 0.0  # No maintenance carbon cost
        return carbon_marsh_m

    def calculate_maintenance_carbon_floodwall(self):
        """Calculate carbon costs from floodwall maintenance"""
        cost_2007 = 100 * 0.89
        cost_million = cost_2007 / 1000000
        return cost_million * 385  # Will be multiplied by SCC in environment 