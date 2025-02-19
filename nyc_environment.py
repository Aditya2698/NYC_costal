from gym import Env
from gym.spaces import Box, Discrete, MultiDiscrete
import numpy as np
import mat73
from values_slr import slr
from values_surge import surge

from cost_calculators.two_floodwall_costs import TwoFloodwallCosts
from cost_calculators.green_space_costs import GreenSpaceCosts
from cost_calculators.oyster_reef_costs import OysterReefCosts
from cost_calculators.salt_marsh_costs import SaltMarshCosts

class NYCEnvironment(Env):
    """Modified NYC Environment with 5 components and system interactions"""
    
    def __init__(self, params):
        # Basic environment parameters
        self.year = 0
        self.horizon = 39
        self.discount_factor = 0.97
        self.interaction_factors = {
            'higher_to_lower': params.get('higher_to_lower_factor', 0.20),  # i%
            'lower_to_higher': params.get('lower_to_higher_factor', 0.15)   # j%
        }
        
        # State space dimensions
        self.n_slr_states = 77
        self.n_surge_states = 72
        
        # Load transition matrices and SCC
        self.slr_transitions = mat73.loadmat('t_slr_245.mat')['t_slr_avg']
        self.surge_transitions = mat73.loadmat('t_surge.mat')['t_surge']
        self.scc = mat73.loadmat('discounted_sum_scc_7_3.mat')['discounted_sum_scc']
        
        # Initialize components with their cost calculators
        self.components = {
            'bronx': {
                'calculator': TwoFloodwallCosts(params.bronx_params),
                'system_state': 0,  # No protection
                'actions': [],  # Action history
                'costs': []  # Cost history
            },
            'manhattan': {
                'calculator': GreenSpaceCosts(params.manhattan_params),
                'system_state': 0,
                'actions': [],
                'costs': []
            },
            'brooklyn': {
                'calculator': TwoFloodwallCosts(params.brooklyn_params),
                'system_state': 0,
                'actions': [],
                'costs': []
            },
            'queens': {
                'calculator': OysterReefCosts(params.queens_params),
                'system_state': 0,
                'actions': [],
                'costs': []
            },
            'staten_island': {
                'calculator': SaltMarshCosts(params.staten_island_params),
                'system_state': 0,
                'actions': [],
                'costs': []
            }
        }
        
        # System state transition matrices (3 actions, 4 states)
        self.system_trans = {
            0: np.array([[1, 0, 0, 0],   # Do nothing
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]]),
            1: np.array([[0, 1, 0, 0],   # Build F1/Nature-based
                        [0, 1, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 0, 1]]),
            2: np.array([[0, 0, 1, 0],   # Build F2/Floodwall
                        [0, 0, 0, 1],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        }
        
        # Define interaction pairs (Higher slope → Lower slope)
        self.interaction_pairs = [
            ('manhattan', 'brooklyn'),
            ('manhattan', 'bronx'),
            ('brooklyn', 'queens'),
            ('brooklyn', 'bronx'),
            ('brooklyn', 'staten_island')
        ]
        
        # Initialize state and action spaces
        self.action_space = MultiDiscrete([3, 3, 3, 3, 3])  # 3 actions for each component
        
        # Water state space
        self.water_state_space = Box(
            low=np.array([0, 0]),
            high=np.array([self.n_slr_states, self.n_surge_states]),
            dtype=np.int32
        )
        
        # System state space for each component
        self.system_state_space = Discrete(4)  # 4 possible states (0,1,2,3)
        
        # Initial states
        self.initial_water_state = np.array([4, 1])  # Initial SLR and surge states
        self.water_state = self.initial_water_state.copy()

    def step(self, actions):
        """
        Execute one timestep in the environment
        
        Args:
            actions: numpy array of shape (5,) with values in [0,1,2] for each component
            
        Returns:
            observation: dict containing water_state and system_states
            reward: float, total cost for this timestep
            done: bool, whether episode has ended
            info: dict with detailed cost breakdown
        """
        # 1. Sample next water level state
        next_slr, next_surge = self._sample_water_levels()
        self.water_state = np.array([next_slr, next_surge])
        
        # 2. Calculate water levels in meters
        water_levels = (
            slr(self.water_state[0]) * 0.01,  # cm to m
            surge(self.water_state[1]) * 0.01  # cm to m
        )
        
        # 3. Update system states based on actions
        self._update_system_states(actions)
        
        # 4. Calculate costs for each component
        component_costs = self._calculate_component_costs(water_levels)
        
        # 5. Apply system interactions
        modified_costs = self._apply_system_interactions(component_costs, water_levels)
        
        # 6. Apply discounting at component level and calculate total reward
        total_reward = 0
        discounted_costs = {}
        
        for comp_name, costs in modified_costs.items():
            # Apply discount to each monetary cost component separately
            discount_factor = self.discount_factor ** self.year
            
            # Store discounted costs
            discounted_costs[comp_name] = {
                'flood_damage': costs['flood_damage'] * discount_factor,
                'flood_carbon': costs['flood_carbon'],  # Already discounted through SCC
                'construction': costs['construction'] * discount_factor,
                'construction_carbon': costs['construction_carbon'],
                'maintenance': costs['maintenance'] * discount_factor,
                'maintenance_carbon': costs['maintenance_carbon']
            }
            
            if 'carbon_uptake' in costs:
                discounted_costs[comp_name]['carbon_uptake'] = costs['carbon_uptake']
            
            # Calculate total cost for this component by summing all cost components
            component_total = (
                # Monetary costs (discounted)
                discounted_costs[comp_name]['flood_damage'] +
                discounted_costs[comp_name]['construction'] +
                discounted_costs[comp_name]['maintenance'] +
                # Carbon costs (already discounted through SCC)
                discounted_costs[comp_name]['flood_carbon'] +
                discounted_costs[comp_name]['construction_carbon'] +
                discounted_costs[comp_name]['maintenance_carbon']
            )
            
            # Add carbon uptake if applicable
            if 'carbon_uptake' in discounted_costs[comp_name]:
                component_total += discounted_costs[comp_name]['carbon_uptake']
            
            # Add to total reward
            total_reward += component_total
            
            # Store costs in component history
            self.components[comp_name]['costs'].append(discounted_costs[comp_name])
        
        # 7. Check if episode is done
        done = self.year >= self.horizon
        self.year += 1
        
        # 8. Prepare observation and info
        observation = {
            'water_state': self.water_state,
            'system_states': {comp: self.components[comp]['system_state'] 
                             for comp in self.components}
        }
        
        info = {
            'component_costs': discounted_costs,  # Now contains discounted costs
            'year': self.year
        }
        
        return observation, total_reward, done, info

    def _sample_water_levels(self):
        """Sample next SLR and surge states"""
        # Sample SLR transition
        trans_slr = self.slr_transitions[self.year]
        trans_slr_state = trans_slr[int(self.water_state[0])]
        next_slr = np.random.choice(np.array(range(self.n_slr_states)), 
                                  p=trans_slr_state)
        
        # Sample surge transition (stationary)
        trans_surge_state = self.surge_transitions[0][int(self.water_state[1])]
        next_surge = np.random.choice(np.array(range(self.n_surge_states)), 
                                    p=trans_surge_state)
        
        return next_slr, next_surge

    def _calculate_component_costs(self, water_levels):
        """
        Calculate base costs for each component
        
        Args:
            water_levels: Tuple of (slr_value, surge_value) in meters
        """
        component_costs = {}
        for comp_name, comp in self.components.items():
            calculator = comp['calculator']
            system_state = comp['system_state']
            
            # Get flood damage costs
            flood_costs = calculator.calculate_flood_damage(water_levels, system_state)

            # Get construction costs if action was taken
            if len(comp['actions']) > 0:
                last_action = comp['actions'][-1]
                construction_costs = calculator.calculate_construction_cost(
                    last_action)
            else:
                construction_costs = {'monetary': 0, 'carbon': 0}
            
            # Get maintenance costs
            maintenance_costs = calculator.calculate_maintenance_cost(
                system_state)
            
            # Calculate carbon uptake if applicable
            carbon_uptake = (calculator.calculate_carbon_absorption(system_state) 
                           if hasattr(calculator, 'calculate_carbon_absorption') 
                           else 0)
            
            # Store all costs
            costs = {
                'flood_damage': flood_costs['monetary'],
                'flood_carbon': flood_costs['carbon'] * self.scc[self.year],
                'construction': construction_costs['monetary'],
                'construction_carbon': construction_costs['carbon'] * self.scc[self.year],
                'maintenance': maintenance_costs['monetary'],
                'maintenance_carbon': maintenance_costs['carbon'] * self.scc[self.year]
            }
            
            if carbon_uptake != 0:
                costs['carbon_uptake'] = carbon_uptake * self.scc[self.year]
                
            component_costs[comp_name] = costs
            
        return component_costs

    def _apply_system_interactions(self, costs, water_levels):
        """
        Apply lateral flooding effects considering bidirectional flow
        
        Args:
            costs: Dictionary of current costs for each component
            water_levels: Tuple of (slr_value, surge_value) in meters
        """
        modified_costs = costs.copy()
        total_height = water_levels[0] + water_levels[1]
        
        for comp_a, comp_b in self.interaction_pairs:
            # Get critical floodwall parameters for both components
            comp_a_has_wall = self._has_critical_floodwall(comp_a)
            comp_b_has_wall = self._has_critical_floodwall(comp_b)
            
            if not (comp_a_has_wall and comp_b_has_wall):
                # Get floodwall heights for both components
                a_heights = self._get_critical_wall_heights(comp_a)
                b_heights = self._get_critical_wall_heights(comp_b)
                
                # Check if water height is in critical range
                if self._is_in_critical_range(total_height, a_heights):
                    if not comp_b_has_wall:
                        # B affects A (lower to higher)
                        modified_costs[comp_a]['flood_damage'] *= (1 + self.interaction_factors['lower_to_higher'])
                        modified_costs[comp_a]['flood_carbon'] *= (1 + self.interaction_factors['lower_to_higher'])
                        
                if self._is_in_critical_range(total_height, b_heights):
                    if not comp_a_has_wall:
                        # A affects B (higher to lower)
                        modified_costs[comp_b]['flood_damage'] *= (1 + self.interaction_factors['higher_to_lower'])
                        modified_costs[comp_b]['flood_carbon'] *= (1 + self.interaction_factors['higher_to_lower'])
        
        return modified_costs

    def _is_in_critical_range(self, water_height, wall_heights):
        """
        Check if water height is between base and top of critical floodwall
        
        Args:
            water_height: Total water height in meters
            wall_heights: Tuple of (base_height, top_height) in meters
        
        Returns:
            Boolean indicating if water height is in critical range
        """
        base_height, top_height = wall_heights
        return base_height < water_height <= top_height

    def _get_critical_wall_heights(self, comp_name):
        """
        Get base and top heights of critical floodwall for a component
        
        Args:
            comp_name: Name of the component
            
        Returns:
            Tuple of (base_height, top_height) in meters
        """
        calculator = self.components[comp_name]['calculator']
        if comp_name in ['bronx', 'brooklyn']:
            return (calculator.b2, calculator.t2)  # F2 is critical
        else:
            return (calculator.b2, calculator.t2)  # F is critical

    def _has_critical_floodwall(self, comp_name):
        """Check if component has its critical floodwall built"""
        system_state = self.components[comp_name]['system_state']
        
        # For two-floodwall environments (Bronx, Brooklyn)
        if comp_name in ['bronx', 'brooklyn']:
            return system_state in [2, 3]  # Has F2
        # For nature-based solution environments
        else:
            return system_state in [2, 3]  # Has F

    def reset(self):
        """Reset environment to initial state"""
        self.year = 0
        self.water_state = self.initial_water_state.copy()
        
        # Reset all components
        for comp in self.components.values():
            comp['system_state'] = 0
            comp['actions'] = []
            comp['costs'] = []
            
        return {
            'water_state': self.water_state,
            'system_states': {comp: 0 for comp in self.components}
        }

    def _update_system_states(self, actions):
        """Update system states based on actions"""
        for i, comp in enumerate(['bronx', 'manhattan', 'brooklyn', 'queens', 'staten_island']):
            current_system = self.components[comp]['system_state']
            action = actions[i]
            
            # Update system state using transition matrix
            system_vector = np.zeros(4)
            system_vector[current_system] = 1
            new_system_vector = system_vector.dot(self.system_trans[action])
            new_system = np.argmax(new_system_vector)
            
            # Update component info
            self.components[comp]['system_state'] = new_system
            self.components[comp]['actions'].append(action) 