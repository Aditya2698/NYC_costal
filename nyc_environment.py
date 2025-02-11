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
        self.interaction_factor = params.get('interaction_factor', 0.20)  # i%
        
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
        
        # 2. Update system states based on actions
        self._update_system_states(actions)
        
        # 3. Calculate costs for each component
        component_costs = self._calculate_component_costs()
        
        # 4. Apply system interactions
        modified_costs = self._apply_system_interactions(component_costs)
        
        # 5. Calculate total reward and prepare info
        total_reward = 0
        for comp_name, costs in modified_costs.items():
            # Sum all cost components
            comp_total = (costs['flood_damage'] + costs['flood_carbon'] + 
                         costs['construction'] + costs['construction_carbon'] +
                         costs['maintenance'] + costs['maintenance_carbon'])
            if 'carbon_uptake' in costs:
                comp_total += costs['carbon_uptake']
            
            # Apply discount factor to monetary costs (not carbon, as SCC includes discount)
            comp_total = self._apply_discount(comp_total, costs)
            total_reward += comp_total
            
            # Store costs in component history
            self.components[comp_name]['costs'].append(costs)
        
        # 6. Check if episode is done
        done = self.year >= self.horizon
        self.year += 1
        
        # 7. Prepare observation and info
        observation = {
            'water_state': self.water_state,
            'system_states': {comp: self.components[comp]['system_state'] 
                            for comp in self.components}
        }
        
        info = {
            'component_costs': modified_costs,
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

    def _calculate_component_costs(self):
        """Calculate base costs for each component"""
        slr_value = slr(self.water_state[0])
        surge_value = surge(self.water_state[1])
        total_height = slr_value + surge_value
        
        component_costs = {}
        for comp_name, comp in self.components.items():
            calculator = comp['calculator']
            system_state = comp['system_state']
            
            # Get flood damage costs
            flood_costs = calculator.calculate_flood_damage(
                self.water_state, system_state)
            
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

    def _apply_system_interactions(self, costs):
        """Apply lateral flooding effects"""
        modified_costs = costs.copy()
        
        for high_comp, low_comp in self.interaction_pairs:
            if not self._has_critical_floodwall(high_comp):
                # Increase flood damage of lower component by i%
                flood_increase = self.interaction_factor
                modified_costs[low_comp]['flood_damage'] *= (1 + flood_increase)
                modified_costs[low_comp]['flood_carbon'] *= (1 + flood_increase)
        
        return modified_costs

    def _has_critical_floodwall(self, comp_name):
        """Check if component has its critical floodwall built"""
        system_state = self.components[comp_name]['system_state']
        
        # For two-floodwall environments (Bronx, Brooklyn)
        if comp_name in ['bronx', 'brooklyn']:
            return system_state in [2, 3]  # Has F2
        # For nature-based solution environments
        else:
            return system_state in [2, 3]  # Has F

    def _apply_discount(self, total, costs):
        """Apply discount factor to monetary costs only"""
        # Carbon costs already include discount through SCC
        monetary_costs = (costs['flood_damage'] + costs['construction'] + 
                         costs['maintenance'])
        carbon_costs = (costs['flood_carbon'] + costs['construction_carbon'] + 
                       costs['maintenance_carbon'])
        if 'carbon_uptake' in costs:
            carbon_costs += costs['carbon_uptake']
            
        return (monetary_costs * (self.discount_factor ** self.year) + 
                carbon_costs)

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