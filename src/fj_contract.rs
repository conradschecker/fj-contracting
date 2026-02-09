use crate::friedkin_johnson::FJNetwork;
use nalgebra::DVector;

/// Represent instances of the FJ contracting problem. 
///
/// Each instance consists of a [`FJNetwork`], importance values of the agents/nodes, and [`ActionSet`]s of the agents/nodes.
#[derive(Debug)]
pub struct FJContractModel {
	network: FJNetwork, 
	agent_importance: DVector<f32>, 
	action_sets: Vec<ActionSet>,
}

/// Store possible actions of an agent in the [`FJContractModel`]
///
/// Each action is described an opinion value, a probability, and comes with some costs.
#[derive(Clone,Debug)]
pub struct ActionSet {
	pub values: DVector<f32>,
	pub probabilities: DVector<f32>,
	pub costs: DVector<f32>,
}

/// Represent actions that an agent takes in the [`FJContractModel`]
#[derive(Copy,Clone,Debug,PartialEq)]
pub enum AgentAction {
	/// The agent chooses an initial opinion from his [`ActionSet`]. The index of this opinion is given.
	Deterministic(usize),
	/// The agent randomizes.
	Random
}

/// Trait for generating an [`ActionSet`]
pub trait ActionSetGenerator {
	fn try_generate(&self, param: f32) -> Result<ActionSet, &'static str>;
}

/// Trait for selecting influencers (contractable agents) in an [`FJContractModel`].
pub trait AgentSelector {
	fn generate_influencer_indicator_vec(&self, instance: &FJContractModel, influencer_count: usize) -> Vec<bool>;
}

impl FJContractModel {
	/// Construct an instance of [`FJContractModel`] from a given [`FJNetwork`].
	///
	/// The importance of the agents (for the reward) and their individual [`ActionSet`]s have to be provided.
	///
	/// Fails if the lengths of the vectors and the size of the network are not matching.
	pub fn try_new(network: FJNetwork, agent_importance: Vec<f32>, action_sets: Vec<ActionSet>) -> Result<Self, &'static str> {
		if agent_importance.len() != network.size() || action_sets.len() != network.size() {
			Err("Could not create ContractNetwork because the input lengths of the given vectors were not matching.")
		} else {
			let agent_importance = DVector::from_vec(agent_importance);
			Ok(Self {network, agent_importance, action_sets})
		}
	}
	
	/// Return the size of the underlying [`FJNetwork`].
	pub fn network_size(&self) -> usize {
		self.network.size()
	}
	
	/// Return a [`Vec`] with the out-degrees of the underlying [`FJNetwork`].
	pub fn network_outdegrees(&self) -> Vec<(usize, usize)> {
		self.network.get_outdegrees()
	}

	/// Return the reward for a given [`DVector`] slice of public opinions.
	pub fn reward(&self, public_opinions: &DVector<f32>) -> f32 {
		(self.agent_importance.transpose() * public_opinions)[(0,0)]
	}
	
	/// Return the total reward contribution of all nodes that are **not** influencers.
	///
	/// Influencers are specified with a boolean slice of length n, where n is the number of nodes in the network.
	/// If and only if agent `i` is an influencer, then the slice contains `True` on position `i`.
	pub fn baseline_reward(&self, influencer_indicator_vec: &[bool]) -> f32 {
		self.weighted_reward_influence_vec().iter()
			.enumerate()
			.filter(|(i, _wri)| !influencer_indicator_vec[*i])	// keep non-agent nodes
			.map(|(i, wri)| self.initial_opinion(i, &AgentAction::Random) * wri)
			.sum()
	}
	
	/// Return the value of the intial opinion value of the `i`-th agent for a given [`AgentAction`].
	/// If the action indicates randomization, then the expected initial opinion is returned.
	pub fn initial_opinion(&self, i: usize, agentaction: &AgentAction) -> f32 {
		match agentaction {
			AgentAction::Deterministic(j) => {
				 self.action_sets[i].values[*j]
			},
			AgentAction::Random => {
				self.action_sets[i].expected_belief()
			}
		}
	}
	
	/// Return the vector of initial opinions for a given state.
	pub fn initial_opinion_vec(&self, state: &Vec<&(AgentAction, f32)>) -> DVector<f32> {
		DVector::from_iterator(self.network.size(), state.iter().enumerate().map(|(i, (agentaction, _))| self.initial_opinion(i, agentaction)))
	}
	
	/// Return a vector where entry `i` describes the linear influence on the reward that the initial opinion of agent `i` has.
	pub fn weighted_reward_influence_vec(&self) -> DVector<f32> {
		(self.agent_importance.transpose() * self.network.get_final_mapping()).transpose()
	}
	
	/// Return the utility for a given state.
	pub fn utility(&self, state: &Vec<&(AgentAction, f32)>) -> f32 {
		let alpha_sum = state.iter().map(|(_, alpha)| *alpha).sum::<f32>();
		let reward = self.reward(&self.network.fj_final(&self.initial_opinion_vec(state)));
		return (1.0 - alpha_sum) * reward;
	}

	/// Compute breakpoints (critical contracts) for a given agent `i` and a given [`ActionSet`].
	/// 
	/// A verbose breakpoint is a tuple composed of 
	/// - An [`AgentAction`] that `i` chooses.
	/// - A minimal linear contract (alpha value) that incentivizes this action.
	/// - The value of the action, i.e., the additive contribution to the reward when it is chosen.
	fn compute_verbose_breakpoints(&self, i: usize) -> Vec<(AgentAction, f32, f32)> {
		let action_set = &self.action_sets[i];
		let action_set_size = action_set.values.len();
		let influence = self.weighted_reward_influence_vec()[i];
		
		// Vec for storing all computed breakpoints
		let mut breakpoints: Vec<(AgentAction,f32,f32)> = Vec::with_capacity(action_set_size);

		let max_reward_contrib = influence * action_set.values.iter().max_by(|a,b| a.total_cmp(&b)).unwrap();

		// first breakpoint is at for alpha = 0.0
		let mut cur_cost = 0.0;
		let mut cur_reward_contrib = action_set.expected_belief()*influence;
		breakpoints.push((AgentAction::Random, 0.0, cur_reward_contrib));
	
		let aux_actions = std::iter::zip(action_set.values.iter(), action_set.costs.iter())
			// obtain indices
			.enumerate()
			// compute reward contribution of the actions.
			.map(|(j,(opinion, cost))| (j, opinion*influence, *cost))
			// remove actions that are worse than randomizing
			.filter(|(_j, reward_contrib, cost)| reward_contrib - cost >= cur_reward_contrib)
			// collect to Vec
			.collect::<Vec<(usize,f32,f32)>>();
		
		while cur_reward_contrib <= max_reward_contrib {
			let intersection = aux_actions
				.iter()
				// remove actions that have a lower reward contribution 
				// they have already been considered for being on the upper envelope
				.filter(|(_j, reward_contrib, _cost)| *reward_contrib > cur_reward_contrib)
				// compute intersections (alpha value) with current action on the upper envelope
				.map(|(j, reward_contrib, cost)| (j, reward_contrib, cost, (cost - cur_cost) / (reward_contrib - cur_reward_contrib)))
				// remove action that have alpha > 1.
				.filter(|(_j, _reward_contrib, _cost, alpha)| *alpha <= 1.0)
				// take the action with minimum alpha
				.min_by(|a,b| (a.3).total_cmp(&(b.3)));
			
			match intersection {
				None => { 
					break; 
				},
				Some((j, reward_contrib, cost, alpha)) => {
					cur_cost = *cost;
					cur_reward_contrib = *reward_contrib;
					breakpoints.push((AgentAction::Deterministic(*j), alpha, cur_reward_contrib));
				}
			}
		}
		breakpoints
	}
	
	/// Compute breakpoints (critical contracts) for a set of influencers.
	///
	/// Influencers are specified with a boolean slice of length n, where n is the number of nodes in the network.
	/// If and only if agent `i` is an influencer, then the slice contains `True` on position `i`.
	///
	/// A breakpoint is a tuple composed of 
	/// - An [`AgentAction`] that `i` chooses.
	/// - A minimal linear contract (alpha value) that incentivizes this action.
	///
	/// For each node in the network, breakpoints are stored as a [`Vec`].
	/// Nodes that are not influencers obtain the Random action with alpha = 0.0 as single breakpoint.
	/// 
	/// Returns a [`Vec`] of length n, composed of the breakpoint [`Vec`]s of all agents.
	pub fn get_breakpoints(&self, influencer_indicator_vec: &[bool]) -> Vec<Vec<(AgentAction, f32)>> {
		if influencer_indicator_vec.len() != self.network.size() {
			panic!("The influencer_indicator_vec needs to be a boolean vector of length n.");
		}
		let all_breakpoints = (0..self.network.size())
			.map(|i| {
				if influencer_indicator_vec[i] {
					self.compute_verbose_breakpoints(i).iter().map(|(action,alpha,_reward_contrib)| (*action,*alpha)).collect()
				} else {
					vec![(AgentAction::Random, 0.0)]
				}
			}).collect();
		return all_breakpoints;
	}
	
	/// Compute verbose breakpoints for a list of influencers.
	///
	/// The list should contain agent indices that are influencers.
	///
	/// A verbose breakpoint is a tuple composed of 
	/// - An index `i` of the corresponding agent/influencer.
	/// - An [`AgentAction`] that `i` chooses.
	/// - A minimal linear contract (alpha value) that incentivizes this action.
	/// - The value of the action, i.e., the additive contribution to the reward when it is chosen.
	/// 
	/// For each influencer, a [`Vec`] of verbose breakpoints is computed.
	/// Returns a [`Vec`] that contains a [`Vec`] with verbose breakpoints for each influencer.
	pub fn get_verbose_breakpoints(&self, influencers_list: &[usize]) -> Vec<Vec<(usize, AgentAction, f32, f32)>> {
		influencers_list.iter()
			.map(|i| (i, self.compute_verbose_breakpoints(*i)))
			.map(|(i, breakpoints_i)| breakpoints_i.iter()
				.map(|(action, alpha, reward_contrib)| (*i, *action, *alpha, *reward_contrib))
				.collect::<Vec<(usize, AgentAction, f32, f32)>>()
			).collect::<Vec<Vec<(usize, AgentAction, f32, f32)>>>()
	}
}

impl ActionSet {	
	/// Returns the expected value of the distribution.
	pub fn expected_belief(&self) -> f32 {
		return (self.values.transpose() * &self.probabilities)[(0,0)];
	}

	/// Construct a new ActionSet.
	///
	/// Input is a vector of triples (value, probability, cost) encoding the probability distribution.
	///
	/// If the vector is empty or if the probabilities don't add up to 1.0, an error is returned.
	pub fn try_new(value_probability_cost_triples: &Vec<(f32, f32, f32)>) -> Result<Self, &'static str> {
		let size = value_probability_cost_triples.len();
		if size > 0 {
			let probabilities: DVector<f32> = DVector::from_iterator(size, value_probability_cost_triples.iter().map(|(_,p,_)| *p));
			if (1.0_f32 - probabilities.iter().sum::<f32>()).abs() > 0.000001_f32 { 
				Err("The sum of probabilities is not (sufficiently close to) 1.0")
			} else {
				let values = DVector::from_iterator(size, value_probability_cost_triples.iter().map(|(v,_,_)| *v));
				let costs = DVector::from_iterator(size, value_probability_cost_triples.iter().map(|(_,_,c)| *c));
				Ok(Self{values, probabilities, costs})
			}
		} else {
			Err("No elements are given.")
		}
	}
}


#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    #[test]
    fn belief_distribution_creation() {
		let value_probability_cost_triples = vec![(0.5, 0.125, 0.3), (0.2, 0.75, 0.1), (0.8, 0.125, 0.6)];
		let belief_distribution = ActionSet::try_new(&value_probability_cost_triples);
		assert!(belief_distribution.is_ok());
		let belief_distribution = belief_distribution.unwrap();
		assert_eq!(belief_distribution.expected_belief(), 5./16.);
    }
    
    #[test]
    fn contract_instance_creation_1() {
		let indecisiveness_vec = vec![0.5, 1.0, 0.75];
		let adjacency_vec = vec![false, false, true, false, false, false, true, true, false];
        let network = FJNetwork::try_new_uniform(&indecisiveness_vec, &adjacency_vec).unwrap();
        let value_probability_cost_triples = vec![(0.5, 0.125, 0.3), (0.2, 0.75, 0.1), (0.8, 0.125, 0.6)];
		let belief_distribution = ActionSet::try_new(&value_probability_cost_triples).unwrap();
        let agent_importance = vec![1., 1., 1.];
        let instance = FJContractModel::try_new(network, agent_importance, vec![belief_distribution.clone(); 3]);
        assert!(instance.is_ok());
        let instance = instance.unwrap();
        for (result, expected) in std::iter::zip(instance.weighted_reward_influence_vec().iter(), DVector::from_vec(vec![11./13., 22./13., 6./13.]).iter()) {
			assert!((result - expected).abs() < 1./(2.0_f32.powf(6.0)), "compare {result} with {expected}");
		}
    }

	#[test]
    fn breakpoint_computation() {
		let indecisiveness_vec = vec![0.25, 0.5];
		let influence_weight_vec = vec![0., 1., 1., 0.];
        let network = FJNetwork::try_new(indecisiveness_vec, influence_weight_vec).unwrap(); 
        
        let value_probability_cost_triples = vec![(0., 0.9, 0.), (3./4., 0.1, 1./50.)];
		let belief_distribution = ActionSet::try_new(&value_probability_cost_triples).unwrap();
        let agent_importance = vec![1., 1.];
        let instance = FJContractModel::try_new(network, agent_importance, vec![belief_distribution.clone(); 2]).unwrap();
        
        let verbose_breakpoints_firstplayer = instance.compute_verbose_breakpoints(0);
        let breakpoints_firstplayer_expected = vec![(AgentAction::Random, 0.0), (AgentAction::Deterministic(1), 28./1215.)];
        for (bp_result, bp_expected) in std::iter::zip(verbose_breakpoints_firstplayer.iter(), breakpoints_firstplayer_expected.iter()) {
			assert_eq!(bp_result.0, bp_expected.0);
			assert!((bp_result.1 - bp_expected.1).abs() <  1./(2.0_f32.powf(6.0)));
		}        
    }
    
    #[test]
    fn breakpoint_computation_2() {
		let indecisiveness_vec = vec![0.5, 0.25, 0.75];
		let adjacency_vec = vec![false, true, false, true, false, true, false, true, false];
        let network = FJNetwork::try_new_uniform(&indecisiveness_vec, &adjacency_vec).unwrap();
        let value_probability_cost_triples = vec![(0., 0.99, 0.), (3./5., 0.01, 1./5.)];
         
		let belief_distribution = ActionSet::try_new(&value_probability_cost_triples).unwrap();
        let agent_importance = vec![1., 1., 1.];
        let instance = FJContractModel::try_new(network, agent_importance, vec![belief_distribution.clone(); 3]);
        assert!(instance.is_ok());
        let instance = instance.unwrap();
        
		let breakpoints = instance.get_breakpoints(&[true, true, true]);
        let breakpoints_expected = vec![
			vec![(AgentAction::Random, 0.0), (AgentAction::Deterministic(1), 50./99.)],
			vec![(AgentAction::Random, 0.0), (AgentAction::Deterministic(1), 50./297.)],
			vec![(AgentAction::Random, 0.0)],
		];
		
		for (result, expected) in std::iter::zip(breakpoints.iter(), breakpoints_expected.iter()) {
			assert_eq!(result.len(), expected.len());
			for (bp_res, bp_exp) in std::iter::zip(result.iter(), expected.iter()) {
				assert_eq!(bp_res.0, bp_exp.0);
				assert!((bp_res.1 - bp_exp.1).abs() < 1./(2.0_f32.powf(6.0)));
			}
		}
				
		let verbose_breakpoints = instance.get_verbose_breakpoints(&[0,1,2]);
		let verbose_breakpoints_expected = vec![vec![(0,AgentAction::Random, 0.0, 2./500.),(0,AgentAction::Deterministic(1), 50./99., 2./5.)], vec![(1,AgentAction::Random, 0.0, 6./500.),(1,AgentAction::Deterministic(1), 50./297., 6./5.)], vec![(2,AgentAction::Random, 0.0, 1./500.)]];
		
		for (result, expected) in std::iter::zip(verbose_breakpoints.iter(), verbose_breakpoints_expected.iter()) {
			assert_eq!(result.len(), expected.len());
			for (bp_res, bp_exp) in std::iter::zip(result.iter(), expected.iter()) {
				assert_eq!(bp_res.0, bp_exp.0);
				assert_eq!(bp_res.1, bp_exp.1);
				assert!((bp_res.2 - bp_exp.2).abs() < 1./(2.0_f32.powf(6.0)));
				assert!((bp_res.3 - bp_exp.3).abs() < 1./(2.0_f32.powf(6.0)));
			}
		}
    }
    
    #[test]
    fn breakpoint_computation_3() {
		let indecisiveness_vec = vec![0.5, 0.25, 0.75];
		let adjacency_vec = vec![false, true, false, true, false, true, false, true, false];
        let network = FJNetwork::try_new_uniform(&indecisiveness_vec, &adjacency_vec).unwrap();
        let value_probability_cost_triples = vec![(0., 0.98, 0.), (1./5., 0.01, 1./32.), (4./5., 0.01, 1./4.)];
         
		let belief_distribution = ActionSet::try_new(&value_probability_cost_triples).unwrap();
        let agent_importance = vec![1., 1., 1.];
        let instance = FJContractModel::try_new(network, agent_importance, vec![belief_distribution.clone(); 3]);
        assert!(instance.is_ok());
        let instance = instance.unwrap();
        
        // weighted reward influence
        let wri_vec = instance.weighted_reward_influence_vec();
        for (wri_res, wri_expected) in std::iter::zip(wri_vec.iter(), vec![2./3., 2., 1./3.].iter()) {
			assert!((wri_res - wri_expected).abs() < 1./(2.0_f32.powf(6.0)));
		}
        
		let breakpoints = instance.get_breakpoints(&[true, true, true]);
        let breakpoints_expected = vec![
			vec![(AgentAction::Random, 0.0), (AgentAction::Deterministic(1), 75./304.), (AgentAction::Deterministic(2), 35./64.)],
			vec![(AgentAction::Random, 0.0), (AgentAction::Deterministic(1), 25./304.), (AgentAction::Deterministic(2), 35./192.)],
			vec![(AgentAction::Random, 0.0), (AgentAction::Deterministic(1), 75./152.)],
		];
		
		// breakpoints
		for (result, expected) in std::iter::zip(breakpoints.iter(), breakpoints_expected.iter()) {
			assert_eq!(result.len(), expected.len());
			for (bp_res, bp_exp) in std::iter::zip(result.iter(), expected.iter()) {
				println!("bp computed: ({:?},{:.4}). bbp expected: ({:?},{:.4})", bp_res.0, bp_res.1, bp_exp.0, bp_exp.1);
				assert_eq!(bp_res.0, bp_exp.0);
				assert!((bp_res.1 - bp_exp.1).abs() < 1./(2.0_f32.powf(6.0)));
			}
		}
    }
}

