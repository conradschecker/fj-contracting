use crate::fj_contract::*;
use crate::statistics::{BoxPlot,Histogram};
use itertools::Itertools;
use nalgebra::DVector;

#[derive(Debug)]
/// Wrapper for [`FJContractModel`] for a fixed set of influencers.
pub struct InstanceWrapper {
	pub instance: FJContractModel,
	pub influencer_indicator_vec: Vec<bool>,
	pub influencer_list: Vec<usize>,
	pub verbose_breakpoints: Vec<Vec<(usize,AgentAction,f32,f32)>>,
	pub weighted_reward_influence_vec: DVector<f32>,
	pub baseline_reward: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Solvers for two extremal cases: Optimal contract or no contract at all. 
pub enum SimpleSolver {
	/// The optimal contract for an instance.
	Optimal,
	/// The uncontracted case where no influencer receives a contract.
	Uncontracted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Heuristical solvers that compute approximations.
pub enum HeuristicSolver {
	/// The best contract where only a single agent (influencer) receives a contract.
	BestSingleAgentContract,
	/// A contracting computed using the greedy algorithm for the multiple choice knapsack problem. This solution can be arbitrarily bad.
	GreedyMCKPOnlyKnapsack,
	/// The GreedyMCKP algorithms compares the knapsack solution with the best single agent contract and returns the better.
	GreedyMCKP,
	/// This improved heuristic uses the sorting from MCKP, but makes replacements only when it is increasing the utility.
	GreedyImprovementsMCKPSorting,
}

#[derive(Clone, Copy, Debug)]
/// The result (utility and approximation ratio) for a single solution (contract).
pub struct SolverResult {
	pub utility: f32,
	pub utility_approx_factor: f32,
	pub contract_count: usize,
	pub alpha_sum: f32,
	pub reward_contribution: f32,
}

pub trait InstanceSolver {
	/// Solve the given instance of the FJ contracting problem.
	fn obtain_state(&self, instancewrapper: &InstanceWrapper) -> Vec<(usize, AgentAction, f32, f32)>;
}

impl SolverResult {
	/// Compute the utility and approximation ratio for the (wrapped) instance of, using a given solver.
	pub fn obtain(instancewrapper: &InstanceWrapper, solver: &impl InstanceSolver, optimal_utility: Option<f32>) -> SolverResult {
		// The approximation factor has to be rounded due to numeric instability.
		let state = solver.obtain_state(instancewrapper);
		let utility = instancewrapper.get_utility_from_verbose_state_2(&state);
		let utility_approx_factor = (1000000.0 * optimal_utility.unwrap_or(utility) / utility).round() / 1000000.0;
		let contract_count = Self::contract_count(&state);
		let alpha_sum : f32 = state.iter().map(|(_i,_action,alpha,_rewardcontribution)| alpha).sum();
		let reward_contribution : f32 = state.iter().map(|(_i,_action,_alpha,rewardcontribution)| rewardcontribution).sum();
		Self { utility, utility_approx_factor, contract_count, alpha_sum, reward_contribution }
	}
	
	/// Obtain a [`BoxPlot`] of the utility for a collection of results.
	pub fn utility_boxplot(resultcollection: &[SolverResult]) -> BoxPlot<f32> {
		BoxPlot::from_nonempty_f32_history_vec(resultcollection.iter().map(|result| result.utility).collect())
	}
	
	/// Obtain a [`BoxPlot`] of the approximation ratio for a collection of results.
	pub fn utility_approx_factor_boxplot(resultcollection: &[SolverResult]) -> BoxPlot<f32> {
		BoxPlot::from_nonempty_f32_history_vec(resultcollection.iter().map(|result| result.utility_approx_factor).collect())
	}
	
	/// Obtain a [`Histogram`] of the number of infleuncers that received a contract.
	pub fn contractcount_histogram(resultcollection: &[SolverResult], size: usize) -> Histogram<usize> {
		Histogram::from_nonempty_usize_history_vec(resultcollection.iter().map(|result| result.contract_count).collect(), size)
	}
	
	fn contract_count(state: &Vec<(usize, AgentAction, f32, f32)>) -> usize {
		state.iter().filter(|state| state.1 != AgentAction::Random).count()
	}
}

impl InstanceWrapper {
	/// Create a wrapper for an instance of [`FJContractModel`] and select influencers.
	pub fn new(instance: FJContractModel, influencer_selector: &impl AgentSelector, influencer_count: usize) -> Self {
		let influencer_indicator_vec = influencer_selector.generate_influencer_indicator_vec(&instance, influencer_count);
		let influencer_list = influencer_indicator_vec.iter().enumerate().filter(|(_i,is_agent)| **is_agent).map(|(i,_isagent)| i).collect::<Vec<usize>>();
		let verbose_breakpoints = instance.get_verbose_breakpoints(&influencer_list);
		let weighted_reward_influence_vec = instance.weighted_reward_influence_vec();
		let baseline_reward = instance.baseline_reward(&influencer_indicator_vec);
		Self { instance, influencer_indicator_vec, influencer_list, verbose_breakpoints, weighted_reward_influence_vec, baseline_reward }
	}
	
	/// The number of breakpoints / critical contracts that each influencer has.
	pub fn get_breakpoint_count_vec(&self) -> Vec<usize> {
		self.verbose_breakpoints.iter().map(|bpvec_for_node_i| bpvec_for_node_i.len()).collect()
	}
	
	/// The total number of brakpoints / critical contracts that exist.
	pub fn get_breakpoint_sum_for_influencers(&self) -> usize {
		self.verbose_breakpoints.iter().map(|verbose_bp_vec| verbose_bp_vec.len()).sum::<usize>()
	}

	/// Compute the utility from a (borrowed) verbose state, in which all contracts are borrowed.
	fn get_utility_from_verbose_state(&self, verbose_state: &[&(usize, AgentAction, f32, f32)]) -> f32 {
		let alpha_sum : f32 = verbose_state.iter().map(|(_i,_action,alpha,_rewardcontribution)| alpha).sum();
		let reward_sum : f32 = verbose_state.iter().map(|(_i,_action,_alpha,rewardcontribution)| rewardcontribution).sum();
		return (1. - alpha_sum) * ( self.baseline_reward + reward_sum);
	}
	
	/// Compute the utility from a (borrowed) verbose state, in which contracts are directly contained.
	fn get_utility_from_verbose_state_2(&self, verbose_state: &[(usize, AgentAction, f32, f32)]) -> f32 {
		let alpha_sum : f32 = verbose_state.iter().map(|(_i,_action,alpha,_rewardcontribution)| alpha).sum();
		let reward_sum : f32 = verbose_state.iter().map(|(_i,_action,_alpha,rewardcontribution)| rewardcontribution).sum();
		return (1. - alpha_sum) * ( self.baseline_reward + reward_sum);
	}
	
	/// Flatten the breakpoint collection such that they are in a single (unnested) [`Vec`].
	fn get_verbose_breakpoints_flattened_positives(&self) -> Vec<(usize, AgentAction, f32, f32)> {
		self.verbose_breakpoints.iter()
			.map(
				|bpvec| bpvec.iter()
					.filter(|(_i,_action,alpha,_rewardcontrib)| *alpha > 0.)
			).flatten()
			.cloned()
			.collect::<Vec<_>>()
	}
}

impl InstanceSolver for SimpleSolver {
	/// Solve the given instance of the FJ contracting problem, using a simple solver.
	fn obtain_state(&self, instancewrapper: &InstanceWrapper) -> Vec<(usize, AgentAction, f32, f32)> {
		match &self {
			SimpleSolver::Optimal => instancewrapper.verbose_breakpoints
				.iter()
				.multi_cartesian_product()
				.max_by(|bp_agentlist_x, bp_agentlist_y| instancewrapper.get_utility_from_verbose_state(bp_agentlist_x).total_cmp(&instancewrapper.get_utility_from_verbose_state(bp_agentlist_y)))
			.unwrap()
			.into_iter()
			.cloned()
			.collect(),
			
			SimpleSolver::Uncontracted => instancewrapper.influencer_list
				.iter()
				.map(|i| 
					(*i, AgentAction::Random, 0.0, instancewrapper.instance.initial_opinion(*i, &AgentAction::Random) * instancewrapper.weighted_reward_influence_vec[*i])
				).collect::<Vec<_>>()
		}
	}
}

impl InstanceSolver for HeuristicSolver {
	/// Solve the given instance of the FJ contracting problem, using a heuristical solver.
	fn obtain_state(&self, instancewrapper: &InstanceWrapper) -> Vec<(usize, AgentAction, f32, f32)> {
		match &self {
			HeuristicSolver::GreedyImprovementsMCKPSorting => {
				let verbose_breakpoints_positives_sorted = HeuristicSolver::get_greedy_mckp_ordering_verbose(&instancewrapper, 1.0);
				HeuristicSolver::greedy_local_search_verbose(&instancewrapper, &verbose_breakpoints_positives_sorted)
			},
			HeuristicSolver::GreedyMCKPOnlyKnapsack => {
				let verbose_breakpoints_positives_sorted = HeuristicSolver::get_greedy_mckp_ordering_verbose(&instancewrapper, 0.5);
				HeuristicSolver::knapsack_verbose(&instancewrapper, &verbose_breakpoints_positives_sorted, 0.5)
			},
			HeuristicSolver::BestSingleAgentContract => {
				let breakpoints_flattened_positives = instancewrapper.get_verbose_breakpoints_flattened_positives();
				HeuristicSolver::at_most_one_contract_verbose(&instancewrapper, &breakpoints_flattened_positives)
			},
			HeuristicSolver::GreedyMCKP => {
				let verbose_breakpoints_positives_sorted = HeuristicSolver::get_greedy_mckp_ordering_verbose(&instancewrapper, 0.5);
				let knapsack_state = HeuristicSolver::knapsack_verbose(&instancewrapper, &verbose_breakpoints_positives_sorted, 0.5);
				let knapsack_utility = instancewrapper.get_utility_from_verbose_state_2(&knapsack_state);
				
				let breakpoints_flattened_positives = instancewrapper.get_verbose_breakpoints_flattened_positives();
				let fallback_state = HeuristicSolver::at_most_one_contract_verbose(&instancewrapper, &breakpoints_flattened_positives);
				let fallback_utility = instancewrapper.get_utility_from_verbose_state_2(&fallback_state);
				
				if knapsack_utility > fallback_utility {
					return knapsack_state;
				} else {
					return fallback_state;
				}
			},
		}
	}
}
impl HeuristicSolver {
	fn greedy_local_search_verbose(instancewrapper: &InstanceWrapper, verbose_breakpointlist_positives_sorted: &[(usize, AgentAction, f32, f32)]) -> Vec<(usize, AgentAction, f32, f32)> {
		let mut state = SimpleSolver::Uncontracted.obtain_state(&instancewrapper);
		let mut intermediate_state : Vec<(usize, AgentAction, f32, f32)>;
		
		for breakpoint in verbose_breakpointlist_positives_sorted {
			let i = breakpoint.0;
			intermediate_state = vec![*breakpoint];
			intermediate_state.extend(state.iter().filter(|(j, _, _, _)| *j != i));
			if instancewrapper.get_utility_from_verbose_state_2(&state) < instancewrapper.get_utility_from_verbose_state_2(&intermediate_state) {
				state = intermediate_state;
			}
		}
		return state;
	}
	
	fn knapsack_verbose(instancewrapper: &InstanceWrapper, verbose_breakpointlist_positives_sorted: &[(usize, AgentAction, f32, f32)], max_alpha_sum: f32) -> Vec<(usize, AgentAction, f32, f32)> {
		let mut state = SimpleSolver::Uncontracted.obtain_state(&instancewrapper);
		let mut alpha_sum = 0.0;
		for breakpoint in verbose_breakpointlist_positives_sorted {
			let i = breakpoint.0;
			let alpha = breakpoint.2;
			let current_bp_position = state.iter().position(|(j,_,_,_)| *j == i).unwrap();
			if alpha_sum - state[current_bp_position].2 + alpha > max_alpha_sum {
				break;
			}
			state[current_bp_position] = *breakpoint;
			alpha_sum += alpha;
		}
		return state;
	}
	
	fn at_most_one_contract_verbose(instancewrapper: &InstanceWrapper, verbose_breakpointlist_positives: &[(usize, AgentAction, f32, f32)]) -> Vec<(usize, AgentAction, f32, f32)> {
		
		let mut state = SimpleSolver::Uncontracted.obtain_state(&instancewrapper);
		let rewardcontrib_uncontracted_nodes = instancewrapper.baseline_reward;
		
		let maximum_breakpoint = verbose_breakpointlist_positives.iter().max_by(
			|(_i1,_action1,alpha1,rewardcontrib1), (_i2,_action2,alpha2,rewardcontrib2)| ((1.0 - alpha1) * (rewardcontrib_uncontracted_nodes + rewardcontrib1)).total_cmp(&((1.0 - alpha2) * (rewardcontrib_uncontracted_nodes + rewardcontrib2)))
		);
		
		if let Some((i,action,alpha,rewardcontrib)) = maximum_breakpoint {
			if (1.0 - alpha) * (rewardcontrib_uncontracted_nodes + rewardcontrib) > rewardcontrib_uncontracted_nodes {
				let switch_position = state.iter().position(|(j,_,_,_)| *j == *i).unwrap();
				state[switch_position] = (*i,*action,*alpha,*rewardcontrib);
			}
		}

		return state;
	}

	fn get_greedy_mckp_ordering_verbose(instancewrapper: &InstanceWrapper, max_alpha_sum: f32) -> Vec<(usize, AgentAction, f32, f32)> {
		let mut verbose_breakpoints_with_marginal_slopes: Vec<(usize, AgentAction, f32, f32, f32)> = Vec::new();
		for breakpointlist_i in &instancewrapper.verbose_breakpoints {
			let mut dominated: Vec<(usize, AgentAction, f32, f32)> = Vec::new();
			
			// find dominated breakpoints
			for bp_x in breakpointlist_i.iter().filter(|(_i,_action,alpha,_rewardcontrib)| *alpha > 0.) {
				for bp_y in breakpointlist_i.iter().filter(|bp| *bp != bp_x) {
					let value_x = bp_x.3;
					let value_y = bp_y.3;
					let alpha_x = bp_x.2;
					let alpha_y = bp_y.2;
					
					if value_x > 0. && value_x >= value_y && value_x / alpha_x >= value_y / alpha_y {
						dominated.push(*bp_y);
					} else {
						for bp_z in breakpointlist_i.iter().filter(|bp| *bp != bp_x && *bp != bp_y)  {
							let value_z = bp_z.3;
							let alpha_z = bp_z.2;
							
							if	value_x <= value_y &&
								value_y <= value_z &&
								alpha_x <= alpha_y &&
								alpha_y <= alpha_z && (
									value_x == value_y ||
									alpha_y == alpha_z ||
									(value_y - value_x) / (alpha_y - alpha_x) <= (value_z - value_y) / (alpha_z - alpha_y)
								) {
								dominated.push(*bp_y);
							}
						}
					}
				}
			}
			// extract remaining breakpoints, i.e., only breakpoints that are 
			// 1. not dominated by another breakpoint, and
			// 2. have an alpha that is not higher than the maximum total alpha (since they cannot be taken anyway)
			let mut remaining_breakpoints = breakpointlist_i.clone()
				.extract_if(.., |x| !dominated.contains(&x))
				.filter(|(_i,_action,alpha,_rewardcontrib)| *alpha <= max_alpha_sum)
				.collect::<Vec<_>>();
			
			// sort remaining breakpoints by increasing alpha
			remaining_breakpoints.sort_by(
				|(_i1,_action1,alpha1,_rewardcontrib1), (_i2,_action2,alpha2,_rewardcontrib2)| alpha1.total_cmp(alpha2)
			);
			
			// Add marginal slopes (reward difference / alpha difference with previous breakpoint) for all breakpoints except for the first one.
			let mut verbose_breakpointlist_i_with_marginal_slopes = remaining_breakpoints
				.iter()
				.enumerate()
				.filter(|(_j,(_i,_action,alpha,_rewardcontrib))| *alpha > 0.)
				.map(|(j,(i,action,alpha,rewardcontrib))| 
					if j > 0 {
						let prev_alpha = remaining_breakpoints[j-1].2;
						let prev_rewardcontrib = remaining_breakpoints[j-1].3;
						let marginal_slope = (rewardcontrib - prev_rewardcontrib) / (alpha - prev_alpha);
						(*i, *action, *alpha, *rewardcontrib, marginal_slope) 
					} else { 
						(*i, *action, *alpha, *rewardcontrib, 0.0) 
					})
				.collect::<Vec<(usize, AgentAction, f32, f32, f32)>>();

			// append to the indexed collection of all breakpoints
			verbose_breakpoints_with_marginal_slopes.append(&mut verbose_breakpointlist_i_with_marginal_slopes)
		}
		
		// sort by decreasing marginal slopes
		verbose_breakpoints_with_marginal_slopes.sort_by(
			|(_i1,_action1,_alpha1,_rewardcontrib1,marginalslope1), (_i2,_action2,_alpha2,_rewardcontrib2,marginalslope2)|
				marginalslope2.total_cmp(marginalslope1)
		);

		// return vector (with margin slopes removed from breakpoint tuples)
		verbose_breakpoints_with_marginal_slopes.iter().map(|(i,action,alpha,rewardcontrib,_marginalslope)| (*i,*action,*alpha,*rewardcontrib)).collect::<Vec<_>>()
	}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::friedkin_johnson::FJNetwork;
    use crate::instancegenerator;
    
    fn get_testing_instance() -> FJContractModel {
		let indecisiveness_vec = vec![0.5, 0.25, 0.75];
		let adjacency_vec = vec![false, true, false, true, false, true, false, true, false];
        let network = FJNetwork::try_new_uniform(&indecisiveness_vec, &adjacency_vec).unwrap();
        let value_probability_cost_triples = vec![(0., 0.98, 0.), (1./5., 0.01, 1./32.), (4./5., 0.01, 1./4.)];
         
		let action_set = ActionSet::try_new(&value_probability_cost_triples).unwrap();
        let reward_weights = vec![1., 1., 1.];
        let instance = FJContractModel::try_new(network, reward_weights, vec![action_set.clone(); 3]);
        assert!(instance.is_ok());
        
        return instance.unwrap();
	}
    
    
    fn get_testing_instance_2() -> FJContractModel {
		let indecisiveness_vec = vec![0.5, 0.25, 0.75];
		let adjacency_vec = vec![false, true, false, true, false, true, false, true, false];
        let network = FJNetwork::try_new_uniform(&indecisiveness_vec, &adjacency_vec).unwrap();
        let value_probability_cost_triples = vec![(0.1, 0.94, 0.1), (0.3, 0.044, 0.12), (0.8, 0.016, 0.6)];

		let action_set = ActionSet::try_new(&value_probability_cost_triples).unwrap();
        let reward_weights = vec![1., 1., 1.];
        let instance = FJContractModel::try_new(network, reward_weights, vec![action_set.clone(); 3]);
        assert!(instance.is_ok());
        
        return instance.unwrap();
	}
	
	#[test]
	fn instancetest_2() {
		 // select all nodes as influencers
        let instance = get_testing_instance_2();
        let instancewrapper = InstanceWrapper::new(instance, &instancegenerator::SimpleAgentSelector::Random, 3);
        assert_eq!(instancewrapper.influencer_indicator_vec, [true, true, true], "check influencer indicator vec when selecting everyone.");
        
        let breakpoints_expected = vec![
			vec![(AgentAction::Random, 0.0), (AgentAction::Deterministic(1), 1.0)],
			vec![(AgentAction::Random, 0.0), (AgentAction::Deterministic(1), 1./3.), (AgentAction::Deterministic(2), 0.48)],
			vec![(AgentAction::Random, 0.0)],
		];
		
		let breakpoints = instancewrapper.instance.get_breakpoints(&[true, true, true]);
				
		for (result, expected) in std::iter::zip(breakpoints.iter(), breakpoints_expected.iter()) {
			assert_eq!(result.len(), expected.len());
			for (bp_res, bp_exp) in std::iter::zip(result.iter(), expected.iter()) {
				assert_eq!(bp_res.0, bp_exp.0);
				assert!((bp_res.1 - bp_exp.1).abs() < 1./(2.0_f32.powf(6.0)));
			}
		}
		
		let expected_opt_actions = vec![(0, AgentAction::Random), (1, AgentAction::Deterministic(2)), (2,AgentAction::Random)];
		
		let opt_state_verbose = SimpleSolver::Optimal.obtain_state(&instancewrapper);
		// ... check OPT actions
		let mut opt_actions = opt_state_verbose.iter().map(|(index, action, _alpha, _rewardcontrib)| (*index,*action)).collect::<Vec<_>>();
		opt_actions.sort_by_key(|(index,_action)| *index);
		assert_eq!(opt_actions, expected_opt_actions, "check OPT actions");
		
		let opt_utility = instancewrapper.get_utility_from_verbose_state_2(&opt_state_verbose);
		assert_eq!(opt_utility, 0.8944);
	}
    
    #[test]
    fn instancesolver_test() {

        // select only one infleuncer. Most influention should be the second node.
        let instance = get_testing_instance();
        let instancewrapper = InstanceWrapper::new(instance, &instancegenerator::SimpleAgentSelector::MostInfluential, 1);
        assert_eq!(instancewrapper.influencer_indicator_vec, [false, true, false], "check influencer_indicator_vec when selecting the single most influential agent as influencer");
        
        // select all nodes as influencers
        let instance = get_testing_instance();
        let instancewrapper = InstanceWrapper::new(instance, &instancegenerator::SimpleAgentSelector::Random, 3);
        assert_eq!(instancewrapper.influencer_indicator_vec, [true, true, true], "check influencer_indicator_vec when selecting everyone");
        
        // keep this instance of all nodes are influencers, and ...
        // ...check breakpoint counts
        assert_eq!(instancewrapper.get_breakpoint_count_vec(), [3,3,2], "check breakpoint counts");
        assert_eq!(instancewrapper.get_breakpoint_sum_for_influencers(), 8, "check breakpoint sums");
        
        // ... denote expected breakpoints
		let breakpoints_expected = vec![
			vec![(AgentAction::Random, 0.0), (AgentAction::Deterministic(1), 75./304.), (AgentAction::Deterministic(2), 35./64.)],
			vec![(AgentAction::Random, 0.0), (AgentAction::Deterministic(1), 25./304.), (AgentAction::Deterministic(2), 35./192.)],
			vec![(AgentAction::Random, 0.0), (AgentAction::Deterministic(1), 75./152.)],
		];
		
		// ... check breakpoints
		for breakpointlist in &instancewrapper.verbose_breakpoints {
			for (i,action,_alpha,_rewardcontribution) in breakpointlist {
				assert!(breakpoints_expected[*i].iter().map(|(exp_action, _exp_alpha)| exp_action).collect::<Vec<_>>().contains(&&action));
			}
		}
		
		// ... check sorting without cropping expensive items (MCKP) [alphas need to be removed again]
		let sorted_mckp_bpcollection_positives = HeuristicSolver::get_greedy_mckp_ordering_verbose(&instancewrapper, 1.0).iter().map(|(index, action, _alpha, _rewardcontrib)| (*index,*action)).collect::<Vec<_>>();
		
		let expected_mckp_sorting = vec![(1, AgentAction::Deterministic(2)), (0, AgentAction::Deterministic(2)), (2, AgentAction::Deterministic(1))];
		assert_eq!(sorted_mckp_bpcollection_positives, expected_mckp_sorting, "check greedy MCKP sorting without cropping expensive items");
		
		// ... check sorting when cropping items with alpha > 0.5 (MCKP) [alphas need to be removed again]
		let sorted_mckp_bpcollection_positives = HeuristicSolver::get_greedy_mckp_ordering_verbose(&instancewrapper, 0.5).iter().map(|(index, action, _alpha, _rewardcontrib)| (*index,*action)).collect::<Vec<_>>();
		
		let expected_mckp_sorting = vec![(1, AgentAction::Deterministic(2)), (2, AgentAction::Deterministic(1))];
		assert_eq!(sorted_mckp_bpcollection_positives, expected_mckp_sorting, "check greedy MCKP sorting when cropping items with alpha > 0.5");
		
		// For this instance, all heuristics should equal OPT
		let expected_actions = vec![(0, AgentAction::Random), (1, AgentAction::Deterministic(2)), (2,AgentAction::Random)];
		
		// ... check OPT actions
		let mut opt_actions = SimpleSolver::Optimal.obtain_state(&instancewrapper).iter().map(|(index, action, _alpha, _rewardcontrib)| (*index,*action)).collect::<Vec<_>>();
		opt_actions.sort_by_key(|(index,_action)| *index);
		assert_eq!(opt_actions, expected_actions, "check OPT actions");
		
		// ... check GreedyMCKP actions (this is the full result of the algorithm, which is the best of knapsack packing and singletons
		let mut greedy_mckp_actions = HeuristicSolver::GreedyMCKP.obtain_state(&instancewrapper).iter().map(|(index, action, _alpha, _rewardcontrib)| (*index,*action)).collect::<Vec<_>>();
		greedy_mckp_actions.sort_by_key(|(index,_action)| *index);
		assert_eq!(greedy_mckp_actions, expected_actions, "check GreedyMCKP actions");
		
		
		// ... check MCKP knapsack action (this is the pure output of the knapsack packing algorithm, without checking singletons)
		let mut mckp_knapsack_actions = HeuristicSolver::knapsack_verbose(&instancewrapper, &HeuristicSolver::get_greedy_mckp_ordering_verbose(&instancewrapper, 0.5), 0.5).iter().map(|(index, action, _alpha, _rewardcontrib)| (*index,*action)).collect::<Vec<_>>();
		mckp_knapsack_actions.sort_by_key(|(index,_action)| *index);
		assert_eq!(mckp_knapsack_actions, expected_actions, "check MCKP knapsack output");
		
		// ... check best single contract action
		let mut best_single_contract = HeuristicSolver::BestSingleAgentContract.obtain_state(&instancewrapper).iter().map(|(index, action, _alpha, _rewardcontrib)| (*index,*action)).collect::<Vec<_>>();
		best_single_contract.sort_by_key(|(index,_action)| *index);
		assert_eq!(best_single_contract, expected_actions, "check best single contract actions");
    }
}
