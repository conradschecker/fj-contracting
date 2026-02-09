use crate::friedkin_johnson::FJNetwork;
use crate::fj_contract::*;
use nalgebra::DVector;
use rand::seq::SliceRandom;
use rand::prelude::*;

#[derive(Debug)]
/// Methods for selecting certain agents as influencers.
pub enum SimpleAgentSelector {
	/// Selecting a subset of agents uniformly at random.
	Random,
	/// Selecting a subset of agents by their (out) degree.
	HighestOutDegree,
	/// Selecting a subset of agents by their linear influence of the reward.
	MostInfluential,
}
impl std::str::FromStr for SimpleAgentSelector {
	type Err = &'static str;
	fn from_str(input: &str) -> Result<SimpleAgentSelector, Self::Err> {
		match input {
			"Random"			=> Ok(SimpleAgentSelector::Random),
			"HighestOutDegree"	=> Ok(SimpleAgentSelector::HighestOutDegree),
			"MostInfluential"	=> Ok(SimpleAgentSelector::MostInfluential),
			_ 					=> Err("Unknown RewardType.")
		}
	}
}

impl AgentSelector for SimpleAgentSelector {
	/// Obtain influencers for a given instance and a given number of influencers with simple selecting methods.
	///
	/// Return type is a boolean [`Vec`] of length `n`, where entry `i` is `true` if and only if `i` is an influencer.
	/// `n` is the number of agents / nodes in the given network.
	fn generate_influencer_indicator_vec(&self, instance: &FJContractModel, influencer_count: usize) -> Vec<bool> {
		let network_size = instance.network_size();
		match &self {
			SimpleAgentSelector::Random => {
				// all but "influencer_count" many entries are "false"
				let mut influencer_indicator_vec = vec![false; network_size - influencer_count];
				// extend the vector by "influencer_count" entries that are "true"
				influencer_indicator_vec.extend(vec![true; influencer_count]);
				// shuffle the vector
				influencer_indicator_vec.shuffle(&mut rand::rng());
				influencer_indicator_vec
			},
			SimpleAgentSelector::HighestOutDegree => {
				// initialize the vector with "false".
				let mut influencer_indicator_vec = vec![false; network_size];
				
				// get (node id, outdegree) pairs from network and sort them by outdegree
				let mut outdegrees: Vec<(usize,usize)> = instance.network_outdegrees();
				outdegrees.sort_by(|(_,degree1), (_,degree2)| degree2.cmp(degree1));
				
				// choose nodes with highest outdegrees
				for agent in outdegrees.iter().map(|(node_id,_)| *node_id).take(influencer_count) {
					influencer_indicator_vec[agent] = true;
				}
				influencer_indicator_vec
			},
			SimpleAgentSelector::MostInfluential => {
				// initialize the vector with "false".
				let mut influencer_indicator_vec = vec![false; network_size];
				
				// get DVector containing the weighted reward influence for each node
				let weighted_reward_influence_dvector = instance.weighted_reward_influence_vec();
				
				// produce (node id, influence) tuples and sort by influence
				let mut weighted_reward_influence: Vec<(usize,&f32)> = weighted_reward_influence_dvector.into_iter().enumerate().collect();
				weighted_reward_influence.sort_by(|(_,influence1), (_,influence2)| influence2.total_cmp(influence1));
				
				// choose nodes with highest outdegrees
				for agent in weighted_reward_influence.iter().map(|(node_id,_)| *node_id).take(influencer_count) {
					influencer_indicator_vec[agent] = true;
				}
				influencer_indicator_vec
			}
		}
	}
}

#[derive(Debug)]
/// Types of functions that incorporate the importance (weight) of agents on the reward.
pub enum RewardType {
	Unweighted
}
impl std::str::FromStr for RewardType {
	type Err = &'static str;
	fn from_str(input: &str) -> Result<RewardType, Self::Err> {
		match input {
			"Unweighted"	=> Ok(RewardType::Unweighted),
			_ 				=> Err("Unknown RewardType.")
		}
	}
}


#[derive(Debug)]
/// Methods of generating random action sets.
pub enum SimpleActionSetGenerator {
	DiscreteUniform(usize),			// parameter: support size
	Geometric(usize),				// parameter: support size
	GeometricFirstvalueZero(usize),	// parameter: support size
	BinaryBeta(f32),				// parameter: beta
	BinaryP0(f32),					// parameter: p0
	BinaryP0Proportional,
}

impl ActionSetGenerator for SimpleActionSetGenerator {
	fn try_generate(&self, network_size: f32) -> Result<ActionSet, &'static str> {
		match &self {
			SimpleActionSetGenerator::DiscreteUniform(action_set_size)			=> SimpleActionSetGenerator::try_generate_random_discrete_uniform(*action_set_size),
			SimpleActionSetGenerator::Geometric(action_set_size)					=> SimpleActionSetGenerator::try_generate_random_geometric(*action_set_size),
			SimpleActionSetGenerator::GeometricFirstvalueZero(action_set_size)	=> SimpleActionSetGenerator::try_generate_random_geometric_firstvalue0(*action_set_size),
			SimpleActionSetGenerator::BinaryBeta(beta) 						=> SimpleActionSetGenerator::try_generate_random_binary_beta(*beta),
			SimpleActionSetGenerator::BinaryP0(p0) 							=> SimpleActionSetGenerator::try_generate_random_binary_p0(*p0),
			SimpleActionSetGenerator::BinaryP0Proportional 					=> SimpleActionSetGenerator::try_generate_random_binary_p0(1. - 1. / network_size),
		}
	}
}

impl SimpleActionSetGenerator {
		pub fn from_str(input_str: &str, action_set_size: Option<usize>, beta: Option<f32>, p0: Option<f32>) -> Option<SimpleActionSetGenerator> {
		match input_str {
			"DiscreteUniform"			=> Some(SimpleActionSetGenerator::DiscreteUniform(action_set_size.expect("Parameter action_set_size needed"))),
			"Geometric"					=> Some(SimpleActionSetGenerator::Geometric(action_set_size.expect("Parameter action_set_size needed"))),
			"GeometricFirstvalueZero"	=> Some(SimpleActionSetGenerator::GeometricFirstvalueZero(action_set_size.expect("Parameter action_set_size needed"))),
			"BinaryBeta"				=> Some(SimpleActionSetGenerator::BinaryBeta(beta.expect("Parameter beta needed"))),
			"BinaryP0"					=> Some(SimpleActionSetGenerator::BinaryP0(p0.expect("Parameter p0 needed"))),
			"BinaryP0Proportional"		=> Some(SimpleActionSetGenerator::BinaryP0Proportional),
			_ => None
		}
	}

	/// Generate a binaryBeta action set.
	///
	/// - First action has value 0, cost 0, and random probability p0 ~ unif([0.5, 1)).
	/// - Second action as random value s1 ~ unif[0,1), cost beta * s1 for given parameter beta, and probability 1 - p0.
	fn try_generate_random_binary_beta(beta: f32) -> Result<ActionSet, &'static str> {
		let mut rng = rand::rng();
		//let p0 = rng.random::<f32>();				// unif [0,1)
		let p0 = rng.random::<f32>() * 0.5 + 0.5;	// unif [0.5,1)
		let s1 = rng.random::<f32>();
		let vpc_triples = vec![(0.0, p0, 0.0), (s1, 1.0 - p0, beta * s1)];
		
		ActionSet::try_new(&vpc_triples)
	}
	
	/// Generate a binaryP0 action set.
	///
	/// - First action has value 0, cost 0, and given probability p0.
	/// - Second action as random value (unif[0,1)), random cost (unif[0,1) * unif[0,1)), and probability 1 - p0.
	fn try_generate_random_binary_p0(p0: f32) -> Result<ActionSet, &'static str> {
		let mut rng = rand::rng();
		let beta = rng.random::<f32>();
		let s1 = rng.random::<f32>();
		let vpc_triples = vec![(0.0, p0, 0.0), (s1, 1.0 - p0, beta * s1)];
		
		ActionSet::try_new(&vpc_triples)
	}

	/// Generate a discrete uniform distribution over uniform-[0,1) values and costs.
	///
	/// Precisely:
	/// - Values are drawn independently from a continous uniform distribution with support [0,1).
	/// - Costs are drawn independently from a continous uniform distribution with support [0,1).
	/// - Probabilities are 1 / action_set_size.
	fn try_generate_random_discrete_uniform(action_set_size: usize) -> Result<ActionSet, &'static str> {
		if action_set_size == 0 {
			Err("Generation of an action set of size 0 is impossbile.")
		} else {
			let mut rng = rand::rng();

			let values = DVector::from_iterator(action_set_size, (0..action_set_size).map(|_| rng.random::<f32>()));
			let costs = DVector::from_iterator(action_set_size, (0..action_set_size).map(|_| rng.random::<f32>()));
			let probabilities = DVector::from_vec(vec![1.0/action_set_size as f32; action_set_size]);

			Ok(ActionSet{values, probabilities, costs})
		}
	}
	/// Generate a geometric distribution over uniform-[0,1) values and costs.
	///
	/// Precisely:
	/// - Values are drawn independently from a continous uniform distribution with support [0,1).
	/// - Costs are drawn independently from a continous uniform distribution with support [0,1).
	/// - Probabilities are 1/2, 1/4, ...
	fn try_generate_random_geometric(action_set_size: usize) -> Result<ActionSet, &'static str> {
		if action_set_size == 0 {
			Err("Generation of an action set of size 0 is impossbile.")
		} else {
			let mut rng = rand::rng();

			let values = DVector::from_iterator(action_set_size, (0..action_set_size).map(|_| rng.random::<f32>()));
			let costs = DVector::from_iterator(action_set_size, (0..action_set_size).map(|_| rng.random::<f32>()));
			let mut probabilities = DVector::from_iterator(action_set_size, (0..action_set_size).map(|s| (2.0 as f32).powi(-(1 + s as i32))));
			
			// make the last two entries identical, since probabilities have to add up to exactly 1.
			probabilities[action_set_size-1] = probabilities[action_set_size-2];

			Ok(ActionSet{values, probabilities, costs})
		}
	}

	/// Generate a geometric distribution over uniform-[0,1) values and costs, with the first value / cost being 0.0
	///
	/// Precisely:
	/// - First value is 0.0, all other values are drawn independently from a continous uniform distribution with support [0,1).
	/// - First cost is 0.0, all other costs are drawn independently from a continous uniform distribution with support [0,1).
	/// - Probabilities are 1/2, 1/4, ...
	fn try_generate_random_geometric_firstvalue0(action_set_size: usize) -> Result<ActionSet, &'static str> {
		if action_set_size <= 1 {
			Err("Generation of an action set of size 0 or size 1 is impossbile.")
		} else {
			let mut rng = rand::rng();

			let mut values = DVector::from_iterator(action_set_size, (0..action_set_size).map(|_| rng.random::<f32>()));
			let mut costs = DVector::from_iterator(action_set_size, (0..action_set_size).map(|_| rng.random::<f32>()));
			let mut probabilities = DVector::from_iterator(action_set_size, (0..action_set_size).map(|s| (2.0 as f32).powi(-(1 + s as i32))));
			
			// make the last two entries identical, since probabilities have to add up to exactly 1.
			probabilities[action_set_size-1] = probabilities[action_set_size-2];
			
			// make the first value and first cost 0.0
			values[0] = 0.0;
			costs[0] = 0.0;

			Ok(ActionSet{values, probabilities, costs})
		}
	}
}

/// Construct a random instance from a given network.
///
/// A [`RewardType`] and a type implementing the trait [`ActionSetGenerator`] have to be provided.
pub fn random_instance_from_network(network: FJNetwork, reward: &RewardType, action_set_generator: &impl ActionSetGenerator) -> FJContractModel {
	let belief_action_set_vec = (0..network.size())
			.map(|_| action_set_generator.try_generate(network.size() as f32).unwrap())
			.collect();
	let reward_vec = match reward {
		RewardType::Unweighted => vec![1.; network.size()]
	};
	return FJContractModel::try_new(network, reward_vec, belief_action_set_vec).unwrap();
}

/// Construct a network with random susceptibility from a graph.
///
/// The size n of the graph and its adjacency matrix in a boolean [`Vec`] of length n² have to be provided.
pub fn random_network_from_graph(size: usize, adjacency_vec: &Vec<bool>) -> FJNetwork {
	let susceptibility_vec = rand::random_iter().take(size).collect();
	return FJNetwork::try_new_uniform(&susceptibility_vec, adjacency_vec).unwrap();
}
