//! Evaluate the performance of the simulation.
//!
//! This is a CLI with the main purpose of providing a testing environment.
//! It extends a given start graph to a specified size, according to the Barabási-Albert model of random graphs.
//! Then, a random instance of the FJ contracting problem is created from this extended graph and the specified parameters.
//! Afterwards, the instance is solved using different methods (exact and approximative) that are implemented.
//! There are extensive measurements of the running time of different parts of the simulation, which enables identification of bottlenecks.
//!
//! ## Parameters
//!
//! Required:
//! - `--input-file` : Path to a text file that contains a (small) start graph.
//! - `--n-extend` : Number of nodes that should be added to the start graph.
//! - `--mu` : Number of edges that should be added for each additional node.
//! - `--action-set-generator` : [`SimpleActionSetGenerator`] that should be used. Possible Choices:
//! 	- `DiscreteUniform` (requires `action-set-size` to be specified)
//! 	- `Geometric` (requires `action-set-size` to be specified)
//! 	- `GeometricFirstvalueZero` (requires `action-set-size` to be specified)
//! 	- `BinaryBeta` (requires `beta` to be specified)
//! 	- `BinaryP0` (requires `p0` to be specified)
//! 	- `BinaryP0Proportional` (uses `BinaryP0` with `p0 = 1 - 1/n`, where `n` is the network size)
//! - `--influencer-selector` : [`SimpleAgentSelector`] that should be used for selecting influencers. Possible Choices:
//! 	- Random
//!		- HighestOutDegree
//!		- MostInfluential
//! - `--influencer-count` : The number of influencers.
//!
//! Optional:
//! - `--minimum_node_id` : Node with the smallest id in the input graph. (Default: 1).
//! - `--action_set_size` : If the action set generator requires a size, this option must be set to an integer. Otherwise, the program will panic.
//! - `--beta` :  If the action set generator requires a parameter beta, this option must be set to a float between 0 and 1. Otherwise, the program will panic.
//! - `--p0` : If the action set generator requires a parameter p0, this option must be set to a float between 0 and 1. Otherwise, the program will panic.
//!
//! ## Example
//! ```
//! ./bench --input-file ./graph.txt --n-extend 1019 --mu 5 --action-set-generator GeometricFirstvalueZero --action-set-size 64 --influencer-count 10 --influencer-selector MostInfluential
//! ```

use fj_contracting::instancegenerator;
use fj_contracting::instancegenerator::{RewardType,SimpleAgentSelector,SimpleActionSetGenerator};
use fj_contracting::instancesolver::{InstanceWrapper,SolverResult,SimpleSolver,HeuristicSolver};
use fj_contracting::graphloader;
use fj_contracting::graphgenerator;

use clap::Parser;
use std::time::Instant;
use std::str::FromStr;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
	/// Path to a text file that contains a (small) start graph.
	#[arg(long)]
    input_file: String,
    
    /// Node with the smallest id in the input graph.
    #[arg(long, default_value_t = 1)]
    minimum_node_id: usize,
    
    /// Number of nodes that should be added to the start graph.
	#[arg(long)]
	n_extend: usize,
    
    /// Number of edges that should be added for each additional node.
    #[arg(long)]
    mu: usize,

	/// SimpleActionSetGenerator that should be used. 
	/// Possible Choices:
	/// DiscreteUniform, Geometric, GeometricFirstvalueZero, BinaryBeta, BinaryP0, BinaryP0Proportional
	#[arg(long)]
    action_set_generator: String,
    
    /// If the action set generator requires a size, this option must be set to an integer.
    #[arg(long)]
    action_set_size: Option<usize>,
    
    /// If the action set generator requires a parameter beta, this option must be set to a float between 0 and 1.
    #[arg(long)]
    beta: Option<f32>,
    
    /// If the action set generator requires a parameter p0, this option must be set to a float between 0 and 1.
    #[arg(long)]
    p0: Option<f32>,
    
     /// SimpleAgentSelector that should be used for selecting influencers.
    /// Possible Choices: Random, HighestOutDegree, MostInfluential
    #[arg(long)]
    influencer_selector: String,
    
    /// The number of influencers.
    #[arg(long)]
    influencer_count: usize,
}

fn main() {
	let separator = ' ';
	let args = Args::parse();
	let generator_type = SimpleActionSetGenerator::from_str(&args.action_set_generator, args.action_set_size, args.beta, args.p0)
		.expect("Action set generator unknown.");
	let reward_type = RewardType::Unweighted;
	let influencer_selector = SimpleAgentSelector::from_str(&args.influencer_selector)
		.expect("Influencer selector unknown.");
	let heuristic_solvers = vec![HeuristicSolver::GreedyImprovementsMCKPSorting, HeuristicSolver::GreedyMCKPOnlyKnapsack, HeuristicSolver::BestSingleAgentContract, HeuristicSolver::GreedyMCKP];
	
	// Load Graph from file.
	let (n0, startedges) = graphloader::try_from_file(&args.input_file, separator, args.minimum_node_id).unwrap();
	println!("Obtained graph with {n0} nodes and {} edges.", startedges.len());
	let extended_edgelist = graphgenerator::ba_edgelist(args.n_extend, args.mu, &startedges).unwrap();
	let n = n0 + args.n_extend;
	let adjacency_vec = graphloader::edgelist_to_adjacency_vec(n, &extended_edgelist);
	println!("Preferential Attachment of {} additional nodes and {} edges for each attached node completed.", args.n_extend, args.mu);
	
	// Create FJ Network
	let stime = Instant::now();
	let network = instancegenerator::random_network_from_graph(n, &adjacency_vec);
	println!("Network with {} nodes created in {:.2?} (mostly matrix inverse computation).", network.size(), stime.elapsed());
	
	// Create and wrap FJ Contract Instance
	let stime = Instant::now();
	let instance = instancegenerator::random_instance_from_network(network, &reward_type, &generator_type);
	let instancewrapper = InstanceWrapper::new(instance, &influencer_selector, args.influencer_count);
	println!("Instance with {} agents created and wrapped in {:.2?} (mostly breakpopint computation; obtained {} breakpoints).", args.influencer_count, stime.elapsed(), instancewrapper.get_breakpoint_sum_for_influencers());
	
	let opt_states_to_check: usize = instancewrapper.get_breakpoint_count_vec().iter().product();
	
	if opt_states_to_check > 5000000 {
		println!("Computing OPT: Find the maximum utility yielded by one out of ~{} million states...", opt_states_to_check / 1000000);
	} else if opt_states_to_check > 5000 {
		println!("Computing OPT: Find the maximum utility yielded by one out of ~{} thousand states...", opt_states_to_check / 1000);
	} else {
		println!("Computing OPT: Find the maximum utility yielded by one out of {} states...", opt_states_to_check);
	}
	
	let stime = Instant::now();
	let opt = SolverResult::obtain(&instancewrapper, &SimpleSolver::Optimal, None);
	let duration = stime.elapsed();
	println!(" > OPT = {:.5?} with {} contract(s), alpha_sum = {}, reward_contribution = {}, obtained in {duration:.2?} (speed: ~{:.2?}M states per second).", opt.utility, opt.contract_count, opt.alpha_sum, opt.reward_contribution, opt_states_to_check as u128 / duration.as_micros());
		
	for heuristic in &heuristic_solvers {

		let stime = Instant::now();
		let heuristic_result = SolverResult::obtain(&instancewrapper, heuristic, Some(opt.utility));
		let duration = stime.elapsed();
		println!(" > {heuristic:?} = {:.5?} with {} contract(s), alpha_sum = {}, reward_contribution = {}, obtained in {duration:.2?}.", heuristic_result.utility, heuristic_result.contract_count, heuristic_result.alpha_sum, heuristic_result.reward_contribution);
	}
}
