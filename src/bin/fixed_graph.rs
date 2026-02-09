//! Perform a repeated simulation on random instances generated from a fixed graph.
//!
//! This CLI is used for performing the simulation on an fixed input graph provided by a text file.
//! For each number of influencers from `min_influencer_count` to `max_influencer_count`, the simulation runs multiple times.
//! In each repetition, an instance of the FJ contracting problem is randomly generated from the input graph,
//! using the given parameters. Then the instance is solved both optimally as well as with all implemented heuristics.
//! A statistical analysis of the obtained revenue and approximation ratios over all repetitions is performed and written to file.
//!
//! ## Parameters
//!
//! Required:
//! - `--input-file` : Path to a text file that contains a graph on which the simulation is performed.
//! - `--output_directory` : Path to a directory where the output files will be placed. Ensure write access to that directory.
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
//!
//! Optional:
//! - `--repetitions` : Number of repeated simulations performed for each influencer count. (Default: 128).
//! - `--minimum_node_id` : Node with the smallest id in the input graph. (Default: 1).
//! - `--min_influencer_count` : Minimum number of influencers that should be simulated. (Default: 2). 
//! - `--max_influencer_count` : Maximum number of influencers that should be simulated. (Default: 12).
//! - `--action_set_size` : If the action set generator requires a size, this option must be set to an integer. Otherwise, the program will panic.
//! - `--beta` :  If the action set generator requires a parameter beta, this option must be set to a float between 0 and 1. Otherwise, the program will panic.
//! - `--p0` : If the action set generator requires a parameter p0, this option must be set to a float between 0 and 1. Otherwise, the program will panic.
//!
//! ## Example
//! ```
//! ./fixed_graph --input-file ./graph.txt --output-directory ./results/ --action-set-generator GeometricFirstvalueZero --action-set-size 64 --influencer-selector MostInfluential
//! ```

use fj_contracting::instancesimulator::Simulator;
use fj_contracting::instancegenerator::{RewardType,SimpleAgentSelector,SimpleActionSetGenerator};
use fj_contracting::instancesolver::HeuristicSolver;
use fj_contracting::graphloader;

use clap::Parser;
use std::str::FromStr;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
	/// Number of repeated simulations performed for each influencer count. 
	#[arg(long, default_value_t = 128)]
    repetitions: usize,
    
    /// Path to a text file that contains a graph on which the simulation is performed.
	#[arg(long)]
    input_file: String,
    
	/// Node with the smallest id in the input graph.
    #[arg(long, default_value_t = 1)]
    minimum_node_id: usize,

	/// Path to a directory where the output files will be placed. Ensure write access to that directory.
	#[arg(long)]
	output_directory: String,

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
    
    /// The minimum number of influencers.
    #[arg(long, default_value_t = 2)]
    min_influencer_count: usize,
    
    /// The maximum number of influencers.
    #[arg(long, default_value_t = 12)]
    max_influencer_count: usize,
}

fn main() {
	let separator = ' ';
	let args = Args::parse();
	let generator_type = SimpleActionSetGenerator::from_str(&args.action_set_generator, args.action_set_size, args.beta, args.p0)
		.expect("Action set generator unknown.");
	let reward_type = RewardType::Unweighted;
	let influencer_selector = SimpleAgentSelector::from_str(&args.influencer_selector)
		.expect("Influencer selector unknown.");
	let heuristic_solvers = vec![HeuristicSolver::GreedyImprovementsMCKPSorting, HeuristicSolver::GreedyMCKP];
	
	// Load Graph from file.
	let (n, edgevec) = graphloader::try_from_file(&args.input_file, separator, args.minimum_node_id).unwrap();
	let adjacency_vec = graphloader::edgelist_to_adjacency_vec(n, &edgevec);
	println!("Input graph has {n} nodes and {} edges.", &edgevec.len());
	let filename_base = format!("fixed_n={n}");
	
	// Simulate
	let stime = Instant::now();
	let mut sim = Simulator::init(n, adjacency_vec, reward_type, generator_type, influencer_selector, heuristic_solvers, &args.output_directory, &filename_base);
	sim.run_parallel(args.min_influencer_count, args.max_influencer_count, args.repetitions);
	println!("Total runtime: {:.2?}", stime.elapsed());
}
