//! Perform a repeated simulation on random instances generated with a set of (hyperbolic random) graphs.
//!
//! This CLI is used for performing the simulation on a set of graphs, typically retrieved from an external source.
//! Each graph has to be provided in a seperate file in the given directory with a certain file name structure.
//! For each repetitions, a different graph is retrieved and an instance of the FJ contracting problem
//! is randomly generated using that graph and the given parameters. 
//! Then the instance is solved both optimally as well as with all implemented heuristics.
//! A statistical analysis of the obtained revenue and approximation ratios over all repetitions is performed and written to file.
//! 
//! The network size may vary from `n_min` to `n_max`, and all sizes have to be powers of two. 
//! For example, if `n_min = 256` and `n_max 2048`, then the network size are 256, 512, 1024 and 2048.
//! The number of repetitions specifies how many graphs of each network size are required to exist.
//! The file names are following the format "hrg\_*n*\_*i*\_edges.txt", 
//! where *n* and *i* are integers specifying the network size and the iteration, respectively. For example, 
//! if `repetitions = 128`, then for each network size *n*, the files "hrg\_*n*\_0\_edges.txt", ..., "hrg\_*n*\_127\_edges.txt"
//! have to exist in the input directory
//!
//! The number of influencers is given by `a * log_2(n) - b`, where `a` and `b` are parameters and `n` is the size of the network.
//!
//! ## Parameters
//!
//! Required:
//! - `--input-directory` : Path to a directory in which the input graphs are contained as text files with a certain file name structure.
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
//! - `--influencer_count_param_a` : Integer parameter of the function `a * log_2(n) - b` for the influencer count.
//! - `--influencer_count_param_b` : Integer parameter of the function `a * log_2(n) - b` for the influencer count.
//! - `--n_min` : The minimum network size. Must be a power of two.
//! - `--n_max` : The maxmimum network size. Must be a power of two.
//!
//! Optional:
//! - `--repetitions` : Number of repeated simulations performed for each influencer count. (Default: 128).
//! - `--minimum_node_id` : Node with the smallest id in the input graph. (Default: 0).
//! - `--action_set_size` : If the action set generator requires a size, this option must be set to an integer. Otherwise, the program will panic.
//! - `--beta` :  If the action set generator requires a parameter beta, this option must be set to a float between 0 and 1. Otherwise, the program will panic.
//! - `--p0` : If the action set generator requires a parameter p0, this option must be set to a float between 0 and 1. Otherwise, the program will panic.
//!
//! ## Example
//! ```
//! ./hrg --input-directory ./graphs/ --output-directory ./results/ --action-set-generator GeometricFirstvalueZero --action-set-size 64 --influencer-selector MostInfluential --influencer-count-param-a 1 --influencer-count-param-b 7 --n-min 256 --n-max 4096
//! ```

use fj_contracting::instancesimulator::MultiSimulator;
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

	/// Node with the smallest id in the input graph.
    #[arg(long, default_value_t = 0)]
    minimum_node_id: usize,
    
	/// Path to a directory in which the input graphs are contained as text files with a certain file name structure.
	#[arg(long)]
    input_directory: String,
	
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

    /// Parameter of the function `a * log_2(n) - b` that gives the number of influencers for a network of size n.
    #[arg(long)]
    influencer_count_param_a: u32,
    
	/// Parameter of the function `a * log_2(n) - b` that gives the number of influencers for a network of size n.
    #[arg(long)]
    influencer_count_param_b: u32,
    
    /// The minimum network size. Must be a power of two.
    #[arg(long)]
    n_min: usize,
    
    /// The maxmimum network size. Must be a power of two.
    #[arg(long)]
    n_max: usize,
}


fn influencer_count(influencer_count_param_a: u32, influencer_count_param_b: u32, logsize: u32) -> usize {
	return (influencer_count_param_a * logsize - influencer_count_param_b) as usize
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
	let filename_base = format!("hrg_agentnumber={}log2(n)-{}", args.influencer_count_param_a, args.influencer_count_param_b);
	
	let influencer_count_alpha = args.influencer_count_param_a;
	let influencer_count_beta = args.influencer_count_param_b;
	let n_min_log = args.n_min.ilog2();
	let n_max_log = args.n_max.ilog2();
	
	let min_influencer_count = influencer_count(influencer_count_alpha, influencer_count_beta, n_min_log);
	let max_influencer_count = influencer_count(influencer_count_alpha, influencer_count_beta, n_max_log);
	
	println!("Initialize simulation with the following parameters:");
	println!(" > Network sizes from {} to {}.", 2_usize.pow(n_min_log), 2_usize.pow(n_max_log));
	println!(" > action-set-generator={:?}", &generator_type);
	println!(" > reward-type={:?}", &reward_type);
	println!(" > influencer-selector={:?}", &influencer_selector);
	println!(" > influencer-count(n)={}log2(n)-{}, i.e., from {} to {}", influencer_count_alpha, influencer_count_beta, min_influencer_count, max_influencer_count);
	println!(" > heuristics {:?}", &heuristic_solvers);
	
	if min_influencer_count <= 0 {
		eprintln!("Error: The influencer count has to be strictly positive for all networks!");
	} else {
	
		let stime = Instant::now();
		let mut sim = MultiSimulator::init(reward_type, generator_type, influencer_selector, heuristic_solvers, &args.output_directory, &filename_base);
		
		for logsize in n_min_log..=n_max_log {
			let size = 2_usize.pow(logsize);
			// Load graphs from file.
			let stime_graphloading = Instant::now();
			println!("Load {} hyperbolic random graphs of size {size} from disk ...", args.repetitions);
			let nodes_agents_adjacency_vec_list: Vec<Vec<bool>> = (0..args.repetitions).into_iter()
				.map(|repetition| {
					let filename = String::from(&format!("{}/hrg_{}_{}_edges.txt", &args.input_directory, size, repetition));
					let (_, edgevec) = graphloader::try_from_file(&filename, separator, 0).unwrap();
					let adjacency_vec = graphloader::edgelist_to_adjacency_vec(size, &edgevec);
					adjacency_vec
				})
				.collect();
			let influencer_count = influencer_count(influencer_count_alpha, influencer_count_beta, logsize);
			println!("... completed in {:.2?}.", stime_graphloading.elapsed());

			// Simulate
			sim.run_parallel_on_alternative_graphs(size, influencer_count, &nodes_agents_adjacency_vec_list);
		}
		println!("Total runtime: {:.2?}.", stime.elapsed());
	}
}


#[cfg(test)]
mod tests {
    use super::*;

	#[test]
	fn test_influencer_count() {
		assert_eq!(influencer_count(1, 4, 5), 1); // a = 1, b = 4, log(n) = 5 ... => a * log(n) - b = 1
		assert_eq!(influencer_count(2, 3, 6), 9); // a = 2, b = 3, log(n) = 6 ... => a * log(n) - b = 9
	}
}
  
