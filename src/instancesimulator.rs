use crate::instancegenerator;
use crate::statistics::{BoxPlot,Histogram};
use crate::instancesolver::{InstanceWrapper,SolverResult,SimpleSolver,HeuristicSolver};
use std::time::Instant;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::Write;
use std::collections::HashMap;
use rayon::prelude::*;

#[derive(Debug)]
/// The results of different algorithms for solving a single instance of the FJ contracting problem.
struct SingleSimulationResult<'a> {
	opt_result: SolverResult,
	uncontracted_result: SolverResult,
	heuristic_results: HashMap<&'a HeuristicSolver,SolverResult>,
}
impl<'a> SingleSimulationResult<'a> {
	/// Perform a single simulation on a graph.
	///
	/// Create a random FJ contract instance from a given graph with the given parameters.
	/// Solves the instance using the given set of heuristics.
	pub fn obtain(network_size: usize, 
			adjacency_vec: &Vec<bool>, 
			reward_type: &instancegenerator::RewardType, 
			generator_type: &instancegenerator::SimpleActionSetGenerator, 
			influencer_selector: &instancegenerator::SimpleAgentSelector, 
			influencer_count: usize, 
			heuristics: &'a [HeuristicSolver]) -> Self {
		
		let network = instancegenerator::random_network_from_graph(network_size, adjacency_vec);
		let instance = instancegenerator::random_instance_from_network(network, reward_type, generator_type);
		let instancewrapper = InstanceWrapper::new(instance, influencer_selector, influencer_count);
	
		let opt_result = SolverResult::obtain(&instancewrapper, &SimpleSolver::Optimal, None);
		let uncontracted_result = SolverResult::obtain(&instancewrapper, &SimpleSolver::Uncontracted, Some(opt_result.utility));
		let heuristic_results = HashMap::from_iter(heuristics.iter().map(|heuristic| (heuristic, SolverResult::obtain(&instancewrapper, heuristic, Some(opt_result.utility)))));
	
		Self { opt_result, uncontracted_result, heuristic_results }
	}
}

#[derive(Debug)]
/// A handler for writing box plot data in a certain format to text files.
///
/// There are two files, one for the box plot and one separate file for all outliers.
struct BoxPlotHandler { stats: File, outliers: File }
impl BoxPlotHandler {
	/// Allocate text files with a meaningful names for writing the data.
	///
	/// The meaningful name is specified by an identifier string slice.
	pub fn init(path_base: &str, identifier: &str) -> Self {
		let stat_filepath = format!("{path_base}_{identifier}.stats.dat");
		let outlier_filepath = format!("{path_base}_{identifier}.outliers.dat");
		let mut stat_file = OpenOptions::new().write(true).create(true).truncate(true).open(stat_filepath).expect("Could not create stats file.");
		writeln!(stat_file, "network_size influencer_count min max median lower_q upper_q lower_w upper_w").expect("Error while writing headlines to stats file.");
		let mut outlier_file = OpenOptions::new().write(true).create(true).truncate(true).open(outlier_filepath).expect("Could not create outlier file.");
		writeln!(outlier_file, "network_size influencer_count outlier_value").expect("Error while writing headlines to outliers file.");
		Self { stats: stat_file, outliers: outlier_file }
	}
	/// Write box plot data to files.
	pub fn write_stats(&mut self, network_size: usize, influencer_count: &impl std::fmt::Display, boxplot: BoxPlot<f32>, ) {
		writeln!(self.stats, "{network_size} {influencer_count} {boxplot}").expect("Error while writing boxplot stats to file.");
		for outlier in boxplot.outliers() {
			writeln!(self.outliers, "{network_size} {influencer_count} {outlier}").expect("Error while writing boxplot outliers to file.");
		}
	}
}

#[derive(Debug)]
/// A handler for writing histogram data in a certain format to a text file.
struct HistogramHandler(File);
impl HistogramHandler {
	/// Allocate a file with a meaningful name for writing the data.
	///
	/// The meaningful name is specified by an identifier string slice.
	pub fn init(path_base: &str, identifier: &str) -> Self {
		let hist_filepath = format!("{path_base}_{identifier}.hist");
		let mut hist_file = OpenOptions::new().write(true).create(true).truncate(true).open(hist_filepath).expect("Could not create hist file.");
		writeln!(hist_file, "contracted_agents percentage").expect("Error while writing headlines to stats file.");
		Self(hist_file)
	}
	/// Write histogram data to file.
	pub fn write_contracted_agents(&mut self, influencer_count: &impl std::fmt::Display, repetitions: usize, histogram: Histogram<usize>) {
		for contract_number in 0..histogram.threshold()+1 {
			writeln!(self.0, "{contract_number}/{influencer_count} {}", histogram.distribution()[contract_number] as f32 / repetitions as f32 * 100.0).expect("Error while writing histogram to file.");
		}
	}
}

#[derive(Debug)]
/// Repeated simulation of instances randomly generated from a fixed graph, with a varying number of influencers.
pub struct Simulator {
	network_size: usize,
	adjacency_vec: Vec<bool>,
	reward_type: instancegenerator::RewardType, 
	generator_type: instancegenerator::SimpleActionSetGenerator, 
	influencer_selector: instancegenerator::SimpleAgentSelector,
	heuristic_solvers: Vec<HeuristicSolver>,
	heuristic_solver_files: Vec<(BoxPlotHandler, BoxPlotHandler, HistogramHandler)>,	// utility, approx, contracted_agents
	opt_files: (BoxPlotHandler, HistogramHandler),										// utility, contracted_agents
	uncontracted_files: (BoxPlotHandler, BoxPlotHandler),								// utility, approx
}
impl Simulator {
	/// Create a simulator for a fixed graph using the specified generator parameters.
	/// Prepare the output files for statistics about the results obtained with specified solvers.
	pub fn init(
		network_size: usize,
		adjacency_vec: Vec<bool>,
		reward_type: instancegenerator::RewardType, 
		generator_type: instancegenerator::SimpleActionSetGenerator, 
		influencer_selector: instancegenerator::SimpleAgentSelector,
		heuristic_solvers: Vec<HeuristicSolver>,
		output_directory: &str,
		filename_base: &str,
	) -> Self {
	
		let path_base = format!("{output_directory}/{filename_base}_support={:?}_reward={:?}_agents={:?}", &generator_type, &reward_type, &influencer_selector);
		
		let heuristic_solver_files: Vec<_> = heuristic_solvers
			.iter()
			.map(|heuristic| (
					BoxPlotHandler::init(&path_base, &format!("{heuristic:?}_util")), 
					BoxPlotHandler::init(&path_base, &format!("{heuristic:?}_approx")),
					HistogramHandler::init(&path_base, &format!("{heuristic:?}_agents"))
				)
			)
			.collect();
		let opt_files = (
			BoxPlotHandler::init(&path_base, "OPT_util"), 
			HistogramHandler::init(&path_base, "OPT_agents")
		);
		let uncontracted_files = (
			BoxPlotHandler::init(&path_base, "Uncontracted_util"), 
			BoxPlotHandler::init(&path_base, "Uncontracted_approx"), 
		);
		
		Self { network_size, adjacency_vec, reward_type, generator_type, influencer_selector, heuristic_solvers, heuristic_solver_files, opt_files, uncontracted_files }
	}
	
	/// Run the simulator iteratively for a varying number of influencers.
	/// For each number of influencers, `repeat` many instances are created and solved in parallel.
	///
	/// Statistical evaluation subject to the influencer number is performed afterwards.
	pub fn run_parallel(&mut self, min_influencer_count: usize, max_influencer_count: usize, repeat: usize) {
		println!("Simulation for support={:?}, reward={:?}, agents={:?} with agent counts from {} to {}, and solving heuristics {:?}",  &self.generator_type, &self.reward_type, &self.influencer_selector, min_influencer_count, max_influencer_count, &self.heuristic_solvers);
		for influencer_count in min_influencer_count..max_influencer_count+1 {
			println!(" > Simulate {} instances with {influencer_count} agents in parallel.", repeat);
			let stime = Instant::now();
			let sim_result_collection: Vec<SingleSimulationResult> = (0..repeat)
				.into_par_iter()	// Parallel
				//.into_iter()		// Iterative
				.map(|_| SingleSimulationResult::obtain(self.network_size, &self.adjacency_vec, &self.reward_type, &self.generator_type, &self.influencer_selector, influencer_count, &self.heuristic_solvers))
				.collect();
			
			println!(" > ... Finished after {:.2?}.", stime.elapsed());
	
			let opt_results: Vec<_> = sim_result_collection.iter().map(|result| result.opt_result).collect();
			self.opt_files.0.write_stats(self.network_size, &influencer_count, SolverResult::utility_boxplot(&opt_results));
			self.opt_files.1.write_contracted_agents(&influencer_count, repeat, SolverResult::contractcount_histogram(&opt_results, influencer_count));
			
			let uncontracted_results: Vec<_> = sim_result_collection.iter().map(|result| result.uncontracted_result).collect();
			self.uncontracted_files.0.write_stats(self.network_size, &influencer_count, SolverResult::utility_boxplot(&uncontracted_results));
			self.uncontracted_files.1.write_stats(self.network_size, &influencer_count, SolverResult::utility_approx_factor_boxplot(&uncontracted_results));
			
			for (solver,(util_file,approx_file,contractcount_file)) in std::iter::zip(&self.heuristic_solvers,&mut self.heuristic_solver_files) {
				let result: Vec<_> = sim_result_collection.iter().map(|result| *result.heuristic_results.get(&solver).unwrap()).collect();
				util_file.write_stats(self.network_size, &influencer_count, SolverResult::utility_boxplot(&result));
				approx_file.write_stats(self.network_size, &influencer_count, SolverResult::utility_approx_factor_boxplot(&result));
				contractcount_file.write_contracted_agents(&influencer_count, repeat, SolverResult::contractcount_histogram(&result, influencer_count));	
			}
		}		
	}
}

#[derive(Debug)]
/// Repeated simulation of instances randomly generated from multiple graphs, with a fixed number of influencers.
pub struct MultiSimulator {
	reward_type: instancegenerator::RewardType, 
	generator_type: instancegenerator::SimpleActionSetGenerator, 
	influencer_selector: instancegenerator::SimpleAgentSelector,
	heuristic_solvers: Vec<HeuristicSolver>,
	heuristic_solver_files: Vec<(BoxPlotHandler, BoxPlotHandler, HistogramHandler)>,	// utility, approx, contracted_agents
	opt_files: (BoxPlotHandler, HistogramHandler),										// utility, contracted_agents
	uncontracted_files: (BoxPlotHandler, BoxPlotHandler),								// utility, approx
}
impl MultiSimulator {
	/// Create a simulator independently from a graph, using the specified generator parameters.
	/// Prepare the output files for statistics about the results obtained with specified solvers.
	pub fn init(
		reward_type: instancegenerator::RewardType, 
		generator_type: instancegenerator::SimpleActionSetGenerator, 
		influencer_selector: instancegenerator::SimpleAgentSelector,
		heuristic_solvers: Vec<HeuristicSolver>,
		output_directory: &str,
		filename_base: &str,
	) -> Self {
	
		let path_base = format!("{output_directory}/{filename_base}_support={:?}_reward={:?}_agents={:?}", &generator_type, &reward_type, &influencer_selector);
		
		let heuristic_solver_files: Vec<_> = heuristic_solvers
			.iter()
			.map(|heuristic| (
					BoxPlotHandler::init(&path_base, &format!("{heuristic:?}_util")), 
					BoxPlotHandler::init(&path_base, &format!("{heuristic:?}_approx")),
					HistogramHandler::init(&path_base, &format!("{heuristic:?}_agents"))
				)
			)
			.collect();
		let opt_files = (
			BoxPlotHandler::init(&path_base, "OPT_util"), 
			HistogramHandler::init(&path_base, "OPT_agents")
		);
		let uncontracted_files = (
			BoxPlotHandler::init(&path_base, "Uncontracted_util"), 
			BoxPlotHandler::init(&path_base, "Uncontracted_approx"), 
		);
		
		Self { reward_type, generator_type, influencer_selector, heuristic_solvers, heuristic_solver_files, opt_files, uncontracted_files }
	}
	
	
	/// Run the simulator for a given set of graphs of identical size in parallel, for a fixed number of influencers.
	pub fn run_parallel_on_alternative_graphs(&mut self, network_size: usize, influencer_count: usize, adjacency_vec_slice: &[Vec<bool>]) {
		let repetitions = adjacency_vec_slice.len();
		
		println!("For each graph, generate instances with {influencer_count} agents and perform simulation in parallel ...");
		
		let stime = Instant::now();
		let sim_result_collection: Vec<SingleSimulationResult> = adjacency_vec_slice
			.into_par_iter()	// Parallel
			//.into_iter()		// Iterative
			.map(|adjacency_vec| SingleSimulationResult::obtain(network_size, adjacency_vec, &self.reward_type, &self.generator_type, &self.influencer_selector, influencer_count, &self.heuristic_solvers))
			.collect();
		
		println!("... finished after {:.2?}.", stime.elapsed());

		let opt_results: Vec<_> = sim_result_collection.iter().map(|result| result.opt_result).collect();
		self.opt_files.0.write_stats(network_size, &influencer_count, SolverResult::utility_boxplot(&opt_results));
		self.opt_files.1.write_contracted_agents(&influencer_count, repetitions, SolverResult::contractcount_histogram(&opt_results, influencer_count));
		
		let uncontracted_results: Vec<_> = sim_result_collection.iter().map(|result| result.uncontracted_result).collect();
		self.uncontracted_files.0.write_stats(network_size, &influencer_count, SolverResult::utility_boxplot(&uncontracted_results));
		self.uncontracted_files.1.write_stats(network_size, &influencer_count, SolverResult::utility_approx_factor_boxplot(&uncontracted_results));
		
		for (solver,(util_file,approx_file,contractcount_file)) in std::iter::zip(&self.heuristic_solvers, &mut self.heuristic_solver_files) {
			let result: Vec<_> = sim_result_collection.iter().map(|result| *result.heuristic_results.get(&solver).unwrap()).collect();
			util_file.write_stats(network_size, &influencer_count, SolverResult::utility_boxplot(&result));
			approx_file.write_stats(network_size, &influencer_count, SolverResult::utility_approx_factor_boxplot(&result));
			contractcount_file.write_contracted_agents(&influencer_count, repetitions, SolverResult::contractcount_histogram(&result, influencer_count));	
		}		
	}
}
