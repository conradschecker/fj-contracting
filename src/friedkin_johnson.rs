use nalgebra::{DMatrix,DVector};

#[derive(Debug)]
/// Implementation of Friedkin-Johnson opinion dynamics.
/// The model goes back to *N. Friedkin and E. Johnsen. Social influence and opinions. J. Math. Soc., 15(3-4): 193–206, 1990.*
pub struct FJNetwork {
	// the number of nodes of the network.
	size: usize,
	
	// stochastic quadratic matrix with "incoming" edge weights (row i, column j is the influence from j on i).
	influence_matrix: DMatrix<f32>,	
	
	// diagonal matrix with susceptibility values.
	lambda_matrix: DMatrix<f32>,
	
	// matrix that represents the mapping of initial opinions to public opinions in equilibrium.
	final_mapping: DMatrix<f32>,
}

impl std::fmt::Display for FJNetwork {
	/// Display the representational matrices for the FJ network.
	/// This is the influence matrix (A) and the suceptibility matrix (Lambda).
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Influence:{} Lambda:{}", self.influence_matrix, self.lambda_matrix)
    }
}


impl FJNetwork {
	/// Construct a new instance.
	///
	/// Expects the susceptibility values ("lambda") as a vector of sizes n.
	/// The weights betweens the nodes have to be given as a vector of size n² that represents the matrix in column-major order.
	/// Entry on row i, column j is the influence from j on i.
	pub fn try_new(susceptibility_vec: Vec<f32>, influence_weight_vec: Vec<f32>) -> Result<Self, &'static str> {
		let size = susceptibility_vec.len();
		
		if size == 0 {
			Err("Could not create FJNetwork: Vec for susceptibilities cannot be empty.")
		} else if size.pow(2) != influence_weight_vec.len() {
			Err("Could not create FJNetwork: Vec for susceptibilities and Vec for influence weights had incompatible lengths.")
		} else {
			let influence_matrix = DMatrix::from_vec(size, size, influence_weight_vec);
			// TODO: Check feasibility: Is the weights matrix stochastic?

			let susceptibilities = DVector::from_vec(susceptibility_vec);
			// TODO: Check feasibility: Are the values from [0,1]?

			Self::new(size, influence_matrix, susceptibilities)
		}
	}
	
	/// Create an instance from a adjacency matrix in vector notation, with uniform influence between the nodes.
	///
	/// Expects the susceptibility values ("lambda") as a vector of size n, and 
	/// a boolean vector of length n² that represents the adjacency matrix in column-major order.
	/// The influence from i to j is 1 / deg(j).
	/// Entry on row i, column j specifices whether there is an edge from i to j.
	pub fn try_new_uniform(susceptibility_vec: &Vec<f32>, adjacency_vec: &Vec<bool>) -> Result<Self, &'static str> {
		let size = susceptibility_vec.len();
		
		if size == 0 {
			Err("Could not create FJNetwork: Vec for susceptibilities cannot be empty.")
		} else if size.pow(2) != adjacency_vec.len() {
			Err("Could not create FJNetwork: Vec for susceptibilities and Vec for adjacency matrix had incompatible lengths.")
		} else {
			// Compute the number of incoming edges for each agent:
			// Take each column from the adjacency matrix and collects the number of "true"-values in it.
			let incoming_edges_vec = &adjacency_vec
				.chunks(size)
				.map(|chunk| chunk
					.iter()
					.map(|x| *x as usize)
					.sum::<usize>()
				).collect::<Vec<usize>>();
			
			
			// Generate uniform weights, which is 1 / incoming_edges for every non-zero entry (and 0 otherwise).
			let influence_matrix = DMatrix::from_iterator(size, size, adjacency_vec
				.iter()
				.enumerate()
				.map(|(i,x)| match x {
					false => 0.0,
					true => 1.0 / incoming_edges_vec[i / size] as f32
				})
			).transpose();

			// Create the susceptibility vector
			// If a node as no incoming edges, it needs to be completely stubborn (susceptibility = 0.0).
			let susceptibilities = DVector::from_iterator(
				size, 
				std::iter::zip(
					susceptibility_vec.iter(), 
					incoming_edges_vec.iter()
				).map(|(susceptibility, incoming_edges)| match incoming_edges {
					0 => 0.,
					_ => *susceptibility
				}));
			// TODO: Check feasibility: Are the values in susceptibility_vec from [0,1]?

			Self::new(size, influence_matrix, susceptibilities)
		}
	}
	
	/// Creates an instance without input check.
	/// Expects a quadratic DMatrix for influence weights and a DVector for the susceptibilities.
	/// Computes a matrix inverse, which is a costly operation!
	/// Matrix inverse computation might fail. In this case, return Err.
	fn new(size: usize, influence_matrix: DMatrix<f32>, susceptibilities: DVector<f32>) -> Result<Self, &'static str> {
		let id: DMatrix<f32> = DMatrix::identity(size, size);
		let lambda_matrix: DMatrix<f32> = DMatrix::from_diagonal(&susceptibilities);
		let inverse_option = (&id - (&lambda_matrix * &influence_matrix)).try_inverse();

		match inverse_option {
			Some(inverse) => {
				let final_mapping: DMatrix<f32> = inverse * (&id - &lambda_matrix);
				Ok(Self { size, influence_matrix, lambda_matrix, final_mapping })
			},
			None => Err("Could not create FJNetwork: Matrix inverse computation failed. Perhaps the given values for influence weights and susceptibilities are not representing a valid Friedkin-Johnson instance.")
		}
	}

	/// Returns the number of nodes in the network.
	pub fn size(&self) -> usize {
		self.size
	}
	
	/// Returns a vector that contains a pair (i, outdegree(i)) for each vertex with index i.
	/// outdegree(i) is the number of (other) nodes on which i has a direct positive influence.
	pub fn get_outdegrees(&self) -> Vec<(usize, usize)> {
		(0..self.size).map(|j| (j, self.influence_matrix.column(j).iter().filter(|x| **x > 0.0).count())).collect::<Vec<(usize, usize)>>()
	}
		
	/// Returns the linear mapping that translates initial opinions to public opinions in equilibrium.
	pub fn get_final_mapping(&self) -> &DMatrix<f32> {
		return &self.final_mapping;
	}
	
	/// Returns a vector of public opinions after a single step of opinion formation.
	/// Parameters are the intrinsic opinions and the current public opinions
	pub fn fj_single_step(&self, initial_opinions: &DVector<f32>, public_opinions: &DVector<f32>) -> DVector<f32> {
		let id: DMatrix<f32> = DMatrix::identity(self.size, self.size);
		return (&id - &self.lambda_matrix) * initial_opinions + &self.lambda_matrix * &self.influence_matrix * public_opinions;
	}
	
	/// Returns a vector of public opinions after a given number of steps in the Friedkin-Johnson model
	/// Parameters are the intrinsic opinions and the current public opinions
	pub fn fj_steps(&self, number_of_rounds: usize, initial_opinions: &DVector<f32>, public_opinions: &mut DVector<f32>) {
		for _ in 0..number_of_rounds {
			*public_opinions = self.fj_single_step(initial_opinions, public_opinions);
		}
	}
	
	/// Returns a vector of final opinions for a given vector of intrinsic beliefs.
	pub fn fj_final(&self, initial_opinions: &DVector<f32>) -> DVector<f32> {
		&self.final_mapping * initial_opinions
	}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn network_creation_from_matrix() {
		let susceptibility_vec = vec![0., 1.];
		let influence_weight_vec = vec![0., 1., 1., 0.];
        let network = FJNetwork::try_new(susceptibility_vec, influence_weight_vec);
        assert!(network.is_ok());
        let network = network.unwrap();
        assert_eq!(network.size(), 2);
    }
    #[test]
    fn network_creation_from_adjacency_vec_1() {
		let susceptibility_vec = vec![0.942, 0.284, 0.345, 0.234, 0.773];
		let adjacency_vec = vec![false, false, true, true, false, false, false, false, true, true, true, false, false, false, false, true, true, false, false, true, false, true, false, true, false];
        let network = FJNetwork::try_new_uniform(&susceptibility_vec, &adjacency_vec);
        assert!(network.is_ok());
        let network = network.unwrap();
		assert_eq!(network.influence_matrix, DMatrix::from_vec(5,5, vec![0.,0.,1.,1./3.,0.,0.,0.,0.,1./3.,1./2.,1./2.,0.,0.,0.,0.,1./2.,1./2.,0.,0.,1./2.,0.,1./2.,0.,1./3.,0.]));
    }
    #[test]
    fn network_creation_from_adjacency_vec_2() {
		let susceptibility_vec = vec![0.5, 1.0, 0.75];
		let adjacency_vec = vec![false, false, true, false, false, false, true, true, false];
        let network = FJNetwork::try_new_uniform(&susceptibility_vec, &adjacency_vec);
        assert!(network.is_ok());
		let influence_matrix_expected = DMatrix::from_vec(3,3, vec![0.,0.,0.5,0.,0.,0.5,1.,0.,0.]);
		let lambda_matrix_expected = DMatrix::from_vec(3,3, vec![0.5,0.,0.,0.,0.,0.,0.,0.,0.75]);
		let final_mapping_expected = DMatrix::from_vec(3, 3, vec![8./13.,0.,3./13.,3./13.,1.,6./13.,2./13.,0.,4./13.]);
		let network = network.unwrap();
		assert_eq!(network.influence_matrix, influence_matrix_expected);
        assert_eq!(network.lambda_matrix, lambda_matrix_expected);
        assert_eq!(network.final_mapping, final_mapping_expected);
    }
    #[test]
    fn network_creation_outdegrees() {
		let susceptibility_vec = vec![0.75, 0.25, 0.1];
		let influence_weight_vec = vec![0., 0.125, 0.25, 1., 0., 0.75, 0., 0.875, 0.];
        let network = FJNetwork::try_new(susceptibility_vec, influence_weight_vec);
        let outdegrees = vec![(0,2), (1,2), (2,1)];
        assert_eq!(network.unwrap().get_outdegrees(), outdegrees);
    }
    #[test]
    fn network_creation_finalopinion_1() {
		let susceptibility_vec = vec![0., 0.25];
		let influence_weight_vec = vec![0., 1., 0., 0.];
        let network = FJNetwork::try_new(susceptibility_vec, influence_weight_vec).unwrap(); 
        let final_mapping_expected = DMatrix::from_vec(2, 2, vec![1., 0.25, 0., 0.75]);
        assert_eq!(*network.get_final_mapping(), final_mapping_expected);
    }
    #[test]
    fn network_creation_finalopinion_2() {
		let susceptibility_vec = vec![0.25, 0.5];
		let influence_weight_vec = vec![0., 1., 1., 0.];
        let network = FJNetwork::try_new(susceptibility_vec, influence_weight_vec).unwrap(); 
        let final_mapping_expected = DMatrix::from_vec(2, 2, vec![6./7., 3./7., 1./7., 4./7.]);
        for (result, expected) in std::iter::zip(network.get_final_mapping().iter(),  final_mapping_expected.iter()) {
			assert!((result - expected).abs() < 1./(2.0_f32.powf(6.0)));
		}
    }
    #[test]
    fn network_creation_finalopinion_3() {
		let susceptibility_vec = vec![0.75, 0.25, 0.1];
		let influence_weight_vec = vec![0., 0.125, 0.25, 1., 0., 0.75, 0., 0.875, 0.];
        let network = FJNetwork::try_new(susceptibility_vec, influence_weight_vec); 
        let intrinsics = DVector::from_vec(vec![1., 0., 1.]);
        let final_opinion_expected = DMatrix::from_vec(3, 3, vec![1259./4895., 47./4895., 7./979., 576./979., 768./979., 72./979., 756./4895., 1008./4895., 900./979.]) * &intrinsics;
		let network = network.unwrap();
		for (result, expected) in std::iter::zip(network.fj_final(&intrinsics).iter(),  final_opinion_expected.iter()) {
			assert!((result - expected).abs() < 1./(2.0_f32.powf(6.0)));
		}
    }
}
