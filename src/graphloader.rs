use std::fs;

/// Loads a graph from a given text file.
///
/// Expects the input file path as string slice, the separating character as a char, and the integer number of the first node.
///
/// The input file format must be as follows:
/// - Each line that shall not be parsed must start with an `%`.
/// - Otherwise, each line represents exactly one edge, represented by two integers separated by `separator`.
///   Nodes are represented by integers, where `firstnode` is the smallest integer that is a node.
/// - Typically, `separator` is either a comma or a white space, and `firstnode` is either 0 or 1.   
///
/// If the input file cannot be parsed, the function might panic!
///
/// If the input file cannot be read, then there will be an error message, and [`None`] is returned.
///
/// Otherwise, a pair `(n, E)` is returned, where `n` is the number of nodes and `E` is the [`Vec`] of edges. 
/// Edges are pairs of different integers, i.e., an undirected graph. 
/// Loop from the input graph are dismissed.

pub fn try_from_file(input_filepath: &str, separator : char, firstnode: usize) -> Option<(usize, Vec<(usize, usize)>)> {
	match fs::read_to_string(input_filepath) {
		Ok(filestring) => {
			let mut n : usize = 0;
			let mut edges = Vec::new();
			
			for line in filestring.lines() {		// check the first char of the line
				let firstchar = line.chars().next();
				
				if firstchar != Some('%') && firstchar != None {
					let i = line.find(separator).unwrap(); // position of first seperator in this line
					let len = line.len();
					let j = match &line[i+1..len].find(separator) {
						Some(k) => *k + i + 1,
						None => len
					};
					let x = &line[0..i];
					let y = &line[i+1..j];
					let u : usize = x.parse::<usize>().unwrap() - firstnode;
					let v : usize = y.parse::<usize>().unwrap() - firstnode;
					
					if u != v { // Do not add loops!
						let edge = (u, v);
						edges.push(edge);
						n = std::cmp::max(n, std::cmp::max(u, v)+1);
					}
				}
			}
			Some((n, edges))
		}
		Err(e) => {
			eprintln!("Could not read file '{input_filepath}': {e:?}");
			None
		}
	}
}

/// Converts a graph is edgelist representation to a graph in adjacency vector representation.
/// 
/// Expects the size `n` (number of nodes) and the list of edges as parameters.
///
/// Returns a symmetric adjacency matrix of the given graph.
///
/// # Example
/// ```
/// use fj_contracting::graphloader::edgelist_to_adjacency_vec;
/// let edgelist = vec![(0,2), (0,3), (1,3), (1,4), (3,4)];
/// let n = 5;
/// let adjacency_vec = edgelist_to_adjacency_vec(n, &edgelist);
/// assert_eq!(adjacency_vec, vec![
///		false, false, true, true, false, 
///		false, false, false, true, true, 
///		true, false, false, false, false, 
///		true, true, false, false, true, 
///		false, true, false, true, false
///	]);
/// ```
		
pub fn edgelist_to_adjacency_vec(n: usize, edgevec: &[(usize, usize)]) -> Vec<bool> {
	let mut e : Vec<bool> = vec![false; n*n];
	
	for (u,v) in edgevec {
		e[u*n + v] = true;
		e[v*n + u] = true;
	}
	return e;
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversion_edgelist_to_adjacency_vec() {
		let edgelist = vec![(0,2), (0,3), (1,3), (1,4), (3,4)];
		let n = 5;
		let adjacency_vec = edgelist_to_adjacency_vec(n, &edgelist);
		assert_eq!(adjacency_vec, vec![
			false, false, true, true, false, 
			false, false, false, true, true, 
			true, false, false, false, false, 
			true, true, false, false, true, 
			false, true, false, true, false
		]);
    }
}
