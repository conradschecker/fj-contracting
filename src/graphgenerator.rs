use rand;

/// Random draw from a geom(p) distribution in O(1).
fn geom(p: f32) -> usize {
	match p {
		0.0 => usize::MAX, // representing infinity
		1.0 => 0,
		_ => {
			let u = rand::random::<f32>();
			((1.0-u).log(1.0-p) - 1.0).ceil() as usize
		}
	}
}

/// Create directed gilbert random graph G(n,p) with n nodes and independent edge probability of p for all edges.
///
/// Returns an boolean adjacency matrix as vector of length n² in column-major order.
pub fn directed_gilbert_graph_without_loops(n: usize, p: f32) -> Vec<bool> {
	let mut e : Vec<bool> = vec![false; n*n];
	let mut k : usize = 0;
	
	while n > 1 {
		let l = geom(p);
		
		// Stop if we would not be within bounds of e.
		if l == usize::MAX || k >= n*n - 1 {
			break;
		}
		
		// Skip l entries and the diagonal
		k += l + 1 + ((l + k + 1) / (n + 1)) - (k / (n + 1));

		// Write back to e if we're still within bounds of e.
		if k < n*n - 1 {
			e[k] = true;
		}
	}
	return e;
}

/// Creates a Barabási-Albert random graph from a given start graph.
///
/// Parameters are n_rand, which is the number of additional nodes, and param_nu, which is the number of edges that each additional node is guaranteed to have.
///
/// The start graph is to be given as a slice of pairs, where each pair is an endge in the original graph.
/// Nodes in the original graph are integers (0,1,..., n_0 - 1).
/// 
/// If param_nu > n_0, an Err is returned, since such a graph cannot be generated.
/// Otherwise, the graph is returned as a Vec of pairs, where each pair is an edge between the nodes.
pub fn ba_edgelist(n_rand: usize, param_nu: usize, startedges: &[(usize,usize)]) -> Result<Vec<(usize, usize)>, &'static str>  {
	let mut edges: Vec<(usize,usize)> = Vec::with_capacity(startedges.len() + n_rand * param_nu);
	
	let mut start_node_count = 0;
	for e in startedges {
		start_node_count = *[start_node_count, e.0 + 1, e.1 + 1].iter().max().unwrap();
		edges.push(e.clone()); // expensive copy, but it should be fine for small start graph
	}
	
	if param_nu <= start_node_count {
		for i in 0..n_rand {
			for j in 0..param_nu {
				let mut newedge: (usize, usize) = (0, start_node_count + i);
				loop {
					let max_edge_index = startedges.len() + param_nu * i;
					let random_edge_index = rand::random_range(0..max_edge_index);
					let e: [usize;2] = [edges[random_edge_index].0, edges[random_edge_index].1];
					let random_node_index = rand::random_range(0..2);
					newedge.0 = e[random_node_index];

					if !&edges[startedges.len()+param_nu*i .. startedges.len() + param_nu*i+j].contains(&newedge) {
						// This edge is actually new
						break;
					}
				}
				
				edges.push(newedge);
			}
		}
		Ok(edges)
	} else {
		Err("param_nu <= n_0 required, where n_0 is the number of nodes in the given starting graph.")
	}
}
