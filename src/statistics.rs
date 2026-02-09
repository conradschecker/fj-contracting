/// Store statistical information as a boxplot.
pub struct BoxPlot<T> {
	min: T,
	max: T,
	median: T,
	lower_quartile: T,
	upper_quartile: T,
	lower_whisker: T,
	upper_whisker: T,
	outliers: Vec<T>,
}

/// Store statistical information as a histogram.
pub struct Histogram<T> {
	min: T,
	max: T,
	total: T,
	distribution: Vec<usize>,
	outlier_count: usize,
	threshold: usize,
}

impl<T: std::fmt::Display + std::fmt::Debug> std::fmt::Display for BoxPlot<T> {
	/// Obtain a single-line that contains the boxplot information.
	/// Useful for writing the results of a statistical experiment to a file.
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {} {} {} {} {} {}", self.min, self.max, self.median, self.lower_quartile, self.upper_quartile, self.lower_whisker, self.upper_whisker)
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for BoxPlot<T> {
	/// Obtain verbose information about a boxplot (for debugging purposes).
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "min: {:?}, max: {:?}, median: {:?}, lower_quartile: {:?}, upper_quartile: {:?}, lower_whisker: {:?}, upper_whisker: {:?}, outliers: {:?}", self.min, self.max, self.median, self.lower_quartile, self.upper_quartile, self.lower_whisker, self.upper_whisker, self.outliers)
    }
}

impl<T: std::fmt::Display + std::fmt::Debug> std::fmt::Display for Histogram<T> {
	/// Write a single line that contains the minimum, the maximum, and the total sum.
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "min: {:?}, max: {:?}, total: {:?}, outlier_count: {}", self.min, self.max, self.total, self.outlier_count)
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for Histogram<T> {
	/// Write a single line that contains the minimum, the maximum, the total sum and the distributional information.
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "min: {:?}, max: {:?}, total {:?}, outlier_count: {}, distribution {:?}", self.min, self.max, self.total, self.outlier_count, self.distribution)
    }
}

impl BoxPlot<f32> {
	/// Create a [`BoxPlot`] from a history of (simulation) results.
	///
	/// The history is given as a [`Vec`] of floating point numbers.
	/// It computes minimum, maximum, median, first (lower) and third (upper) quartiles and whiskers.
	/// The upper whisker contains all values that have a distance of at most 1.5 IQR to the third quartile.
	/// The lower whisker contains all values that have a distance of at most 1.5 IQR to the first quartile.
	///
	/// Panics if the vector is empty.
	///
	/// # Example
	/// ``` 
	/// use fj_contracting::statistics::BoxPlot;
	///
	/// // Specify a history.
	/// let history = vec![14., 12., 9., 15., 7., 4., 28., 3., 28.];
	///
	/// // Obtain the boxplot from this history.
	/// let bplot = BoxPlot::from_nonempty_f32_history_vec(history);
	///
	/// let bplot_output = format!("{bplot:?}");
	/// let expected_output = "min: 3.0, max: 28.0, median: 12.0, lower_quartile: 7.0, upper_quartile: 15.0, lower_whisker: 3.0, upper_whisker: 15.0, outliers: [28.0, 28.0]";
	/// assert_eq!(bplot_output, expected_output);
	/// assert_eq!(*bplot.outliers(), vec![28., 28.]);
	/// ```
	pub fn from_nonempty_f32_history_vec(mut vec: Vec<f32>) -> BoxPlot<f32> {
		vec.sort_by(|x, y| x.total_cmp(y));
		let min = vec[0];
		let max = vec[vec.len()-1];
		let (median, lower_quartile, upper_quartile) = match vec.len() % 2 {
			0 => {
				(0.5 * (vec[vec.len() / 2 - 1] + vec[vec.len() / 2]),
				0.5 * (vec[vec.len() / 4 - 1] + vec[vec.len() / 4]),
				0.5 * (vec[vec.len() * 3 / 4 - 1] + vec[vec.len() *3 / 4]))
			},
			1 => {
				(vec[vec.len() / 2],
				vec[vec.len() / 4],
				vec[ vec.len() * 3 / 4])
			},
			_ => panic!("{} modulo 2 should not be greater or equal 2.", vec.len())
		};
		let fifty_percent_range = upper_quartile - lower_quartile;
		let lower_whisker = vec
			.iter()
			.filter(|x| lower_quartile - *x <= 1.5 * fifty_percent_range)
			.fold(lower_quartile as f32, |acc, e| acc.min(*e));
		let upper_whisker = vec
			.iter()
			.filter(|x| *x - upper_quartile <= 1.5 * fifty_percent_range)
			.fold(upper_quartile as f32, |acc, e| acc.max(*e));
		let outliers = vec
			.into_iter()
			.filter(|x| lower_quartile - *x > 1.5 * fifty_percent_range || *x - upper_quartile > 1.5 * fifty_percent_range)
			.collect::<Vec<f32>>();
		
		BoxPlot { min, max, median, lower_quartile, upper_quartile, lower_whisker, upper_whisker, outliers }
	}
	
	pub fn outliers(&self) -> &Vec<f32> {
		&self.outliers
	}
}

impl Histogram<usize> {
	/// Create a [`Histogram`] from a history of (simulation) results and a threshold.
	///
	/// - The history of results is a [`Vec`] of integers of type [`usize`].
	/// - The threshold is the maximum integer until values are considered as outliers.
	///
	/// For all integers in `0..threshold`, the number of occurences in the history is computed.
	///
	/// Panics if the vector is empty.
	///
	/// # Example
	/// ```
	/// use fj_contracting::statistics::Histogram;
	///
	/// // Specify a history.
	/// let history = vec![2, 2, 8, 1, 3, 3, 3, 6, 2, 5, 5];
	///
	/// // Create a histogram from this history with threshold 5.
	/// let histogram = Histogram::from_nonempty_usize_history_vec(history, 5);
	///
	/// let histogram_output = format!("{histogram}");
	/// // The minimum is 1, the maximum is 8, the total value is 40, and
	/// // there are two values that are strictly bigger than the threshold.
	/// assert_eq!(histogram_output, "min: 1, max: 8, total: 40, outlier_count: 2");
	///
	/// // In the history, 0 did not occur, 1 occured once, 2 occured three times, ... 
	/// assert_eq!(*histogram.distribution(), vec![0,1,3,3,0,2]);
	/// ```
	pub fn from_nonempty_usize_history_vec(mut vec: Vec<usize>, threshold: usize) -> Histogram<usize> {
		vec.sort();
		let min = vec[0];
		let max = vec[vec.len()-1];
		let total = vec.iter().sum();
		
		let mut distribution: Vec<usize> = vec![0;threshold+1];
		let mut outlier_count = 0;
		for entry in vec {
			if entry <= threshold {
				distribution[entry] += 1;
			} else {
				outlier_count += 1;
			}
		}
		
		Histogram { min, max, total, distribution, outlier_count, threshold }
	}
	
	/// Return the threshold.
	pub fn threshold(&self) -> usize {
		self.threshold
	}
	
	/// Return the histogram (distribtion).
	pub fn distribution(&self) -> &Vec<usize> {
		&self.distribution
	}
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn boxplot_even_history_length() {
		let history = vec![14., 12., 9., 15., 7., 4., 28., 3.];
		let bplot = BoxPlot::from_nonempty_f32_history_vec(history);
		assert_eq!(bplot.min, 3.);
		assert_eq!(bplot.max, 28.);
		assert_eq!(bplot.median, 21./2.);
		assert_eq!(bplot.lower_quartile, 11./2.);
		assert_eq!(bplot.upper_quartile, 29./2.);
		assert_eq!(bplot.lower_whisker, 3.);
		assert_eq!(bplot.upper_whisker, 28.);
		assert_eq!(bplot.outliers, vec![]);
    }
    
    #[test]
    fn boxplot_tiny() {
		let history = vec![14.,];
		let bplot = BoxPlot::from_nonempty_f32_history_vec(history);
		assert_eq!(bplot.median, bplot.lower_whisker);
	}
}
