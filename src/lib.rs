/// Implementation of the Friedkin Johnson model for opinion dynamics.
pub mod friedkin_johnson;
/// Implementation of the FJ contracting model.
pub mod fj_contract;

/// Helper functions for loading and converting graphs.
pub mod graphloader;
/// Simple random graph generators, mainly for testing purposes.
pub mod graphgenerator;
/// Generators of random instances of the FJ contracting problem.
pub mod instancegenerator;
/// Solving instances of the FJ contracting problem using different heuristics.
pub mod instancesolver;
/// Wrapper for repeatedly generating and solving instances of the FJ contracting problem with statistical evaluation.
pub mod instancesimulator;

/// Statistical evaluation.
pub mod statistics;

