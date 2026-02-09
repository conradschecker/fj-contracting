This is an implementation of the FJ contracting problem using the Rust programming languange.
It was used for making experiments about the performance of greedy algorihms described in the paper "Contract Design in Opinion Formation Games".

## Build and Run

The easiest way to run this application is to use [Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html), the package manager for Rust.
After cloning the repository, the project can be built using 
```sh
cargo build
``` 
For performance reasons, using the `--release` flag is highly recommended in order to enable compiler optimizations.

There are two main CLIs in this project.
- `fixed_graph` for simulations on a fixed graphs with varying numbers of influencers.
- `hrg` for simulations on hyperbolic random graphs (or any set of graphs graphs).

Additionally, there is a binary `bench` which provides a testing environment for identifying bottlenecks of the implementation.
All CLIs provide additional information about the required parameters using the `-h` flag.
Note that you can also build and run the CLIs directly using cargo, e.g., by 
```sh
cargo run --release --bin fixed_graph -- -h
```

There is an extensive documentation that can be built using 
```sh
cargo doc --no-deps
```

## Reproducing Results

As described in the paper, simulations were performed on several graphs from the [Networkrepository](https://networkrepository.com/) as well as on hyperbolic random graphs, for which we used the implementation from [chistopher/girgs](https://github.com/chistopher/girgs).
We provide a shell script that invokes their CLI `genhrg` for batch generation of hyperbolic random graphs of the same size.
Details can be found in the comments of the script (`./generate_hrg.sh`).

Once all necessary hyperbolic random graphs have been generated, the simulations can be started with the following command
```sh
mkdir -p ./results/

cargo run --release --bin hrg -- --input-directory ./graphs/ --output-directory ./results/ --action-set-generator BinaryP0Proportional --influencer-selector MostInfluential --influencer-count-param-a 1 --influencer-count-param-b 2 --n-min 256 --n-max 4096
```
This will run the simulations (with 128 repetitions) for graphs with sizes n = 256, 512, ..., 4096, using `BinaryP(1 - 1/n)` for generating action sets, and selecting `log(n) - 2` most influential agents as influencers in each run.
The necessary graphs have to be in the `./graphs/` directory.

You can also store fixed graphs in that directory.
For example, if you want to run the simulation on `socfb-Caltech36.mtx` in the directory `./graphs/`, you can run simulations by the following command:
```sh
mkdir -p ./results/

cargo run --release --bin fixed_graph -- --input-file ./graphs/socfb-Caltech36.mtx --output-directory ./results/ --action-set-generator GeometricFirstvalueZero --action-set-size 64 --influencer-selector MostInfluential
```
This will generate an action set using `GeometricFirstvalueZero(64)`, and select most influential agents as influencers.
By default, there will be 2 to 12 influencers, and for each influencer count, 128 repetitions will be made.

Note that simulations take some time already on relatively small graphs with ~1000 nodes.
By default, the simulations are parallelized.
