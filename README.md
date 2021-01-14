## Usage

```julia
using BayesOpt;

# define an objective function
f = X -> sum((0.25 * sin.(0.25 * X)) + (0.75 * cos.(X)) + 3 * cos.(pi * (X .- 25) / 50); dims=1)

# Make a GP with a zero mean, Matern 5/2 kernel, and 0.001 as the noise std dev
gp = GaussianProcess(ZeroMean(), Matern52(), 0.001);

# choose an acquisition function
acEI   = ExpectedImprovement()

# or, choose a batch acquisition function
acLocalBatchEI = LocalPenalizedBatch(acEI)

# set up the optimization data, including the bounds
data = OptimizationData(gp.mean, gp.kernel, gp.sigma, acEI, [0.0], [50.0])

for i in 1:20
		local pt = ProposeAndEvaluateNextPoint!(data, f)
		println("Proposed: $(pt) Objective: $(f(pt))")
end

# get optimization data
println(data.tX)
println(data.ty)

# do the same with a batch
batch_data = BatchOptimizationData(gp.mean, gp.kernel, gp.sigma, acLocalBatchEIfn, [0.0], [50.0])

for i in 1:20
		local pts = ProposeAndEvaluateNextBatch!(batch_data, f; batchSize=5)
		println("Proposed: $(pts) Objective: $(f(pts))")
end

println(batch_data.tX)
println(batch_data.ty)


```
