module TestBINOCULARS

using Pkg, Random, Plots, Distributions, LinearAlgebra, SpecialFunctions, LaTeXStrings, StatsBase;
include("./BayesOpt/src/BayesOpt.jl")
using .BayesOpt;
# Pkg.develop(path="./BayesOpt");
# using BayesOpt;

n = 250;
x = reshape(LinRange(0, 1, n), 1, :)
f = x -> tan.(0.5 .* x) - cos.(13 .* x) .+ (x .+ 0.5).^3 .* erf.(sin.(x))
gp = GaussianProcess(ConstantMean(mean(f(x))), Matern52(), sqrt(1e-5));

# acMES = MutualInformationMES(gp, [0.0], [1.0])
# acMIBatchMES = MutualInformationPenalizedBatch(acMES)
acMES = ExpectedImprovement()
acMIBatchMES = LocalPenalizedBatch(acMES);

data = BatchOptimizationData(gp.mean, gp.kernel, gp.sigma, acMIBatchMES, [0.0], [1.0]) 
anim = @animate for i in 1:20
	pts = ProposeNextBatch(data; batchSize=5, restarts=1)
	println("Proposed: $(pts) Objective: $(f(pts))")
	pt = reshape(pts[:, 1], (size(pts, 1), 1))
	if(data.tX != nothing)
		w = vec(AcquisitionScore(acMES, gp, pts, data.tX, data.tY))
		bestA, bestIdx = findmax(w)
		pt = reshape(pts[:, bestIdx], (size(pts, 1), 1))
	end
	y = f(pt)
	data.tX = data.tX == nothing ? pt : hcat(data.tX, pt)
	data.tY = data.tY == nothing ? vec(y) : vcat(data.tY, vec(y))
	global gp = ConditionGP(GaussianProcess(data.mean, data.kernel, data.sigma), data.tX, data.tY);
	std = Std(gp, x);
	
	p2 = plot(title="Bayesian Optimization Test With BINOCULARS (Observation $(i))", legend=:bottomright);
	plot!(p2, vec(x), Mean(gp, x), ribbon=((1.96 * std),(1.96 * std)), linewidth=2, label=L"\mu");
	scatter!(p2, vec(data.tX), data.tY, label="Observations");
	plot!(p2, vec(x), vec(f(x)), label=L"f(x)", linecolor=:red);
	# global acMES = MutualInformationMES(gp, [0.0], [1.0])
	# global acMIBatchMES = MutualInformationPenalizedBatch(acMES)
end

gif(anim, "testBINOCULARS.gif", fps=2)

end
