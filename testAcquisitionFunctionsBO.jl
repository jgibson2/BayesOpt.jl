module TestAcquisitionFunctionsBO

using Pkg, Random, Plots, Distributions, LinearAlgebra, QuadGK, LaTeXStrings;
include("./BayesOpt/src/BayesOpt.jl")
using .BayesOpt;
# Pkg.develop(path="./BayesOpt");
# using BayesOpt;

gp = GaussianProcess(ZeroMean(), Matern52(), 0.001);
n = 250;
x = reshape(LinRange(0, 50, n), 1, :)
g = X -> sum((0.25 * sin.(0.25 * X)) + (0.75 * cos.(X)) + 3 * cos.(pi * (X .- 25) / 50); dims=1)
f = X -> g(X) .- (sum(g(x)) / n)

acEI   = ExpectedImprovement()
acKG   = KnowledgeGradientCP()
acUCB  = UpperConfidenceBound(beta=4.0)
acPI   = ProbabilityOfImprovement(tau=0.5)
acMES  = MutualInformationMES(gp, [0.0], [50.0])
acOPES = MutualInformationOPES(gp, [0.0], [50.0])
acMIBatchEI = MutualInformationPenalizedBatch(acEI)
acCVBatchEI = CovariancePenalizedBatch(acEI)
acLocalBatchEI = LocalPenalizedBatch(acEI)
acThompson = ThompsonSampleBatch()
acATS = ATSSampleBatch(gp.mean, gp.kernel, gp.sigma; bounds_length_scale=(0.5, 2.5))


acfns = [
	 ("EI", acEI),
	 ("KGCP", acKG),
	 ("UCB (B=4.0)", acUCB),
	 ("PoI (t=0.5)", acPI),
	 ("MES", acMES),
	 ("OPES", acOPES),
];

batchacfns = [
	 ("MI Penalized Batch", acMIBatchEI, 5, 1),
	 ("Covariance Penalized Batch", acCVBatchEI, 5, 1),
	 ("Local Penalized Batch", acLocalBatchEI, 5, 1),
	 ("Thompson Sample Batch", acThompson, 10, 0),
	 ("ATS Sample Batch", acATS, 10, 0),
];

for (lab,fn) in acfns
	data = OptimizationData(gp.mean, gp.kernel, gp.sigma, fn, [0.0], [50.0]) 
	println("Method: $(lab)")
	anim = @animate for i in 1:20
		local pt = ProposeAndEvaluateNextPoint!(data, f)
		println("Proposed: $(pt) Objective: $(f(pt))")
		gpi = ConditionGP(GaussianProcess(data.mean, data.kernel, data.sigma), data.tX, data.tY);
		std = Std(gpi, x);
		
		p2 = plot(title="Bayesian Optimization Test With $(lab)");
		plot!(p2, vec(x), Mean(gpi, x), ribbon=((1.96 * std),(1.96 * std)), linewidth=2, label=L"\mu");
		scatter!(p2, vec(data.tX), data.tY, label="Observations");
		plot!(p2, vec(x), vec(f(x)), label=L"f(x)", linecolor=:red);
		global acMES  = MutualInformationMES(gpi, [0.0], [50.0])
		global acOPES = MutualInformationOPES(gpi, [0.0], [50.0])
	end

	gif(anim, "testBO_$(lab).gif", fps=2)
end

for (lab,fn,bs,r) in batchacfns
	data = BatchOptimizationData(gp.mean, gp.kernel, gp.sigma, fn, [0.0], [50.0]) 
	println("Method: $(lab)")
	anim = @animate for i in 1:5
		local pt = ProposeAndEvaluateNextBatch!(data, f; restarts=r, batchSize=bs)
		println("Proposed: $(pt) Objective: $(f(pt))")
		gpi = ConditionGP(GaussianProcess(data.mean, data.kernel, data.sigma), data.tX, data.tY);
		std = Std(gpi, x);
		
		p2 = plot(title="Batch Bayesian Optimization Test With $(lab)");
		plot!(p2, vec(x), Mean(gpi, x), ribbon=((1.96 * std),(1.96 * std)), linewidth=2, label=L"\mu");
		scatter!(p2, vec(data.tX), data.tY, label="Observations");
		plot!(p2, vec(x), vec(f(x)), label=L"f(x)", linecolor=:red);
	end

	gif(anim, "testBO_$(lab).gif", fps=1)
end

end
