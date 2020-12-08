module TestBINOCULARS

using Pkg, Random, Plots, Distributions, LinearAlgebra, SpecialFunctions, LaTeXStrings, StatsBase;
include("./BayesOpt/src/BayesOpt.jl")
using .BayesOpt;
# Pkg.develop(path="./BayesOpt");
# using BayesOpt;

n = 250;
x = reshape(LinRange(0, 1, n), 1, :)
f = x -> tan.(0.5 .* x) - cos.(13 .* x) .+ (x .+ 0.5).^3 .* erf.(sin.(x))
gp = GaussianProcess(ConstantMean(mean(f(x))), Matern52(l=0.1), sqrt(1e-5));


acMES = MutualInformationMES(gp, [0.0], [1.0])
acMIBatchMES = MutualInformationPenalizedBatch(acMES)

println("Method: MI Penalized Batch MES")
data = BatchOptimizationData(gp.mean, gp.kernel, gp.sigma, acMIBatchMES, [0.0], [1.0]) 
anim = @animate for i in 1:10
	l = @layout[a ; b]
	p2 = plot(title="MI Penalized Batch w/ MES BINOCULARS (Observation $(i))", legend=:bottomright, layout=l);
	plot!(p2[2], vec(x), AcquisitionScore(acMES, gp, x, data.tX, data.tY), title="Acquisition Function", label="MES", color=:green)
	pts = ProposeNextBatch(data; batchSize=5, restarts=1)
	println("Proposed: $(pts) Objective: $(f(pts))")
	println("f* samples: $(acMES.fstar_samples)")
	pt = reshape(pts[:, 1], (size(pts, 1), 1))
	if(data.tX != nothing)
		w = vec(AcquisitionScore(acMES, gp, pts, data.tX, data.tY))
		println("Acquisition scores: $(w)")
		bestA, bestIdx = findmax(w)
		pt = reshape(pts[:, bestIdx], (size(pts, 1), 1))
	end
	y = f(pt)
	data.tX = data.tX == nothing ? pt : hcat(data.tX, pt)
	data.tY = data.tY == nothing ? vec(y) : vcat(data.tY, vec(y))
	global gp = ConditionGP(GaussianProcess(data.mean, data.kernel, data.sigma), data.tX, data.tY);
	std = Std(gp, x);
	
	plot!(p2[1], vec(x), Mean(gp, x), ribbon=((1.96 * std),(1.96 * std)), linewidth=2, label=L"\mu");
	scatter!(p2[1], vec(data.tX), data.tY, label="Observations");
	plot!(p2[1], vec(x), vec(f(x)), label=L"f(x)", linecolor=:red);
	global acMES = MutualInformationMES(gp, [0.0], [1.0])
	global acMIBatchMES = MutualInformationPenalizedBatch(acMES)
	data.batchAcquisitionFunction = acMIBatchMES
end

gif(anim, "testBINOCULARS_MES.gif", fps=1.5)

gp = GaussianProcess(ConstantMean(mean(f(x))), Matern52(l=0.1), sqrt(1e-5));
acEI = ExpectedImprovement()
acLPBatchEI = LocalPenalizedBatch(acEI);
println("Method: Local Penalized Batch EI")
data = BatchOptimizationData(gp.mean, gp.kernel, gp.sigma, acLPBatchEI, [0.0], [1.0]) 
anim = @animate for i in 1:10
	l = @layout[a ; b]
	p2 = plot(title="Local Penalized Batch w/ EI BINOCULARS (Observation $(i))", legend=:bottomright, layout=l);
	plot!(p2[2], vec(x), AcquisitionScore(acEI, gp, x, data.tX, data.tY), title="Acquisition Function", label="EI", color=:green)
	pts = ProposeNextBatch(data; batchSize=5, restarts=5)
	println("Proposed: $(pts) Objective: $(f(pts))")
	pt = reshape(pts[:, 1], (size(pts, 1), 1))
	if(data.tX != nothing)
		w = vec(AcquisitionScore(acEI, gp, pts, data.tX, data.tY))
		println("Acquisition scores: $(w)")
		bestA, bestIdx = findmax(w)
		pt = reshape(pts[:, bestIdx], (size(pts, 1), 1))
	end
	y = f(pt)
	data.tX = data.tX == nothing ? pt : hcat(data.tX, pt)
	data.tY = data.tY == nothing ? vec(y) : vcat(data.tY, vec(y))
	global gp = ConditionGP(GaussianProcess(data.mean, data.kernel, data.sigma), data.tX, data.tY);
	std = Std(gp, x);
	
	plot!(p2[1], vec(x), Mean(gp, x), ribbon=((1.96 * std),(1.96 * std)), linewidth=2, label=L"\mu");
	scatter!(p2[1], vec(data.tX), data.tY, label="Observations");
	plot!(p2[1], vec(x), vec(f(x)), label=L"f(x)", linecolor=:red);
end

gif(anim, "testBINOCULARS_EI.gif", fps=1.5)

gp = GaussianProcess(ConstantMean(mean(f(x))), Matern52(l=0.1), sqrt(1e-5));
acUCB = UpperConfidenceBound()
acCVBatchUCB = CovariancePenalizedBatch(acUCB);
println("Method: Local Penalized Batch EI")
data = BatchOptimizationData(gp.mean, gp.kernel, gp.sigma, acLPBatchEI, [0.0], [1.0]) 
anim = @animate for i in 1:10
	l = @layout[a ; b]
	p2 = plot(title="Cov Penalized Batch w/ UCB BINOCULARS (Observation $(i))", legend=:bottomright, layout=l);
	plot!(p2[2], vec(x), AcquisitionScore(acUCB, gp, x, data.tX, data.tY), title="Acquisition Function", label="UCB", color=:green)
	pts = ProposeNextBatch(data; batchSize=5, restarts=5)
	println("Proposed: $(pts) Objective: $(f(pts))")
	pt = reshape(pts[:, 1], (size(pts, 1), 1))
	if(data.tX != nothing)
		w = vec(AcquisitionScore(acEI, gp, pts, data.tX, data.tY))
		println("Acquisition scores: $(w)")
		bestA, bestIdx = findmax(w)
		pt = reshape(pts[:, bestIdx], (size(pts, 1), 1))
	end
	y = f(pt)
	data.tX = data.tX == nothing ? pt : hcat(data.tX, pt)
	data.tY = data.tY == nothing ? vec(y) : vcat(data.tY, vec(y))
	global gp = ConditionGP(GaussianProcess(data.mean, data.kernel, data.sigma), data.tX, data.tY);
	std = Std(gp, x);
	
	plot!(p2[1], vec(x), Mean(gp, x), ribbon=((1.96 * std),(1.96 * std)), linewidth=2, label=L"\mu");
	scatter!(p2[1], vec(data.tX), data.tY, label="Observations");
	plot!(p2[1], vec(x), vec(f(x)), label=L"f(x)", linecolor=:red);
	b = 2.0 * log((i+4)^2 * pi^2 / 60)
	global acUCB = UpperConfidenceBound(beta=b)
	global acCVBatchUCB = CovariancePenalizedBatch(acUCB)
	data.batchAcquisitionFunction = acCVBatchUCB
end

gif(anim, "testBINOCULARS_UCB.gif", fps=1.5)
end
