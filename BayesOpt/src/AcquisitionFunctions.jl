using LinearAlgebra, SpecialFunctions, Distributions, QuadGK, Optim, Sobol, Roots;

struct ExpectedImprovement
	xi
	ExpectedImprovement(;xi=0.0) = new(xi)
end

struct KnowledgeGradientCP
	xi
	KnowledgeGradientCP(;xi=0.0) = new(xi)
end

struct UpperConfidenceBound
	beta
	UpperConfidenceBound(;beta=4.0) = new(beta)
end

struct ProbabilityOfImprovement
	tau
	eps
	ProbabilityOfImprovement(;tau=0.01, eps=1e-5) = new(tau, eps)
end

struct MutualInformationMES
	fstar_samples
	function MutualInformationMES(gp, lbounds, ubounds; samples = 250, quantiles = 20)
		seq = SobolSeq(lbounds, ubounds);
		points = hcat([next!(seq) for i = 1:samples]...)
		mus = Mean(gp, points)
		mu_avg = sum(mus) / size(mus, 1)
		mu_max, mu_max_idx = findmax(mus)
		sigmas = Std(gp, points)
		quants = LinRange(0.01, 0.99, quantiles)
		n = Normal();
		fstar_cdf = (z, u) -> sum(log.(clamp.(cdf(n, (z .- mus) ./ sigmas), 1e-5, 1e5))) - log(u);
		function f(u)
			try
				find_zero(z -> fstar_cdf(z, u), mu_avg, Order1())
			catch e
				invlogcdf(Normal(mu_max, sigmas[mu_max_idx]), log(u))
			end
		end
		fstar_samples = [ f(u) for u=quants ]
		new(fstar_samples)
	end
end

struct MutualInformationOPES
	fstar_samples
	function MutualInformationOPES(gp, lbounds, ubounds; samples = 250, quantiles = 20)
		seq = SobolSeq(lbounds, ubounds);
		points = hcat([next!(seq) for i = 1:samples]...)
		mus = Mean(gp, points)
		mu_avg = sum(mus) / size(mus, 1)
		mu_max = maximum(mus)
		sigmas = Std(gp, points)
		quants = LinRange(0.01, 0.99, quantiles)
		n = Normal();
		fstar_cdf = (z, u) -> sum(log.(clamp.(cdf(n, (z .- mus) ./ sigmas), 1e-5, 1e5))) - log(u);
		function f(u)
			try
				find_zero(z -> fstar_cdf(z, u), mu_avg, Order1())
			catch e
				invlogcdf(Normal(mu_max, sigmas[mu_max_idx]), log(u))
			end
		end
		fstar_samples = [ f(u) for u=quants ]
		new(fstar_samples)
	end
end

function AcquisitionScore(fn::ExpectedImprovement, gp, X, tX, tY)
	if tY == nothing
		tY = Mean(gp, X)
		tX = X
	end
	bestY, bestYIdx = findmax(tY)
	bestX = reshape(tX[:, bestYIdx], (size(tX)[1], 1))
	bestPhi = Mean(gp, bestX)
	us = Mean(gp, X)
	sigmas = Std(gp, X)
	r = us .- (bestPhi .+ fn.xi)
	Z = r ./ (sigmas .+ 1e-8)
	(r .* cdf(Normal(), Z)) .+ (sigmas .* pdf(Normal(), Z))
end


function AcquisitionScore(fn::KnowledgeGradientCP, gp, X, tX, tY)
	if tY == nothing
		tY = Mean(gp, X)
		tX = X
	end
	bestY, bestYIdx = findmax(tY)
	bestX = reshape(tX[:, bestYIdx], (size(tX)[1], 1))
	bestPhi = Mean(gp, bestX)
	us = Mean(gp, X)
	us_tX = Mean(gp, tX)
	sigmas = Std(gp, X)
	r = us .- (bestPhi .+ fn.xi)
	Z = r ./ (sigmas .+ 1e-8)
	ei = (r .* cdf(Normal(), Z)) .+ (sigmas .* pdf(Normal(), Z))
	ei .- max.(us .- maximum(us_tX), 0)
end


function AcquisitionScore(fn::UpperConfidenceBound, gp, X, tX, tY)
	Mean(gp, X) + (fn.beta .* Std(gp, X))
end


function AcquisitionScore(fn::ProbabilityOfImprovement, gp, X, tX, tY)
	bestY, bestYIdx = findmax(tY)
	bestX = reshape(tX[:, bestYIdx], (size(tX)[1], 1))
	r = max(maximum(tY) - minimum(tY), fn.eps)
	mu = Mean(gp, bestX)
	tau = (fn.tau * r) .+ mu
	cdf(Normal(), (Mean(gp, X) .- tau) ./ Std(gp, X))
end


function AcquisitionScore(fn::MutualInformationMES, gp, X, tX, tY)
	mus_X = Mean(gp, X)
	sigmas_X = Std(gp, X)
	n = Normal()
	Z = hcat([(fn.fstar_samples .- u) ./ s for (u,s)=zip(mus_X, sigmas_X)]...)
	mes = Z .* pdf(n, Z) ./ cdf(n, Z) - log.(cdf(n, Z)).^2
	res = (0.5 / size(fn.fstar_samples, 1)) .* mapslices(sum, mes; dims=1)
	vec(res)
end


function AcquisitionScore(fn::MutualInformationOPES, gp, X, tX, tY)
	mus_X = Mean(gp, X)
	sigmas_X = Std(gp, X)
	n = Normal()
	Z = hcat([(fn.fstar_samples .- u) ./ s for (u,s)=zip(mus_X, sigmas_X)]...)
	Q = pdf(n, Z) ./ cdf(n, Z)
	R = 1.0 .- (Z .* Q) .- (Q.^2)
	S = sqrt.(mapslices(c -> sigmas_X.^2 .* c, R; dims=2) .+ gp.sigma^2)
	T = vec(mapslices(sum, log.(S); dims=1))
	U = log.(sqrt.(sigmas_X.^2 .+ gp.sigma^2))
	U - (1.0 / size(fn.fstar_samples, 1) .* T)
end


function AcquirePoint(fn, gp, lbounds, ubounds, tX, tY; restarts=20)
	dist = Product(Uniform.(lbounds, ubounds))
	point = rand(dist, 1)
	best_f = Inf;
	f = x -> -1 * sum(AcquisitionScore(fn, gp, reshape(x, (size(x, 1), 1)), tX, tY))
	for iter in 1:restarts
		initial_x = rand(dist, 1)
		result = optimize(
				  f,
				  lbounds,
				  ubounds,
				  initial_x,
				  Fminbox(LBFGS()),
				  Optim.Options(show_trace=false,
						 f_tol=1e-4,
						 x_tol=1e-4,
						 g_tol=1e-4,
						 iterations=50))
		if best_f > Optim.minimum(result)
			point = Optim.minimizer(result)
			best_f = Optim.minimum(result)
		end
	end
	reshape(point, (size(point, 1), 1))
end
