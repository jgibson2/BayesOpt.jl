using LinearAlgebra, SpecialFunctions, Distributions;

struct MutualInformationPenalizedBatch
	acquisitionFunction
	λ
	MutualInformationPenalizedBatch(ac; l=1.0) = new(ac, l)
end

struct CovariancePenalizedBatch
	acquisitionFunction
	λ
	CovariancePenalizedBatch(ac; l=1.0) = new(ac, l)
end

struct LocalPenalizedBatch
	acquisitionFunction
end

struct ThompsonSampler end

struct ATSSampler
	mean::Union{ZeroMean,ConstantMean,FunctionMean}
	kernel::Union{SquaredExponential,
		RationalQuadratic,
		Matern,
		Matern12,
		Matern32,
		Matern52}
	sigma::Float32
	bounds_s::Tuple{Float32, Float32}
	bounds_l::Tuple{Float32, Float32}
	kwargs::Any
	ATSSampler(m, k, s; bounds_output_scale = (0.01, 2.0), bounds_length_scale = (0.01, 2.0), kwargs...) = new(m, k, s, bounds_output_scale, bounds_length_scale, kwargs)
end


# Evaluates one score for  _all_ X in batch
function AcquireScore(fn::MutualInformationPenalizedBatch, gp, X, tX, tY)
	α = vec(mapslices(x -> AcquireScore(fn.acquisitionFunction, gp, reshape(x, (size(x, 1), 1)), tX, tY), X; dims=1))
	g = (x1, x2) -> 0.5 * (log(2 * pi * ℯ * K(gp.kernel, x1, x1)) + log(2 * pi * ℯ * K(gp.kernel, x2, x2)) - log((2 * pi * ℯ)^2 * ((K(gp.kernel, x1, x1) * K(gp.kernel, x2, x2)) - (K(gp.kernel, x1, x2) * K(gp.kernel, x2, x1)))))
	MI = [g(x1, x2) for x1=eachcol(X), x2=eachcol(X) ]
	MI[diagind(MI)] .= 0 # disregard diagonal, which will be Inf
	[sum(α) - fn.λ * sum(MI)]
end


# Evaluates one score for  _all_ X in batch
function AcquireScore(fn::CovariancePenalizedBatch, gp, X, tX, tY)
	α = vec(mapslices(x -> AcquireScore(fn.acquisitionFunction, gp, reshape(x, (size(x, 1), 1)), tX, tY), X; dims=1))
	C = Cov(gp, X)
	C[diagind(C)] .= 0 # disregard diagonal, which will be 1
	[sum(α) - sum((fn.λ / size(X, 2)) * sum(C; dims=1))]
end


# Evaluates one score for  _all_ X in batch
function AcquireScore(fn::LocalPenalizedBatch, gp, X, tX, tY)
	μ∇ = (Mean(gp, X .+ 0.001) - Mean(gp, X .- 0.001)) ./ 0.002
	L = maximum(mapslices(norm, μ∇; dims=1))
	α = vec(mapslices(x -> AcquireScore(fn.acquisitionFunction, gp, reshape(x, (size(x, 1), 1)), tX, tY), X; dims=1))
	function rho(x1, x2, L)
		z = 1.0 / sqrt(2 * Cov(gp, reshape(x2, (size(x2, 1), 1)))[1,1]) * (L * norm(x1 .- x2) - maximum(tY) + Mean(gp, reshape(x2, (size(x2, 1), 1)))[1])
		0.5 * erfc(-z)
	end
	ρ = [rho(x1, x2, L) for x1=eachcol(X), x2=eachcol(X)]
	ρ[diagind(ρ)] .= 1.0 # disregard diagonal, since we shouldn't penalize ourselves
	sum(prod(ρ; dims=2) .* α)
end


# Evaluates n Thompson samples for the GP and returns the batch of optima
function SampleOptima(sp::ThompsonSampler, gp, X; num_samples=100)
	dist = MvNormal(Mean(gp, X), Cov(gp, X))
	samples = rand(dist, num_samples)
	optima = mapslices(findmax, samples; dims=1)
	xs = []; ys = [];
	for (v, i) in optima
		append!(xs, X[:, i])
		append!(ys, v)
	end
	(xs, ys)
end


# Evaluates Thompson samples with varying length scales for the GP and returns the batch of optima
function SampleOptima(sp::ATSSampler, gp, X; num_samples=20)
	dist_s = Uniform(sp.bounds_s[1], sp.bounds_s[2])
	dist_l = Uniform(sp.bounds_l[1], sp.bounds_l[2])
	xs = []; ys = [];
	for n in 1:num_samples
		kernel = typeof(sp.kernel)(;s=max(rand(dist_s), 1e-3), l=max(rand(dist_l), 1e-3), sp.kwargs...)
		gp2 = GaussianProcess(sp.mean, kernel, sp.sigma)
		dist = MvNormal(Mean(gp2, X), Cov(gp2, X) + 1e-5*I)
		sample = rand(dist)
		v, i = findmax(sample)
		append!(xs, X[:, i])
		append!(ys, v)
	end
	(xs, ys)
end
