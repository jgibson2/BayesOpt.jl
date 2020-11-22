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
	[sum(α) - sum((fn.λ / size(x, 2)) * sum(C; dims=1))]
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
