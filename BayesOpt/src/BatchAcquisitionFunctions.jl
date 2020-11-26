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

struct ThompsonSampleBatch 
	numPoints::Int
	ThompsonSampleBatch(;numPoints = 250) = new(numPoints)
end

struct ATSSampleBatch
	numPoints::Int
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
	ATSSampleBatch(m, k, s; numPoints=250, bounds_output_scale = (0.01, 2.0), bounds_length_scale = (0.01, 2.0), kwargs...) = new(numPoints, m, k, s, bounds_output_scale, bounds_length_scale, kwargs)
end


# Evaluates one score for  _all_ X in batch
function BatchAcquisitionScore(fn::MutualInformationPenalizedBatch, gp, X, tX, tY)
	α = vec(mapslices(x -> AcquisitionScore(fn.acquisitionFunction, gp, reshape(x, (size(x, 1), 1)), tX, tY), X; dims=1))
	g = (x1, x2) -> 0.5 * (log(2 * pi * ℯ * K(gp.kernel, x1, x1)) + log(2 * pi * ℯ * K(gp.kernel, x2, x2)) - log((2 * pi * ℯ)^2 * ((K(gp.kernel, x1, x1) * K(gp.kernel, x2, x2)) - (K(gp.kernel, x1, x2) * K(gp.kernel, x2, x1)))))
	MI = [g(x1, x2) for x1=eachcol(X), x2=eachcol(X) ]
	MI[diagind(MI)] .= 0 # disregard diagonal, which will be Inf
	sum(α) - fn.λ * sum(MI)
end


# Evaluates one score for  _all_ X in batch
function BatchAcquisitionScore(fn::CovariancePenalizedBatch, gp, X, tX, tY)
	α = vec(mapslices(x -> AcquisitionScore(fn.acquisitionFunction, gp, reshape(x, (size(x, 1), 1)), tX, tY), X; dims=1))
	C = Cov(gp, X)
	C[diagind(C)] .= 0 # disregard diagonal, which will be 1
	sum(α) - sum((fn.λ / size(X, 2)) * sum(C; dims=1))
end


# Evaluates one score for  _all_ X in batch
function BatchAcquisitionScore(fn::LocalPenalizedBatch, gp, X, tX, tY)
	μ∇ = (Mean(gp, X .+ 0.001) - Mean(gp, X .- 0.001)) ./ 0.002
	L = maximum(mapslices(norm, μ∇; dims=1))
	α = vec(mapslices(x -> AcquisitionScore(fn.acquisitionFunction, gp, reshape(x, (size(x, 1), 1)), tX, tY), X; dims=1))
	function rho(x1, x2, L)
		z = 1.0 / sqrt(2 * Cov(gp, reshape(x2, (size(x2, 1), 1)))[1,1]) * (L * norm(x1 .- x2) - maximum(tY) + Mean(gp, reshape(x2, (size(x2, 1), 1)))[1])
		0.5 * erfc(-z)
	end
	ρ = [rho(x1, x2, L) for x1=eachcol(X), x2=eachcol(X)]
	ρ[diagind(ρ)] .= 1.0 # disregard diagonal, since we shouldn't penalize ourselves
	sum(prod(ρ; dims=2) .* α)
end


# Evaluates n Thompson samples for the GP and returns the batch of optima
function AcquireBatch(fn::ThompsonSampleBatch, gp, lbounds, ubounds, tX, tY; restarts=20, batchSize=100, return_ys=false)
	seq = SobolSeq(lbounds, ubounds);
	X = hcat([next!(seq) for i = 1:fn.numPoints]...)
	dist = MvNormal(Mean(gp, X), Cov(gp, X))
	samples = rand(dist, batchSize)
	optima = mapslices(findmax, samples; dims=1)
	xs = zeros(Float32, size(lbounds, 1), batchSize); ys = zeros(Float32, batchSize);
	for (idx, data) in enumerate(optima)
		(v, i) = data
		xs[:, idx] = X[:, i]
		ys[idx] = v
	end
	if return_ys
		(xs, ys)
	else
		xs
	end
end


# Evaluates Thompson samples with varying length scales for the GP and returns the batch of optima
function AcquireBatch(fn::ATSSampleBatch, gp, lbounds, ubounds, tX, tY; restarts=20, batchSize=100, return_ys=false)
	seq = SobolSeq(lbounds, ubounds);
	X = hcat([next!(seq) for i = 1:fn.numPoints]...)
	dist_s = Uniform(fn.bounds_s[1], fn.bounds_s[2])
	dist_l = Uniform(fn.bounds_l[1], fn.bounds_l[2])
	xs = zeros(Float32, size(lbounds, 1), batchSize); ys = zeros(Float32, batchSize);
	for n in 1:batchSize
		kernel = typeof(fn.kernel)(;s=max(rand(dist_s), 1e-3), l=max(rand(dist_l), 1e-3), fn.kwargs...)
		gp2 = GaussianProcess(fn.mean, kernel, fn.sigma)
		dist = MvNormal(Mean(gp2, X), Cov(gp2, X) + 1e-5*I)
		sample = rand(dist)
		v, i = findmax(sample)
		xs[:, n] = X[:, i]
		ys[n] = v
	end
	if return_ys
		(xs, ys)
	else
		xs
	end
end

function AcquireBatch(fn, gp, lbounds, ubounds, tX, tY; restarts=20, batchSize=5)
	dist = Product(Uniform.(lbounds, ubounds))
	point = rand(dist, batchSize)
	best_f = Inf;
	f = x -> -1 * sum(BatchAcquisitionScore(fn, gp, reshape(x, (size(x, 1), batchSize)), tX, tY))
	for iter in 1:restarts
		initial_x = rand(dist, batchSize)
		result = optimize(
				  f,
				  hcat([lbounds for _ in 1:batchSize]...),
				  hcat([ubounds for _ in 1:batchSize]...),
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
	reshape(point, (size(point, 1), batchSize))
end
