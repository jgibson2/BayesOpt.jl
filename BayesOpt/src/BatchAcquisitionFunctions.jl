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


# Evaluates one score for  _all_ X in batch
function Acquire(fn::MutualInformationPenalizedBatch, gp, X, tX, tY)
	α = vec(mapslices(x -> Acquire(fn.acquisitionFunction, gp, reshape(x, (size(x, 1), 1)), tX, tY), X; dims=1))
	g = (x1, x2) -> 0.5 * (log(2 * pi * ℯ * K(gp.kernel, x1, x1)) + log(2 * pi * ℯ * K(gp.kernel, x2, x2)) - log((2 * pi * ℯ)^2 * ((K(gp.kernel, x1, x1) * K(gp.kernel, x2, x2)) - (K(gp.kernel, x1, x2) * K(gp.kernel, x2, x1)))))
	MI = [g(x1, x2) for x1=eachcol(X), x2=eachcol(X) ]
	MI[diagind(MI)] .= 0 # disregard diagonal, which will be Inf
	[sum(α) - fn.λ * sum(MI)]
end


# Evaluates one score for  _all_ X in batch
function Acquire(fn::CovariancePenalizedBatch, gp, X, tX, tY)
	α = vec(mapslices(x -> Acquire(fn.acquisitionFunction, gp, reshape(x, (size(x, 1), 1)), tX, tY), X; dims=1))
	C = Cov(gp, X)
	C[diagind(C)] .= 0 # disregard diagonal, which will be 1
	[sum(α) - sum((fn.λ / size(x, 2)) * sum(C; dims=1))]
end
