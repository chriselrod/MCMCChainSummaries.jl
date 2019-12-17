module MCMCChainSummaries

using VectorizationBase, SIMDPirates
using Statistics, FFTW
using PrettyTables

export MCMCChainSummary

include("quantiles.jl")
include("mean_and_variance.jl")
include("summaries.jl")
include("precompile.jl")
_precompile_()

end # module
