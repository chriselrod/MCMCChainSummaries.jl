module MCMCChainSummaries

using VectorizationBase, SIMDPirates#, PaddedMatrices
using ProbabilityModels
using Statistics, FFTW
using TypedTables, PrettyTables



struct MCMCChainSummary
    summary::Matrix{Float64}
    parameter_names::Vector{String}
    summary_header::Vector{String}
end

# function autocovariances!()

# end

function MCMCChainSummary(chains::AbstractArray{Float64,3}, parameter_names::Vector{String}, quantiles = (0.025, 0.25, 0.5, 0.75, 0.975))
    D, N, C = size(chains)
    chain_means = mean(chains, dims = 2)
    means, B = mean_and_var(reshape(chain_means, (D,C)), 2)
    # Calculat eall autocorrelations
    Np2 = VectorizationBase.nextpow2(N)
    padded_chains = Array{Float64,3}(undef, D, Np2, C)
    @inbounds for c in 1:C
        for n in 1:N
            @simd for d in 1:D
                padded_chains[d,n,c] = chains[d,n,c] - chain_means[d,c]
            end
        end
        for n in N+1:Np2, d in 1:D
            padded_chains[d,n,c] = 0.0
        end
    end
    FFTW.r2r!(padded_chains, FFTW.R2HC, dims = 2)
    Nh = Np2 >> 1
    @fastmath @inbounds for c in 1:C
        @simd for d in 1:D
            padded_chains[d,1,c] = abs2(padded_chains[d,1,c])
        end
        for n in 2:Nh
            Nc = Np2 + 2 - n
            @simd for d in 1:D
                padded_chains[d,n,c] = abs2(padded_chains[d,n,c]) + abs2(padded_chains[d,Nc,c])
                padded_chains[d,Nc,c] = 0.0
            end
        end
        @simd for d in 1:D
            padded_chains[d,1+Nh,c] = abs2(padded_chains[d,1+Nh,c])
        end
    end
    FFTW.r2r!(padded_chains, FFTW.HC2R, dims = 2)
    # const SUMMARY_HEADER = ["Parameter", "Mean", "St.Dev.", "MCMC SE", "ESS", "PSRF"]
    # Need to calculate mean of variances (W) -> PSRF = sqrt((N-1)/N + B/W); St.Dev. = W
    # Then calculate ESS
    # https://mc-stan.org/docs/2_20/reference-manual/notation-for-samples-chains-and-draws.html
    # https://mc-stan.org/docs/2_20/reference-manual/effective-sample-size-section.html
    # multithread this code??
end



end # module
