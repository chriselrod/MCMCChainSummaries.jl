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
    # Need to calculate mean of variances (W) -> PSRF = sqrt((N-1)/N + B/W); var = (N-1)*W/N + B
    # Then calculate ESS
    # https://mc-stan.org/docs/2_20/reference-manual/notation-for-samples-chains-and-draws.html
    # https://mc-stan.org/docs/2_20/reference-manual/effective-sample-size-section.html
    # multithread this code??
    W, Wshift = VectorizationBase.pick_vector_width_shift(Float64)
    V = VectorizationBase.pick_vector(Float64)
    Ddiv4Witer = D >> (Wshift + 2)
    PSRF = Vector{Float64}(undef, D); ptrpsrf = pointer(PSRF)
    stdev = Vector{Float64}(undef, D); ptrstdev = pointer(stdev)
    mcmcse = Vector{Float64}(undef, D); ptrmcmcse = pointer(mcmcse)
    ess = Vector{Float64}(undef, D); ptress = pointer(ess)
    ptrcov = pointer(padded_chains)
    ptrar = ptrcov
    ptrb = pointer(B)
    
    invdenom = vbroadcast(V, inv((N-1) * N2p))
    nfrac = vbroadcast(V, (N-1) / N )
    ninvn = vbroadcast(V, -1/N)
    invc = vbroadcast(V, 1/C)
    Nh = N >> 1
    for _ in 1:Ddiv4Witer
        Base.Cartesian.@nexprs 4 j -> begin
            Wvar_j = vmul(invdenom, vload(V, ptrcov))
            ptrcov += VectorizationBase.REGISTER_SIZE
        end
        Base.Cartesian.@nexprs 4 j -> begin
            var_j = vmuladd(nfrac, Wvar_j, vload(V, ptrb))
            ptrb += VectorizationBase.REGISTER_SIZE
        end
        Base.Cartesian.@nexprs 4 j -> begin
            r̂²_j = vfdiv(var_j, Wvar_j)
            sqrt_j = vsqrt(var_j)
            rhat_j = vsqrt(r̂²_j)
        end
        Base.Cartesian.@nexprs 4 j -> begin
            vstore!(ptrstdev, sqrt_j)
            ptrstdev += VectorizationBase.REGISTER_SIZE
            vstore!(ptrpsrf, rhat_j)
            ptrpsrf += VectorizationBase.REGISTER_SIZE
        end
        Base.Cartesian.@nexprs 4 j -> begin
            mask_j = VectorizationBase.max_mask(Float64)
            for n in 0:Nh-1
                ρ₊_j = vload(V, ptrar + (2n  )*8D)
                ρ₋_j = vload(V, ptrar + (2n+1)*8D)
                for c in 1:C-1
                    ρ₊_j = vadd(ρ₊_j, vload(V, ptrar + (c*N+2n  )*8D))
                    ρ₋_j = vadd(ρ₋_j, vload(V, ptrar + (c*N+2n+1)*8D))
                end
                mask_j === zero(mask_j) && break
            end
            ptrar += VectorizationBase.REGISTER_SIZE
        end
        Base.Cartesian.@nexprs 4 j -> begin
            vstore!(ptress, ess_j)
            ptress += VectorizationBase.REGISTER_SIZE
            vstore!(ptrmcmcse, mcmcse_j)
            ptrmcmcse += VectorizationBase.REGISTER_SIZE
        end
    end
    Witer = (D & ((W << 2)-1)) >> Wshift
    for i in 1:Witer

    end
    rem = D & (W - 1)
    for i in 1:rem

    end

end



end # module
