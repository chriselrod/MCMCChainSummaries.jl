module MCMCChainSummaries

using VectorizationBase, SIMDPirates#, PaddedMatrices
# using ProbabilityModels
using Statistics, FFTW
using PrettyTables

const SUMMARY_HEADER = ["Parameter", "Mean", "St.Dev.", "MCMC SE", "ESS", "PSRF"]
const STANDARD_QUANTILES_HEADER = ["Parameter", "2.5%", "25%", "50%", "75%", "97.5"]

# abstract type AbstractSummary <: AbstractMatrix{Union{String,Float64}} end
struct Summary <: AbstractMatrix{Union{String,Float64}}#<: AbstractSummary
    parameter_names::Vector{String}
    header::Vector{String}
    summary::Matrix{Float64}
end
# struct Quantiles <: AbstractSummary
#     parameter_names::Vector{String}
#     quantile_names::Vector{String}
#     quantile_values::Matrix{Float64}
# end
struct MCMCChainSummary
    summary::Summary
    quantiles::Summary
end
Base.size(s::Summary) = size(s.summary) .+ (0,1)
Base.getindex(s::Summary, i, j) = j == 1 ? s.parameter_names[i] : s.summary[i, j-1]

const LARGE_DIFF = Highlighter( (data,i,j) -> (j == 2 && data.summary[i,1] / data.summary[i,2] > 2 ), bold = false, foreground = :green )
const SMALL_NEFF = Highlighter( (data,i,j) -> (j == 5 && data.summary[i,4] < 100 ), bold = false, foreground = :red )
const PSRF_ALERT = Highlighter( (data,i,j) -> (j == 6 && data.summary[i,5] > 1.1 ), bold =  true, foreground = :red )
const EXTREME_QT = Highlighter( (data,i,j) -> (((j == 2) | (j == 6)) && signbit(data.summary[i,j-1]) == (4<j) ), bold =  true, foreground = :green )
const MINOR_QTLE = Highlighter( (data,i,j) -> (((j == 3) | (j == 5)) && signbit(data.summary[i,j-1]) == (4<j) ), bold = false, foreground = :green )
    
function Base.show(io::IO, s::Summary; kwargs...)
    if s.header === SUMMARY_HEADER
        pretty_table(io, s.summary, s.header; highlighters = (LARGE_DIFF, SMALL_NEFF, PSRF_ALERT), crop = :none )
    elseif s.header === STANDARD_QUANTILES_HEADER
        pretty_table(io, s.summary, s.header; highlighters = (MINOR_QTLE, EXTREME_QT), crop = :none )
    else
        pretty_table(io, s.summary, s.header; crop = :none )
    end
end
function Base.show(io::IO, s::MCMCChainSummary)
    # ss = s.summary; sq = s.quantiles
    # pretty_table(io, ss, ss.header; highlighters = (LARGE_DIFF, SMALL_NEFF, PSRF_ALERT), crop = :none)
    # pretty_table(io, sq, sq.header; highlighters = (MINOR_QTLE, EXTREME_QT), crop = :none )
    show(io, s.summary)
    show(io, s.quantiles)
end

function regularized_cov_block_quote(W::Int, T, reps_per_block::Int, stride, mask_last::Bool = false, mask = :r)# = 0xff)
    # loads from ptr_sample
    # stores in ptr_s² and ptr_invs
    # needs vNinv, mulreg, and addreg to be defined
    reps_per_block -= 1
    size_T = sizeof(T)
    WT = size_T*W
    V = Vec{W,T}
    quote
        $([Expr(:(=), Symbol(:μ_,i), :(vload($V, ptr_smpl + $(WT*i), $([mask for _ ∈ 1:((i==reps_per_block) & mask_last)]...)))) for i ∈ 0:reps_per_block]...)
        $([Expr(:(=), Symbol(:Σδ_,i), :(vbroadcast($V,zero($T)))) for i ∈ 0:reps_per_block]...)
        $([Expr(:(=), Symbol(:Σδ²_,i), :(vbroadcast($V,zero($T)))) for i ∈ 0:reps_per_block]...)
        for n ∈ 1:N-1
            $([Expr(:(=), Symbol(:δ_,i), :(vsub(vload($V, ptr_smpl + $(WT*i) + n*$stride*$size_T),$(Symbol(:μ_,i))))) for i ∈ 0:reps_per_block]...)
            $([Expr(:(=), Symbol(:Σδ_,i), :(vadd($(Symbol(:δ_,i)),$(Symbol(:Σδ_,i))))) for i ∈ 0:reps_per_block]...)
            $([Expr(:(=), Symbol(:Σδ²_,i), :(vmuladd($(Symbol(:δ_,i)),$(Symbol(:δ_,i)),$(Symbol(:Σδ²_,i))))) for i ∈ 0:reps_per_block]...)
        end
        $([Expr(:(=), Symbol(:xbar_,i), :(vmuladd(vNinv, $(Symbol(:Σδ_,i)), $(Symbol(:μ_,i))))) for i ∈ 0:reps_per_block]...)
        $([Expr(:(=), Symbol(:ΣδΣδ_,i), :(vmul($(Symbol(:Σδ_,i)),$(Symbol(:Σδ_,i))))) for i ∈ 0:reps_per_block]...)
        $([Expr(:(=), Symbol(:s²_,i), :(vmul(vNm1inv,vfnmadd($(Symbol(:ΣδΣδ_,i)),vNinv,$(Symbol(:Σδ²_,i)))))) for i ∈ 0:reps_per_block]...)
        $([:(vstore!(ptr_mean, $(Symbol(:xbar_,i)), $([mask for _ ∈ 1:((i==reps_per_block) & mask_last)]...)); ptr_mean += $WT) for i ∈ 0:reps_per_block]...)
        $([:(vstore!(ptr_vars, $(Symbol(:s²_,i)), $([mask for _ ∈ 1:((i==reps_per_block) & mask_last)]...)); ptr_vars += $WT) for i ∈ 0:reps_per_block]...)
        ptr_smpl += $(WT*(reps_per_block+1))
    end
end
@generated function mean_and_var!(
    means::AbstractVector{T}, vars::AbstractVector{T}, sample::AbstractArray{T}
) where {T}
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    V = Vec{W,T}
    quote 
        D, N = size(sample); sample_stride = stride(sample, 2)
        @boundscheck if length(means) < D || length(vars) < D
            throw(BoundsError("Size of sample: ($D,$N); length of preallocated mean vector: $(length(means)); length of preallocated var vector: $(length(vars))"))
        end
        ptr_mean = pointer(means); ptr_vars = pointer(vars); ptr_smpl = pointer(sample)
        vNinv = vbroadcast($V, 1/N); vNm1inv = vbroadcast($V, 1/(N-1))
        for _ in 1:(D >>> $(Wshift + 2)) # blocks of 4 vectors
            $(regularized_cov_block_quote(W, T, 4, :sample_stride))
        end
        for _ in 1:((D & $((W << 2)-1)) >>> $Wshift) # single vectors
            $(regularized_cov_block_quote(W, T, 1, :sample_stride))
        end
        r = D & $(W-1)
        if r > 0 # remainder
            mask = VectorizationBase.mask(T, r)
            $(regularized_cov_block_quote(W, T, 1, :sample_stride, true, :mask))
        end
        nothing
    end
end

function quantile_names(q)
    q == (0.025, 0.25, 0.5, 0.75, 0.975) && return STANDARD_QUANTILES_HEADER
    NQ = length(q)
    NQ == 0 && return String[]
    q_names = Vector{String}(undef, NQ + 1)
    q_names[1] = first(STANDARD_QUANTILES_HEADER)
    for nq in 1:NQ
        q_names[nq+1] = "$(round(100*q[nq],sigdigits=4))%"
    end
    q_names
end

function MCMCChainSummary(chains_in::AbstractArray{Float64,3}, parameter_names::Vector{String}, quantiles = (0.025, 0.25, 0.5, 0.75, 0.975))
    D, N, C = size(chains_in)
    if iseven(N) # split chains
        N >>= 1
        C <<= 1
    end
    chains = reshape(chains_in, (D,N,C))
    chain_means = mean(chains, dims = 2)
    summary = Matrix{Float64}(undef, D, 5)
    means = @view(summary[:,1])
    B = @view(summary[:,2])
    mean_and_var!(means, B, reshape(chain_means, (D,C)))
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
    FFTW.r2r!(padded_chains, FFTW.R2HC, 2)
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
    FFTW.r2r!(padded_chains, FFTW.HC2R, 2)
    # Then calculate ESS
    # https://mc-stan.org/docs/2_20/reference-manual/notation-for-samples-chains-and-draws.html
    # https://mc-stan.org/docs/2_20/reference-manual/effective-sample-size-section.html
    W, Wshift = VectorizationBase.pick_vector_width_shift(Float64)
    V = VectorizationBase.pick_vector(Float64)
    Ddiv4Witer = D >> (Wshift + 2)
    ptrstdev = pointer(summary) + 8D
    ptrmcmcse = pointer(summary) + 16D
    ptress = pointer(summary) + 24D
    ptrpsrf = pointer(summary) + 32D
    ptrcov = pointer(padded_chains)
    ptrar = ptrcov
    ptrb = pointer(B)
    invWdenom = vbroadcast(V, inv(C * (N-1) * Np2))
    nfrac = vbroadcast(V, (N-1) / N )
    invn = vbroadcast(V, 1/N)
    invC = vbroadcast(V, 1/C)
    NC = vbroadcast(V, Float64(N*C))
    Nh = N >> 1
    for _ in 1:Ddiv4Witer
        Base.Cartesian.@nexprs 4 j -> begin
            Wvar_j = vload(V, ptrcov)
            for c in 1:C-1
                Wvar_j = vadd(Wvar_j, vload(V, ptrcov + c*Np2*8D))
            end
            Wvar_j = vmul(invWdenom, Wvar_j)
            ptrcov += VectorizationBase.REGISTER_SIZE
        end
        Base.Cartesian.@nexprs 4 j -> begin
            B_j = vload(V, ptrb)
            var_j = vmuladd(nfrac, Wvar_j, B_j)
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
            prec_j = SIMDPirates.vfdiv(invWdenom, var_j)
            tau_j = SIMDPirates.vbroadcast(V, 1.0)
            mask_j = VectorizationBase.max_mask(Float64)
            for n in 1:Nh-1
                ρ₊_j = vload(V, ptrar + (2n-1)*8D)
                ρ₋_j = vload(V, ptrar + (2n  )*8D)
                for c in 1:C-1
                    ρ₊_j = vadd(ρ₊_j, vload(V, ptrar + (c*Np2+2n-1)*8D))
                    ρ₋_j = vadd(ρ₋_j, vload(V, ptrar + (c*Np2+2n  )*8D))
                end
                bwa = vfnmadd(invn, Wvar_j, B_j)
                p_j = vmul(vfmadd(invC, vadd(ρ₊_j, ρ₋_j), vadd(bwa,bwa)), prec_j)
                # tau_j =  vadd(tau_j, p_j)
                tau_j =  vifelse(mask_j, vmuladd(vbroadcast(V, 2.0), p_j, tau_j), tau_j)
                mask_j = SIMDPirates.vand(mask_j, SIMDPirates.vgreater(p_j, SIMDPirates.vbroadcast(V, 0.0)))
                mask_j === zero(mask_j) && break
            end
            ess_j = vfdiv(NC, tau_j)
            mcmcse_j = vfdiv(sqrt_j, vsqrt(ess_j))
            vstore!(ptress, ess_j)
            vstore!(ptrmcmcse, mcmcse_j)
            ptrar += VectorizationBase.REGISTER_SIZE
            ptress += VectorizationBase.REGISTER_SIZE
            ptrmcmcse += VectorizationBase.REGISTER_SIZE
        end
    end
    Witer = (D & ((W << 2)-1)) >> Wshift
    for _ in 1:Witer
        Wvar_ = vload(V, ptrcov)
        for c in 1:C-1
            Wvar_ = vadd(Wvar_, vload(V, ptrcov + c*Np2*8D))
        end
        Wvar_ = vmul(invWdenom, Wvar_)
        ptrcov += VectorizationBase.REGISTER_SIZE
        B_ = vload(V, ptrb)
        var_ = vmuladd(nfrac, Wvar_, B_)
        ptrb += VectorizationBase.REGISTER_SIZE
        r̂²_ = vfdiv(var_, Wvar_)
        sqrt_ = vsqrt(var_)
        rhat_ = vsqrt(r̂²_)
        vstore!(ptrstdev, sqrt_)
        ptrstdev += VectorizationBase.REGISTER_SIZE
        vstore!(ptrpsrf, rhat_)
        ptrpsrf += VectorizationBase.REGISTER_SIZE
        prec_ = SIMDPirates.vfdiv(invWdenom, var_)
        tau_ = SIMDPirates.vbroadcast(V, 1.0)
        mask_ = VectorizationBase.max_mask(Float64)
        for n in 1:Nh-1
            ρ₊_ = vload(V, ptrar + (2n-1)*8D)
            ρ₋_ = vload(V, ptrar + (2n  )*8D)
            for c in 1:C-1
                ρ₊_ = vadd(ρ₊_, vload(V, ptrar + (c*Np2+2n-1)*8D))
                ρ₋_ = vadd(ρ₋_, vload(V, ptrar + (c*Np2+2n  )*8D))
            end
            bwa = vfnmadd(invn, Wvar_, B_)
            p_ = vmul(vfmadd(invC, vadd(ρ₊_, ρ₋_), vadd(bwa,bwa)), prec_)
            tau_ =  vifelse(mask_, vmuladd(vbroadcast(V, 2.0), p_, tau_), tau_)
            mask_ = SIMDPirates.vand(mask_, SIMDPirates.vgreater(p_, SIMDPirates.vbroadcast(V, 0.0)))
            mask_ === zero(mask_) && break
        end
        ess_ = vfdiv(NC, tau_)
        mcmcse_ = vfdiv(sqrt_, vsqrt(ess_))
        vstore!(ptress, ess_)
        vstore!(ptrmcmcse, mcmcse_)
        ptrar += VectorizationBase.REGISTER_SIZE
        ptress += VectorizationBase.REGISTER_SIZE
        ptrmcmcse += VectorizationBase.REGISTER_SIZE
    end
    sinvWdenom = inv(C * (N-1) * Np2)
    snfrac = (N-1) / N
    sinvn = 1/N
    sinvC = 1/C
    sNC = Float64(N*C)
    for d in 1+(D & (-W)):D
        Wvar_d = 0.0
        for c in 1:C
            Wvar_d += padded_chains[d,1,c]
        end
        Wvar_d *= sinvWdenom
        B_d = B[d]
        var_d = muladd(snfrac, Wvar_d, B_d)
        sqrt_d = sqrt(var_d)
        summary[d,5] = sqrt(var_d / Wvar_d)
        summary[d,2] = sqrt_d
        prec_d = sinvWdenom / var_d
        tau_d = 1.0
        for n in 1:Nh-1
            ρ₊_d = 0.0
            ρ₋_d = 0.0
            for c in 1:C
                ρ₊_d += padded_chains[d,n,c]
                ρ₋_d += padded_chains[d,n,c]
            end
            bwa_d = vfnmadd(sinvn, Wvar_d, B_d)
            p_d = vfmadd(sinvC, ρ₊_d + ρ₋_d, bwa_d + bwa_d ) * prec_d
            # tau_j =  vadd(tau_j, p_j)
            tau_d =  muladd(2.0, p_d, tau_d)
            tau_d > 0.0 || break
        end
        ess_d = sNC / tau_d
        summary[d,4] = ess_d
        summary[d,3] = sqrt_d / sqrt(ess_d)
    end
    dquantiles = Matrix{Float64}(undef, D, NQ)
    if NQ > 0
        sorted_samples =  sort!(copy(reshape(chains,(D,N*C))'), dims = 1)
        for d in 1:D
            dquantiles[d,:] .= quantile(@view(sorted_samples[:,d]), quantiles, sorted = true)
        end
    end
    MCMCChainSummary(
        Summary(parameter_names, SUMMARY_HEADER, summary),
        Summary(parameter_names, quantile_names(q), dquantiles)
    )
end


end # module
