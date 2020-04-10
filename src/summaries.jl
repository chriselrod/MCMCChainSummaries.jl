

const SUMMARY_HEADER = ["Parameter", "Mean", "St.Dev.", "MCMC SE", "ESS", "PSRF"]
const STANDARD_QUANTILES_HEADER = ["Parameter", "2.5%", "25%", "50%", "75%", "97.5%"]

const SIGFIGS = Ref(5)

struct Summary <: AbstractMatrix{String}
    parameter_names::Vector{String}
    header::Vector{String}
    summary::Matrix{Float64}
    number_samples::Int
    number_chains::Int
end
struct MCMCChainSummary
    summary::Summary
    quantiles::Summary
end
Base.size(s::Summary) = size(s.summary) .+ (0,1)
Base.getindex(s::Summary, i, j) = j == 1 ? s.parameter_names[i] : string(round(s.summary[i, j-1], sigdigits = SIGFIGS[]))

# const LARGE_DIFF = Highlighter( (data,i,j) -> (j == 2 && data.summary[i,1] / data.summary[i,2] > 2 ), bold = false, foreground = :green )
const SMALL_NEFF = Highlighter( (data,i,j) -> (j == 5 && data.summary[i,4] < 0.1 * (data.number_samples*data.number_chains) ), bold = false, foreground = :red )
const PSRF_ALERT = Highlighter( (data,i,j) -> (j == 6 && data.summary[i,5] > 1.05 ), bold =  true, foreground = :red )
# const EXTREME_QT = Highlighter( (data,i,j) -> (((j == 2) | (j == 6)) && signbit(data.summary[i,j-1]) == (4<j) ), bold =  true, foreground = :green )
# const MINOR_QTLE = Highlighter( (data,i,j) -> (((j == 3) | (j == 5)) && signbit(data.summary[i,j-1]) == (4<j) ), bold = false, foreground = :green )
    
function Base.show(io::IO, s::Summary; kwargs...)
    if s.header === SUMMARY_HEADER
        pretty_table(io, s, s.header; highlighters = (SMALL_NEFF, PSRF_ALERT), crop = :none )
    else
        pretty_table(io, s, s.header; crop = :none )
    end
end
function Base.show(io::IO, s::MCMCChainSummary)
    println("$(s.summary.number_chains) chains of $(s.summary.number_samples) samples.")
    show(io, s.summary)
    show(io, s.quantiles)
end
function print_latex(s::MCMCChainSummary)
    s1 = s.summary; s2 = s.quantiles
    pretty_table(s1, s1.header; highlighters = (SMALL_NEFF, PSRF_ALERT), crop = :none, backend = :latex )
    pretty_table(s2, s2.header; crop = :none, backend = :latex )
end



function MCMCChainSummary(
    chains_in::AbstractArray{Float64,3},
    parameter_names::Vector{String} = ["x[$i]" for i ∈ 1:size(chains_in,1)],
    quantiles = (0.025, 0.25, 0.5, 0.75, 0.975)
    # threaded::Bool = Threads.nthreads() > 1
)
    threaded = true
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
    NQ = length(quantiles)
    dquantiles = if threaded
        calc_quantiles_threaded(reshape(chains,(D,N*C)), quantiles)
    else
        calc_quantiles(reshape(chains,(D,N*C)), quantiles)
    end
    median_ind = findfirst(q -> q == 0.5, quantiles)
    if median_ind === nothing
        mean_and_var!(means, B, reshape(chain_means, (D,C)))
    else # if we calculate the median, it is more numerically accurate to use that rather than the first sample in mean_and_var!.
        mean_and_var!(means, B, reshape(chain_means, (D,C)), @view(dquantiles[:,median_ind]))
    end
    # Calculate all autocorrelations
    Np2 = VectorizationBase.nextpow2(N)
    padded_chains = Array{Float64,3}(undef, D, Np2, C)
    @inbounds @fastmath for c in 1:C
        for n in 1:N
            for d in 1:D
                padded_chains[d,n,c] = chains[d,n,c] - chain_means[d,1,c]
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
            @simd ivdep for d in 1:D
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
    WV = VectorizationBase.pick_vector_width_val(Float64)
    V = VectorizationBase.pick_vector(Float64)
    Ddiv4Witer = D >> (Wshift + 2)
    ptrstdev = gep(pointer(summary), D)
    ptrmcmcse = gep(pointer(summary), 2D)
    ptress = gep(pointer(summary), 3D)
    ptrpsrf = gep(pointer(summary), 4D)
    # @show size(padded_chains)
    # @show padded_chains[:,2:3,:] ./ padded_chains[:,1:1,:]
    ptrcov = pointer(padded_chains)
    ptrar = ptrcov
    ptrb = pointer(B)
    invWdenom = vbroadcast(V, inv(C * (N-1) * Np2))
    nfrac = vbroadcast(V, (N-1) / N )
    invn = vbroadcast(V, 1/N)
    invC = vbroadcast(V, 1/C)
    NC = vbroadcast(V, Float64(N*C))
    Nh = N >> 1
    GC.@preserve summary padded_chains B begin
        for _ in 1:Ddiv4Witer
            Base.Cartesian.@nexprs 4 j -> begin
                Wvar_j = vload(V, ptrcov)
                for c in 1:C-1
                    Wvar_j = extract_data(vadd(Wvar_j, vload(ptrcov, _MM(WV, c*Np2*D))))
                end
                Wvar_j = extract_data(vmul(invWdenom, Wvar_j))
                ptrcov = gep(ptrcov, W)
            end
            Base.Cartesian.@nexprs 4 j -> begin
                B_j = vload(V, ptrb)
                var_j = vmuladd(nfrac, Wvar_j, B_j)
                ptrb = gep(ptrb, W)
                bwa_j = vfnmadd_fast(invn, Wvar_j, B_j)
                bwa_j = vadd(bwa_j, bwa_j)
            end
            Base.Cartesian.@nexprs 4 j -> begin
                r̂²_j = vfdiv(var_j, Wvar_j)
                sqrt_j = extract_data(vsqrt(var_j))
                rhat_j = vsqrt(r̂²_j)
            end
            Base.Cartesian.@nexprs 4 j -> begin
                vstore!(ptrstdev, sqrt_j)
                ptrstdev = gep(ptrstdev, W)
                vstore!(ptrpsrf, rhat_j)
                ptrpsrf = gep(ptrpsrf, W)
            end
            Base.Cartesian.@nexprs 4 j -> begin
                prec_j = SIMDPirates.vfdiv(invWdenom, var_j)
                ρ_j = vload(ptrar, _MM(WV, D))
                for c in 1:C-1
                    ρ_j = vadd(ρ_j, vload(ptrar, _MM(WV, (c*Np2+1)*D)))
                end
                p_j = vmul(vfmadd_fast(invC, ρ_j, bwa_j), prec_j)
                # tau_j =  vadd(tau_j, p_j)
                tau_j =  extract_data(vfmadd_fast(vbroadcast(V, 4.0), p_j, SIMDPirates.vbroadcast(V, 1.0)))
                # tau_j = extract_data(SIMDPirates.vbroadcast(V, 1.0))
                mask_j = VectorizationBase.max_mask(Float64)
                for n in 1:Nh-2
                    ρ₋_j = vload(ptrar, _MM(WV, (2n  )*D))
                    ρ₊_j = vload(ptrar, _MM(WV, (2n+1)*D))
                    for c in 1:C-1
                        ρ₋_j = vadd(ρ₋_j, vload(ptrar, _MM(WV, (c*Np2+2n  )*D)))
                        ρ₊_j = vadd(ρ₊_j, vload(ptrar, _MM(WV, (c*Np2+2n+1)*D)))
                    end
                    p_j = vmul(vfmadd_fast(invC, vadd(ρ₊_j, ρ₋_j), bwa_j), prec_j)
                    # tau_j =  vadd(tau_j, p_j)
                    tau_j =  extract_data(vifelse(mask_j, vfmadd_fast(vbroadcast(V, 4.0), p_j, tau_j), tau_j))
                    mask_j &= SIMDPirates.vgreater(p_j, SIMDPirates.vbroadcast(V, 0.0))
                    mask_j === zero(mask_j) && break
                end
                ess_j = vfdiv(NC, tau_j)
                mcmcse_j = vfdiv(sqrt_j, vsqrt(ess_j))
                vstore!(ptress, ess_j)
                vstore!(ptrmcmcse, mcmcse_j)
                ptrar = gep(ptrar, W)
                ptress = gep(ptress, W)
                ptrmcmcse = gep(ptrmcmcse, W)
            end
        end
        #    Witer = (D & ((W << 2)-1)) >> Wshift
        Witer = D & (-4W)
        #for _ in 1:Witer
        while Witer < D
            _mask = SIMDPirates.svrange(_MM(WV,Witer)) < D
            Witer += W
            Wvar_ = vload(V, ptrcov, _mask)
            for c in 1:C-1
                Wvar_ = extract_data(vadd(Wvar_, vload(ptrcov, _MM(WV, c*Np2*D), _mask)))
            end
            Wvar_ = extract_data(vmul(invWdenom, Wvar_))
            ptrcov = gep(ptrcov, W)
            B_ = vload(V, ptrb, _mask)
            var_ = vmuladd(nfrac, Wvar_, B_)
            ptrb = gep(ptrb, W)
            r̂²_ = vfdiv(var_, Wvar_)
            sqrt_ = extract_data(vsqrt(var_))
            rhat_ = vsqrt(r̂²_)
            vstore!(ptrstdev, sqrt_, _mask)
            ptrstdev = gep(ptrstdev, W)
            vstore!(ptrpsrf, rhat_, _mask)
            ptrpsrf = gep(ptrpsrf, W)
            prec_ = SIMDPirates.vfdiv(invWdenom, var_)
            mask_ = _mask
            bwa = vfnmadd_fast(invn, Wvar_, B_)
            bwa = vadd(bwa, bwa)
            ρ_ = vload(ptrar, _MM(WV, D), _mask)
            for c in 1:C-1
                ρ_ = vadd(ρ_, vload(ptrar, _MM(WV, (c*Np2+1)*D), _mask))
            end
            ρ_ = vmul(vfmadd_fast(invC, ρ_, bwa), prec_)
            tau_ =  extract_data(vfmadd_fast(vbroadcast(V, 4.0), ρ_, SIMDPirates.vbroadcast(V, 1.0)))
            # tau_ =  extract_data(SIMDPirates.vbroadcast(V, 1.0))
            for n in 1:Nh-2
                ρ₋_ = vload(ptrar, _MM(WV, (2n  )*D), _mask)
                ρ₊_ = vload(ptrar, _MM(WV, (2n+1)*D), _mask)
                for c in 1:C-1
                    ρ₋_ = vadd(ρ₋_, vload(ptrar, _MM(WV, (c*Np2+2n  )*D), _mask))
                    ρ₊_ = vadd(ρ₊_, vload(ptrar, _MM(WV, (c*Np2+2n+1)*D), _mask))
                end
                p_ = vmul(vfmadd_fast(invC, vadd(ρ₊_, ρ₋_), bwa), prec_)
                tau_ =  extract_data(vifelse(mask_, vfmadd_fast(vbroadcast(V, 4.0), p_, tau_), tau_))
                # @show SIMDPirates.vgreater(p_, SIMDPirates.vbroadcast(V, 0.0))
                mask_ &= SIMDPirates.vgreater(p_, SIMDPirates.vbroadcast(V, 0.0))
                mask_ === zero(mask_) && break
                # @show SIMDPirates.vgreater(p_, SIMDPirates.vbroadcast(V, 0.0)) === zero(mask_) && break
            end
            ess_ = vfdiv(NC, tau_)
            mcmcse_ = vfdiv(sqrt_, vsqrt(ess_))
            vstore!(ptress, ess_, _mask)
            vstore!(ptrmcmcse, mcmcse_, _mask)
            ptrar = gep(ptrar, W)
            ptress = gep(ptress, W)
            ptrmcmcse = gep(ptrmcmcse, W)
        end
        # dquantiles = Matrix{Float64}(undef, D, NQ)
        # if NQ > 0
        # sorted_samples =  sort!(copy(reshape(chains,(D,N*C))'), dims = 1)
        # for d in 1:D
        # dquantiles[d,:] .= quantile(@view(sorted_samples[:,d]), quantiles, sorted = true)
        # end
        # end
    end
    D, N, C = size(chains_in)
    MCMCChainSummary(
        Summary(parameter_names, SUMMARY_HEADER, summary, N, C),
        Summary(parameter_names, quantile_names(quantiles), dquantiles, N, C)
    )
end

function MCMCChainSummary(
    chains_in::AbstractMatrix{Float64},
    parameter_names::Vector{String} = ["x[$i]" for i ∈ 1:size(chains_in,1)],
    quantiles = (0.025, 0.25, 0.5, 0.75, 0.975);
    threaded::Bool = Threads.nthreads() > 1
)    
    M, N = size(chains_in)
    MCMCChainSummary(reshape(chains_in, (M,N,1)), parameter_names, quantiles, threaded = threaded)
end

Statistics.mean(chn_sum::MCMCChainSummary) = chn_sum.summary.summary[:,1]
Statistics.std(chn_sum::MCMCChainSummary) = chn_sum.summary.summary[:,2]
function Statistics.median(chn_sum::MCMCChainSummary)
    q = chn_sum.quantiles
    q.header === STANDARD_QUANTILES_HEADER && return q.summary[:,3]
    q.summary[:,findfirst(s -> s == "50.0%", q.header)-1]
end




