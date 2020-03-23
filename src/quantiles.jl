

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

# This implementation of the radix sort is copied from (with modifications):
# https://github.com/JuliaCollections/SortingAlgorithms.jl/blob/master/src/SortingAlgorithms.jl#L43
# Under MIT license
# https://github.com/JuliaCollections/SortingAlgorithms.jl/blob/master/LICENSE.md

using Base.Sort
using Base.Order

struct RadixSortAlg <: Algorithm end

const RadixSort = RadixSortAlg()

## Radix sort

# Map a bits-type to an unsigned int, maintaining sort order
uint_mapping(::ForwardOrdering, x::Unsigned) = x
uint_mapping(::ForwardOrdering, x::Integer) = unsigned(x)
uint_mapping(::ForwardOrdering, x::Float32)  = (y = reinterpret(Int32, x); reinterpret(UInt32, ifelse(y < 0, ~y, xor(y, typemin(Int32)))))
uint_mapping(::ForwardOrdering, x::Float64)  = (y = reinterpret(Int64, x); reinterpret(UInt64, ifelse(y < 0, ~y, xor(y, typemin(Int64)))))

uint_mapping(rev::ReverseOrdering, x) = ~uint_mapping(rev.fwd, x)
uint_mapping(::ReverseOrdering{ForwardOrdering}, x::Real) = ~uint_mapping(Forward, x) # maybe unnecessary; needs benchmark

uint_mapping(o::By,   x     ) = uint_mapping(Forward, o.by(x))
uint_mapping(o::Perm, i::Int) = uint_mapping(o.order, o.data[i])
uint_mapping(o::Lt,   x     ) = error("uint_mapping does not work with general Lt Orderings")

const RADIX_SIZE = 11
const RADIX_MASK = 0x7FF

@inline function icumsum!(b, a)
    s = zero(eltype(a))
    @inbounds for i in eachindex(b,a)
        b[i] = s += a[i]
    end
end


const BIN64 = [ Matrix{UInt32}(undef, 1 << RADIX_SIZE,  ceil(Integer, 8*8/RADIX_SIZE)) for _ in 1:Sys.CPU_THREADS ]
const CBIN = [ Vector{UInt32}(undef, 1 << RADIX_SIZE) for _ in 1:Sys.CPU_THREADS ]

function bin_type(::Type{T}) where {T}
    sizeof(T) == 8 && return fill!(BIN64[Threads.threadid()], zero(UInt32))
    iters = ceil(Integer, sizeof(T)*8/RADIX_SIZE)
    zeros(UInt32, 1<<RADIX_SIZE, iters)    
end

@inline function to_ind(v, j)
    Base.unsafe_trunc(Int, (v >> ((j-1)*RADIX_SIZE)) & RADIX_MASK) + 1
end

function Base.sort!(
                    vs::AbstractVector, ts::AbstractVector, lo::Int, hi::Int, ::RadixSortAlg, o::Ordering
                    )
    # Input checking
    # lo < hi || return vs, ts, 0

    # Make sure we're sorting a bits type
    T = Base.Order.ordtype(o, vs)
    isbitstype(T) || error("Radix sort only sorts bits types (got $T)")

    # Init
    iters = ceil(Integer, sizeof(T)*8/RADIX_SIZE)
    bin = bin_type(T) #zeros(UInt32, 1<<RADIX_SIZE, iters)
    lo > 1 && (@inbounds bin[1,:] .= lo-1)

    # Histogram for each element, radix
    @inbounds for i = lo:hi
        v = uint_mapping(o, vs[i])
        for j = 1:iters
            idx = to_ind(v, j)
            bin[idx, j] += 1
        end
    end

    # Sort!
    swaps = 0
    len = hi-lo+1

    cbin = CBIN[Threads.threadid()]
    @inbounds for j = 1:iters
        # Unroll first data iteration, check for degenerate case
        v = uint_mapping(o, vs[hi])
        idx = to_ind(v, j)

        # are all values the same at this radix?
        bin[idx,j] == len && continue

        icumsum!(cbin, @view(bin[:,j]))
        i = hi
        while true
            ci = cbin[idx]
            ts[ci] = vs[i]
            cbin[idx] = ci - 1
            i -= 1
            i == lo - 1 && break
            idx = to_ind(uint_mapping(o, vs[i]), j)
        end
        vs,ts = ts,vs
        swaps += 1
    end
    vs, ts, swaps
end
function Base.sort!(vs::AbstractVector, lo::Int, hi::Int, ::RadixSortAlg, o::Ordering, ts=similar(vs))
    lo < hi || return vs
    vs, ts, swaps  = sort!(vs, ts, lo, hi,  RadixSort, o)
    if isodd(swaps)
        vs,ts = ts,vs
        @inbounds for i = lo:hi
            vs[i] = ts[i]
        end
    end
    vs
end

function Base.Sort.Float.fpsort!(v::AbstractVector, ::RadixSortAlg, o::Ordering)
    lo, hi = Base.Sort.Float.nans2end!(v,o)
    sort!(v, lo, hi, RadixSort, o)
end

function calc_quantiles(A::AbstractMatrix{T}, qs::NTuple{Q,T}) where {Q,T}
    D, S = size(A)
    At = copy(A') # better on memory bandwidth
    quantiles = Matrix{T}(undef, D, Q)
    vs = Vector{T}(undef, S)
    ts = Vector{T}(undef, S)
    @inbounds for d in 1:D
        for s in 1:S
            vs[s] = At[s,d]
        end
        vs, ts = sort!(vs, ts, 1, S, RadixSort, ForwardOrdering())
        qvs = quantile(vs, qs, sorted = true)
        for q in 1:Q
            quantiles[d,q] = qvs[q]
        end
    end
    quantiles
end

function calc_quantiles_threaded(A::AbstractMatrix{T}, qs::NTuple{Q,T}) where {Q,T}
    D, S = size(A)
    quantiles = Matrix{T}(undef, D, Q)
    Saligned = (S + 63) & -64
    Ts = Matrix{T}(undef, Saligned, Threads.nthreads())
    Vs = Matrix{T}(undef, Saligned, D)
    permutedims!(@view(Vs[1:S,:]), A, (2,1))
    Threads.@threads for d in 1:D
        sort!(
            @view(Vs[:,d]), 1, S, RadixSort, ForwardOrdering(), @view(Ts[:,Threads.threadid()])
        )
    end
    for d in 1:D
        qvs = quantile(@view(Vs[:,d]), qs, sorted = true)
        @inbounds for q in 1:Q
            quantiles[d,q] = qvs[q]
        end
    end
    quantiles
end

# b = randn(10^6); a = similar(b); c = similar(b);
# quantile(sort!(copy!(a,b), 1, length(a), RadixSort2, ForwardOrdering(), c), (0.025, 0.25, 0.5, 0.75, 0.975), sorted = true)




