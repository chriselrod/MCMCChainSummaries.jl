


function mean_and_var!(
    means::AbstractVector{T}, vars::AbstractVector{T}, sample::AbstractArray{T}
) where {T}
    V = VectorizationBase.pick_vector(T)
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    WT = VectorizationBase.REGISTER_SIZE
    D, N = size(sample); sample_stride = stride(sample, 2) * sizeof(T)
    @boundscheck if length(means) < D || length(vars) < D
        throw(BoundsError("Size of sample: ($D,$N); length of preallocated mean vector: $(length(means)); length of preallocated var vector: $(length(vars))"))
    end
    ptr_mean = pointer(means); ptr_vars = pointer(vars); ptr_smpl = pointer(sample)
    vNinv = vbroadcast(V, 1/N); vNm1inv = vbroadcast(V, 1/(N-1))
    GC.@preserve means vars sample begin
        for _ in 1:(D >>> (Wshift + 2)) # blocks of 4 vectors
            Base.Cartesian.@nexprs 4 i -> μ_i = vload(V, ptr_smpl + WT * (i-1))
            Base.Cartesian.@nexprs 4 i -> Σδ_i = vbroadcast(V, zero(T))
            Base.Cartesian.@nexprs 4 i -> Σδ²_i = vbroadcast(V, zero(T))
            for n ∈ 1:N-1
                Base.Cartesian.@nexprs 4 i -> δ_i = vsub(vload(V, ptr_smpl + WT * (i-1) + n*sample_stride), μ_i)
                Base.Cartesian.@nexprs 4 i -> Σδ_i = vadd(δ_i, Σδ_i)
                Base.Cartesian.@nexprs 4 i -> Σδ²_i = vmuladd(δ_i, δ_i, Σδ²_i)
            end
            Base.Cartesian.@nexprs 4 i -> xbar_i = vmuladd(vNinv, Σδ_i, μ_i)
            Base.Cartesian.@nexprs 4 i -> ΣδΣδ_i = vmul(Σδ_i, Σδ_i)
            Base.Cartesian.@nexprs 4 i -> s²_i = vmul(vNm1inv, vfnmadd(ΣδΣδ_i, vNinv, Σδ²_i))
            Base.Cartesian.@nexprs 4 i -> (vstore!(ptr_mean, xbar_i); ptr_mean += WT)
            Base.Cartesian.@nexprs 4 i -> (vstore!(ptr_vars, s²_i); ptr_vars += WT)
            ptr_smpl += 4WT
        end
        for _ in 1:((D & ((W << 2)-1)) >>> Wshift) # single vectors
            μ_i = vload(V, ptr_smpl)
            Σδ_i = vbroadcast(V, zero(T))
            Σδ²_i = vbroadcast(V, zero(T))
            for n ∈ 1:N-1
                δ_i = vsub(vload(V, ptr_smpl + n*sample_stride), μ_i)
                Σδ_i = vadd(δ_i, Σδ_i)
                Σδ²_i = vmuladd(δ_i, δ_i, Σδ²_i)
            end
            xbar_i = vmuladd(vNinv, Σδ_i, μ_i)
            ΣδΣδ_i = vmul(Σδ_i, Σδ_i)
            s²_i = vmul(vNm1inv, vfnmadd(ΣδΣδ_i, vNinv, Σδ²_i))
            vstore!(ptr_mean, xbar_i); ptr_mean += WT
            vstore!(ptr_vars, s²_i); ptr_vars += WT
            ptr_smpl += WT
        end
        r = D & (W-1)
        if r > 0 # remainder
            mask = VectorizationBase.mask(T, r)
            μ_i = vload(V, ptr_smpl, mask)
            Σδ_i = vbroadcast(V, zero(T))
            Σδ²_i = vbroadcast(V, zero(T))
            for n ∈ 1:N-1
                δ_i = vsub(vload(V, ptr_smpl + n*sample_stride, mask), μ_i)
                Σδ_i = vadd(δ_i, Σδ_i)
                Σδ²_i = vmuladd(δ_i, δ_i, Σδ²_i)
            end
            xbar_i = vmuladd(vNinv, Σδ_i, μ_i)
            ΣδΣδ_i = vmul(Σδ_i, Σδ_i)
            s²_i = vmul(vNm1inv, vfnmadd(ΣδΣδ_i, vNinv, Σδ²_i))
            vstore!(ptr_mean, xbar_i, mask)
            vstore!(ptr_vars, s²_i, mask)
        end
    end
    nothing
end


function mean_and_var!(
    means::AbstractVector{T}, vars::AbstractVector{T}, sample::AbstractArray{T}, init_means::AbstractVector{T}
) where {T}
    V = VectorizationBase.pick_vector(T)
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    WT = VectorizationBase.REGISTER_SIZE
    D, N = size(sample); sample_stride = stride(sample, 2) * sizeof(T)
    @boundscheck if length(means) < D || length(vars) < D
        throw(BoundsError("Size of sample: ($D,$N); length of preallocated mean vector: $(length(means)); length of preallocated var vector: $(length(vars))"))
    end
    ptr_mean = pointer(means); ptr_vars = pointer(vars); ptr_smpl = pointer(sample); ptr_im = pointer(init_means)
    vNinv = vbroadcast(V, 1/N); vNm1inv = vbroadcast(V, 1/(N-1))
    GC.@preserve means vars sample init_means begin
        for _ in 1:(D >>> (Wshift + 2)) # blocks of 4 vectors
            Base.Cartesian.@nexprs 4 i -> (μ_i = vload(V, ptr_im); ptr_im += WT)
            Base.Cartesian.@nexprs 4 i -> Σδ_i = vbroadcast(V, zero(T))
            Base.Cartesian.@nexprs 4 i -> Σδ²_i = vbroadcast(V, zero(T))
            for n ∈ 0:N-1
                Base.Cartesian.@nexprs 4 i -> δ_i = vsub(vload(V, ptr_smpl + WT * (i-1) + n*sample_stride), μ_i)
                Base.Cartesian.@nexprs 4 i -> Σδ_i = vadd(δ_i, Σδ_i)
                Base.Cartesian.@nexprs 4 i -> Σδ²_i = vmuladd(δ_i, δ_i, Σδ²_i)
            end
            Base.Cartesian.@nexprs 4 i -> xbar_i = vmuladd(vNinv, Σδ_i, μ_i)
            Base.Cartesian.@nexprs 4 i -> ΣδΣδ_i = vmul(Σδ_i, Σδ_i)
            Base.Cartesian.@nexprs 4 i -> s²_i = vmul(vNm1inv, vfnmadd(ΣδΣδ_i, vNinv, Σδ²_i))
            Base.Cartesian.@nexprs 4 i -> (vstore!(ptr_mean, xbar_i); ptr_mean += WT)
            Base.Cartesian.@nexprs 4 i -> (vstore!(ptr_vars, s²_i); ptr_vars += WT)
            ptr_smpl += 4WT
        end
        for _ in 1:((D & ((W << 2)-1)) >>> Wshift) # single vectors
            μ_i = vload(V, ptr_im); ptr_im += WT
            Σδ_i = vbroadcast(V, zero(T))
            Σδ²_i = vbroadcast(V, zero(T))
            for n ∈ 0:N-1
                δ_i = vsub(vload(V, ptr_smpl + n*sample_stride), μ_i)
                Σδ_i = vadd(δ_i, Σδ_i)
                Σδ²_i = vmuladd(δ_i, δ_i, Σδ²_i)
            end
            xbar_i = vmuladd(vNinv, Σδ_i, μ_i)
            ΣδΣδ_i = vmul(Σδ_i, Σδ_i)
            s²_i = vmul(vNm1inv, vfnmadd(ΣδΣδ_i, vNinv, Σδ²_i))
            vstore!(ptr_mean, xbar_i); ptr_mean += WT
            vstore!(ptr_vars, s²_i); ptr_vars += WT
            ptr_smpl += WT
        end
        r = D & (W-1)
        if r > 0 # remainder
            mask = VectorizationBase.mask(T, r)
            μ_i = vload(V, ptr_im, mask)
            Σδ_i = vbroadcast(V, zero(T))
            Σδ²_i = vbroadcast(V, zero(T))
            for n ∈ 0:N-1
                δ_i = vsub(vload(V, ptr_smpl + n*sample_stride, mask), μ_i)
                Σδ_i = vadd(δ_i, Σδ_i)
                Σδ²_i = vmuladd(δ_i, δ_i, Σδ²_i)
            end
            xbar_i = vmuladd(vNinv, Σδ_i, μ_i)
            ΣδΣδ_i = vmul(Σδ_i, Σδ_i)
            s²_i = vmul(vNm1inv, vfnmadd(ΣδΣδ_i, vNinv, Σδ²_i))
            vstore!(ptr_mean, xbar_i, mask)
            vstore!(ptr_vars, s²_i, mask)
        end
    end
    nothing
end



