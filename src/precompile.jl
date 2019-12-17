function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    precompile(Tuple{MCMCChainSummaries.var"#32#threadsfor_fun#5"{Int64,Array{Float64,2},Array{Float64,2},UnitRange{Int64}}})
end
