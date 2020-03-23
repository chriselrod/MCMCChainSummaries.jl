using MCMCChainSummaries
using Test

@testset "MCMCChainSummaries.jl" begin
    samples = 9999; nchains = 17;
    for samples ∈ 9_999:10_000
        for P in 15:33
            chn_sm = MCMCChainSummary(randn(P,samples,nchains))
            #[abs(chn_sm.summary.summary[p,2] - 1) for p in 1:P]'
            # All standard deviations reasonably close to 1
            chn_sm_mat = chn_sm.summary.summary;
            # TODO: test vs Chi-square quantiles?
            @test all(p -> abs(chn_sm_mat[p,2] - 1) < 0.01, 1:P)
            # Means less than 10 MCMC SE away from 0
            @test all(p -> abs(chn_sm_mat[p,1] / chn_sm_mat[p,3]) < 10, 1:P)
            # ESS within 1% of samples * nchains
            # [abs(1 - chn_sm_mat[p,4] / samples*nchains) < 0.01 for p in 1:P]
            @test all(p -> abs(1 - chn_sm_mat[p,4] / (samples*nchains)) < 0.01, 1:P)
            @test all(p -> chn_sm_mat[p,5] < 1.001, 1:P)
        end
    end
end
