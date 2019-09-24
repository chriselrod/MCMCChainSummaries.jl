using Documenter, MCMCChainSummaries

makedocs(;
    modules=[MCMCChainSummaries],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/chriselrod/MCMCChainSummaries.jl/blob/{commit}{path}#L{line}",
    sitename="MCMCChainSummaries.jl",
    authors="Chris Elrod",
    assets=String[],
)

deploydocs(;
    repo="github.com/chriselrod/MCMCChainSummaries.jl",
)
