using Documenter, VIEstimators

makedocs(;
    modules=[VIEstimators],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/willtebbutt/VIEstimators.jl/blob/{commit}{path}#L{line}",
    sitename="VIEstimators.jl",
    authors="Will Tebbutt <wt0881@my.bristol.ac.uk>",
    assets=String[],
)

deploydocs(;
    repo="github.com/willtebbutt/VIEstimators.jl",
)
