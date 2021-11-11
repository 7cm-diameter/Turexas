using Distributions, Turing, Plots, StatsPlots, MCMCChains, DataFrames

function BiExponential(theta1::Real, theta2::Real, p::Real)
    return MixtureModel([Exponential(theta1),
                         Exponential(theta2)], [p, 1 - p])
end

function generate_samples(theta1::Real, theta2::Real, p::Real, n::Int64)
    m = BiExponential(theta1, theta2, p)
    rand(m, n)
end

@model bi_exponential(samples::Array{Real, 1}) = begin
    theta1 ~ Gamma(1, 10)
    theta2 ~ truncated(Normal(), theta1, theta1 + 100)
    p ~ Beta(1, 1)
    m = BiExponential(theta1, theta2, p)
    for i in 1:length(samples)
        samples[i] ~ m
    end
end

function svf(x::Array{Real, 1}, digits::Int64)
    x = round.(x, digits=digits)
    unqx = unique(x) |> sort
    counts = map(x_ -> sum(x .== x_), unqx)
    DataFrame(x = unqx, count = counts, svr = 1. .- cumsum(counts ./ sum(counts)))
end

IRTs = Array{Real, 1}(generate_samples(0.1, 5, 0.9, 2000))
IRTs_svf = svf(IRTs, 2)
plot(IRTs_svf.x, log10.(IRTs_svf.svr))
ylims!(-3, 0)

_ = sample(bi_exponential(IRTs), NUTS(), MCMCThreads(), 2, 4)
chains = sample(bi_exponential(IRTs), NUTS(), MCMCThreads(), 500, 4)
plot(chains)
