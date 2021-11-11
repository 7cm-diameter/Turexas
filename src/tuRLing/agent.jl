module Agent
using Distributions
export AbstractAgent, QLearningAgent, spawn, update, policy

abstract type AbstractAgent end

mutable struct QLearningAgent <: AbstractAgent
    q::Array{Real, 1}
    α::Real
    β::Real
end

function spawn(α::Real, β::Real, k::Int64)
    q = Array{Real, 1}(zeros(k))
    return QLearningAgent(q, α, β)
end

function update(agent::QLearningAgent, action::Real, reward::Real)
    agent.q[action] += agent.α * (reward - agent.q[action])
end

function policy(agent::QLearningAgent)
    qmax = maximum(agent.q)
    qexp = exp.(agent.β .* (agent.q .- qmax))
    return qexp ./ sum(qexp)
end

mutable struct ParticleFilterAgent <: AbstractAgent
    npartcles::Real
    particles::Array{Real, 2}
    q::Array{Real, 1}
    σ2::Real
    β::Real
end

function sigmoid(x::Real)
    1 / (1 + exp(-x))
end

function spawn(npartcles::Real, σ2::Real, β::Real, k::Real)
    particles = rand(Normal(), (k, npartcles))
    probs = sigmoid.(particles)
    q = sum(probs, dims=2)[:, 1] / size(probs, 2)
    ParticleFilterAgent(npartcles, particles, q, σ2, β)
end

function choice(x::Array{Real, 1}, p::Array{Real, 1}, n::Real)
    ret = Array{Real, 1}(undef, n)
    p = cumsum(p)
    for i in 1:n
        th = rand()
        for (x_, p_) in zip(x, p)
           if th <= p_
               ret[i] = x_
               break
           end
       end
    end
    return ret
end

function update(agent::ParticleFilterAgent, action::Real, reward::Real)
    noiz = Array{Real, 1}(rand(Normal(0, agent.σ2), agent.npartcles))
    particles = Array{Real, 1}(agent.particles[action, :] + noiz)
    probs = sigmoid.(particles)
    loglikelihood = pdf.(Bernoulli.(probs), reward)
    lnormalize = Array{Real, 1}(loglikelihood ./ sum(loglikelihood))
    agent.q[action] = sum(lnormalize .* probs)
    agent.particles[action, :] = choice(particles, lnormalize, agent.npartcles)
end

function policy(agent::ParticleFilterAgent)
    qmax = maximum(agent.q)
    qexp = exp.(agent.β .* (agent.q .- qmax))
    p = qexp ./ sum(qexp)
    p ./ sum(p)
end

end#modulie
