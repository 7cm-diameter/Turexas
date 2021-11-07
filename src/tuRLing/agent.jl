module Agent
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
end#module
