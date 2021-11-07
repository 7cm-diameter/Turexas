using Distributions, DataFrames, Plots, Turing, StatsPlots
import Turexas.Environment as e
import Turexas.Agent as a

function step(agent::a.AbstractAgent, env::e.Bandit)
    p = a.policy(agent)
    action = rand(Categorical(p))
    reward = e.step(env, action)
    a.update(agent, action, reward)
    d = DataFrame(action = action, reward = reward)
    return hcat(d, DataFrame(agent.q', :auto))
end

function run(agent::a.AbstractAgent, env::e.Bandit, trial::Int64)
    return reduce(vcat, map(_ -> step(agent, env), 1:trial))
end

# Run MCMC with a small number of samples
function warmup_sampler(result::AbstractDataFrame, model, k::Int64)
    actions = Array{Real, 1}(result.action)
    rewards = Array{Real, 1}(result.reward)
    sample(model(actions, rewards, k), NUTS(), 2)
end

function fit(result::AbstractDataFrame, model, k::Int64, iter::Int64)
    actions = Array{Real, 1}(result.action)
    rewards = Array{Real, 1}(result.reward)
    sample(model(actions, rewards, k), NUTS(), iter)
end

# generate an environment which is shared with all simulations
bandit = e.Bandit([0.2, 0.8])
k = length(bandit.p)

# Run simulation and model fitting with Q-learning
@model QLearningModel(actions::Array{Real, 1}, rewards::Array{Real, 1}, k::Int64) = begin
    T = length(actions)
    α ~ Beta(1, 1)
    β ~ Gamma(1, 100)
    agent = a.spawn(α, β, k)
    for t in 1:T
        action = actions[t]
        reward = rewards[t]
        p = a.policy(agent)
        actions[t] ~ Categorical(p)
        a.update(agent, action, reward)
    end
end

q_agent = a.spawn(0.05, 2., k)
result = run(q_agent, bandit, 500)
# Run MCMC with a small number of samples in advance because it takes a long time to run MCMC the first time.
_ = warmup_sampler(result, QLearningModel, k)
chains = fit(result, QLearningModel, k, 500)
plot(chains)
