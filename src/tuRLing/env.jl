module Environment
export Bandit, step

struct Bandit
    p::Array{Real, 1}
end

function step(env::Bandit, action::Int64)
    k = length(env.p)
    rewards = env.p .>= rand(k)
    return rewards[action]
end

end#module
