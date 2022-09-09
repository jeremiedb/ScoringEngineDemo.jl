@info "Initializing packages"
using ScoringEngineDemo
using JLD2
using Serialization

using CSV
using DataFrames
using Random

using Distributed
@everywhere using Statistics: mean
@everywhere using Flux
@everywhere using Flux: params, update!
@everywhere using Flux.Losses: logitbinarycrossentropy

results_path = joinpath(@__DIR__, "results")
isdir(results_path) || mkdir(results_path)
ENV["RESULTS_FILE"] = results_path

@info "nworkers" nworkers()
@info "workers" workers()

@info "Initializing assets"
const assets_path = joinpath(@__DIR__, "..", "assets")
const preproc_flux = JLD2.load(joinpath(assets_path, "preproc-flux.jld2"))["preproc"]
const adapter_flux = JLD2.load(joinpath(assets_path, "adapter-flux.jld2"))["adapter"]
# const preproc_flux = BSON.load(joinpath(assets_path, "preproc-flux.bson"), @__MODULE__)[:preproc]
# const adapter_flux = BSON.load(joinpath(assets_path, "adapter-flux.bson"), @__MODULE__)[:adapter]

df_tot = ScoringEngineDemo.load_data(joinpath(assets_path, "training_data.csv"))

# set target
transform!(df_tot, "claim_amount" => ByRow(x -> x > 0 ? 1.0f0 : 0.0f0) => "event")

# train/eval split
Random.seed!(123)
df_train, df_eval = ScoringEngineDemo.data_splits(df_tot, 0.9)

df_train = preproc_flux(df_train)
df_eval = preproc_flux(df_eval)

x_train, y_train = adapter_flux(df_train, true)
x_eval, y_eval = adapter_flux(df_eval, true)

@everywhere function loss(m, x, y)
    p = m(x)
    l = logitbinarycrossentropy(p, y; agg=mean)
    return l
end

# cb() = @show(loss(X_eval, y_eval))
@everywhere function logloss(data, m)
    logloss = 0.0
    count = 0
    for (x, y) in data
        p = m(x)
        logloss += logitbinarycrossentropy(p, y; agg=sum)
        count += size(x)[end]
    end
    return logloss / count
end

@everywhere function train_loop!(m, θ, opt, loss; dtrain, deval=nothing)
    for d in dtrain
        grads = gradient(θ) do
            loss(m, d...)
        end
        update!(opt, θ, grads)
    end
    metric = deval === nothing ? logloss(dtrain, m) : logloss(deval, m)
    println(metric)
end

@everywhere function fit(; nrounds, num_feats, h1, dtrain, deval)

    m = Chain(
        BatchNorm(num_feats),
        Dense(num_feats, h1, relu),
        Dropout(0.5),
        Dense(h1 => 1),
        x -> vec(x))

    opt = ADAM(1e-3)
    θ = params(m)

    for i in 1:nrounds
        train_loop!(m, θ, opt, loss, dtrain=dtrain, deval=deval)
    end

    eval_metric = logloss(deval, m)
    return (
        eval_metric=eval_metric,
        h1=h1,
        m=m)
end

num_feats = size(x_train, 1)
nrounds = 25

[@spawnat p x_train = x_train for p in workers()]
[@spawnat p y_train = y_train for p in workers()]
[@spawnat p x_eval = x_eval for p in workers()]
[@spawnat p y_eval = y_eval for p in workers()]

h1_list = 32:32:256
length(h1_list)
@time results = pmap(h1_list) do h1
    dtrain = Flux.Data.DataLoader((x_train, y_train), batchsize=1024, shuffle=true)
    deval = Flux.Data.DataLoader((x_eval, y_eval), batchsize=1024, shuffle=false)
    fit(; nrounds, num_feats, h1, dtrain, deval)
end

df_results = map(results) do n
    (h1=n[:h1], eval_metric=n[:eval_metric])
end |> DataFrame

m_best = results[findmin(df_results[:, :eval_metric])[2]][:m]

CSV.write(joinpath(results_path, "hyper-flux.csv"), df_results)
JLD2.save(joinpath(results_path, "model-flux.jld2"), Dict("model" => m_best))
serialize(joinpath(results_path, "model-flux.dat"), m_best)
# m1 = deserialize(joinpath(results_path, "model-flux.dat"))

# m = Chain(
#     BatchNorm(14),
#     Dense(14, 22, relu),
#     Dropout(0.5),
#     Dense(22 => 1),
#     x -> vec(x))

# using Serialization
# JLD2.save(joinpath(results_path, "model-flux-test.jld2"), Dict("model" => m))
# serialize(joinpath(results_path, "model-flux-test.dat"), m)
# deserialize(joinpath(results_path, "model-flux-test.dat"))

# x1 = rand(5)
# serialize("vec-test.dat", x1)
# x2 = deserialize("vec-test.dat")

# m4 = JLD2.load(joinpath(results_path, "model-flux-test.jld2"))["model"]
# m4 = Serialization.serialize(joinpath(results_path, "model-flux-test.jld2"))["model"]
