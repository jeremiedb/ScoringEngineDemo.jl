using Revise
using ScoringEngineDemo
using DataFrames
using Statistics
using StatsBase: sample
using CairoMakie
using Random

using BSON
using JLD2

global targetname = "event"

df_tot = ScoringEngineDemo.load_data("assets/training_data.csv")

# minimal DF verbs
dfg = groupby(df_tot, "pol_coverage")
df = combine(dfg, [:vh_age, :vh_value] .=> mean ∘ skipmissing .=> [:vh_age, :vh_value])
select(df, ["pol_coverage", "vh_value"])

# set target
transform!(df_tot, "claim_amount" => ByRow(x -> x > 0 ? 1.0f0 : 0.0f0) => "event")

norm_feats = ["vh_age", "vh_value", "vh_speed", "vh_weight", "drv_age1",
    "pol_no_claims_discount", "pol_coverage", "density",
    "drv_exp_yrs", "pol_duration", "pol_sit_duration",
    "drv_sex1", "has_drv2", "is_drv2_male"]

# train/eval split
Random.seed!(123)
df_train, df_eval = ScoringEngineDemo.data_splits(df_tot, 0.9)

density(collect(skipmissing(df_train.vh_age)))
density(collect(skipmissing(df_train.drv_age1)))

preproc = ScoringEngineDemo.build_preproc(df_train, norm_feats=norm_feats)
adapter = ScoringEngineDemo.build_adapter_flux(norm_feats, targetname)

df_train_pre = preproc(df_train)

density(collect(skipmissing(df_train_pre.vh_age)))
density(collect(skipmissing(df_train_pre.drv_age1)))

JLD2.save("assets/preproc-flux.jld2", Dict("preproc" => preproc))
JLD2.save("assets/adapter-flux.jld2", Dict("adapter" => adapter))

# BSON.bson("assets/preproc-flux.bson", Dict(:preproc => preproc))
# BSON.bson("assets/adapter-flux.bson", Dict(:adapter => adapter))