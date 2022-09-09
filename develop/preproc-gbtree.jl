using Revise
using ScoringEngineDemo
using DataFrames
using Statistics
using StatsBase: sample
using CairoMakie
using EvoTrees
using Random
using BSON
using JLD2

global targetname = "event"

df_tot = ScoringEngineDemo.load_data("assets/training_data.csv")

# set target
transform!(df_tot, "claim_amount" => ByRow(x -> x > 0 ? 1.0f0 : 0.0f0) => "event")

norm_feats = ["vh_age", "vh_value", "vh_speed", "vh_weight", "drv_age1",
    "pol_no_claims_discount", "pol_coverage", "density", 
    "drv_exp_yrs", "pol_duration", "pol_sit_duration",
    "drv_sex1", "has_drv2", "is_drv2_male"]

# train/eval split
Random.seed!(123)
df_train, df_eval = ScoringEngineDemo.data_splits(df_tot, 0.9)

preproc = ScoringEngineDemo.build_preproc(df_train, norm_feats = norm_feats)
adapter = ScoringEngineDemo.build_adapter_gbt(norm_feats, targetname)

df_train_pre = preproc(df_train)

density(collect(skipmissing(df_train_pre.vh_age)))
density(collect(skipmissing(df_train_pre.drv_age1)))

JLD2.save("assets/preproc-gbt.jld2", Dict("preproc" => preproc))
JLD2.save("assets/adapter-gbt.jld2", Dict("adapter" => adapter))

# BSON.bson("assets/preproc-gbt.bson", Dict(:preproc => preproc))
# BSON.bson("assets/adapter-gbt.bson", Dict(:adapter => adapter))

df_train = preproc_gbt(df_train)
df_eval = preproc_gbt(df_eval)

x_train, y_train = adapter_gbt(df_train, true)
x_eval, y_eval = adapter_gbt(df_eval, true)

CSV.write("assets/df_train.csv", df_train)
CSV.write("assets/df_eval.csv", df_eval)

CSV.write("assets/x_train.csv", DataFrame(x_train, norm_feats))
CSV.write("assets/y_train.csv", DataFrame([y_train], [targetname]))
CSV.write("assets/x_eval.csv", DataFrame(x_eval, norm_feats))
CSV.write("assets/y_eval.csv", DataFrame([y_eval], [targetname]))
