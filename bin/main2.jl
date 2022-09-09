@info "Initializing packages"
# using Stipple
# using StippleUI
# using StipplePlotly
# using PlotlyBase
# using Random
# using StatsBase: sample
# using Statistics: mean, std
#####################################

using Revise
using ScoringEngineDemo
using Serialization
using JLD2
using HTTP
using Sockets
using JSON3
using JSONTables
using DataFrames
using Stipple
using StippleUI
using StipplePlotly
using PlotlyBase
using Random

using ShapML
using Weave

using StatsBase: sample
using Statistics: mean, std

@reactive mutable struct Name <: ReactiveModel
  name::R{String} = "World!"
end

include("setup.jl")
includet("app.jl")
# includet("ui.jl")

function ui(model)
  page( model, class="container", [
      h1([
        "Hello "
        span("", @text(:name))
      ])

      p([
        "What is your name? "
        input("", placeholder="Type your name", @bind(:name))
      ])
    ]
  )
end

route("/") do
  model = Name |> init
  html(ui(model), context = @__MODULE__)
end

up(8000, "0.0.0.0"; async=false)    