using Lux, NNlib, Optimisers, Random, Statistics, Zygote, Plots

model = Lux.Chain(Lux.Dense(1 => 100, relu), Lux.Dense(100 => 100, tanh), Lux.Dense(100 => 100, sigmoid), Lux.Dense(100 => 100, relu), Lux.Dense(100 => 100, relu), Lux.Dense(100 => 1))
opt = ADAM(0.01)

function loss_function(model, ps, st, data)
    y_pred, st = Lux.apply(model, data[1], ps, st)
    mse_loss = mean(abs2, y_pred .- data[2])
    return mse_loss, st, ()
end

rng = MersenneTwister()
Random.seed!(rng, 12345)
tstate = Lux.Training.TrainState(rng, model, opt; transform_variables=Lux.gpu)
vjp_rule = Lux.Training.ZygoteVJP()

function main(tstate::Lux.Training.TrainState, vjp::Lux.Training.AbstractVJP, train_data::Tuple,
    val_data::Tuple, epochs::Int)
    train_data = train_data .|> Lux.gpu
    val_data = val_data .|> Lux.gpu

    val_loss = Inf
    final_tstate = 0
    for epoch in 1:epochs
        grads, loss, stats, tstate = Lux.Training.compute_gradients(vjp, loss_function,
                                                                train_data, tstate)
        @info epoch=epoch loss=loss
        tstate = Lux.Training.apply_gradients(tstate, grads)
        val_fit = Lux.cpu(Lux.apply(tstate.model, Lux.gpu(val_data[1]), tstate.parameters, tstate.states)[1])
        val_loss_ = sum(abs2, val_data[1].- val_fit)
        if val_loss_ < val_loss
            val_loss = val_loss_
            final_tstate = tstate
        end
    end
    return final_tstate
end


function generate_data(rng::AbstractRNG)
    x = reshape(collect(range(0, 1000, 1000)), (1, 1000))
    # y = sin.(x) .+ randn(rng, (1, 1000)) .* 0.1f0
    y = 60 .* sin.(1e-2 .* x) .+ 200
    return (x, y)
end

normalization(array, reference_array) = 2 .* (array .- minimum(reference_array)) ./ (maximum(reference_array) - minimum(reference_array)) .- 1


(x, y) = generate_data(rng)
y = normalization(y, y)
x = normalization(x, x)
(x_train, y_train), (x_val, y_val) = splitobs((x, y); at=0.8, shuffle=false)
(x_val, y_val), (x_test, y_test) = splitobs((x_val, y_val); at=0.5, shuffle=false)


tstate = main(tstate, vjp_rule, (x_train, y_train), (x_val, y_val), 2000)
y_pred = Lux.cpu(Lux.apply(tstate.model, Lux.gpu(x), tstate.parameters, tstate.states)[1])

plot(vec(x), vec(y), label="True")
plot!(vec(x), vec(y_pred), label="Fit")
