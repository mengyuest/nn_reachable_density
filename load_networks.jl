using Flux, Tracker, JLD2, FileIO, MAT, NPZ

# Converts network weights and biases to just weights. i.e. only linear maps, not affine maps.
# Works for Dense Flux networks
function flux2augmented(NN)
	Weights = Vector{Array{Float64,2}}(undef,length(NN))
	for layer in 1:(length(NN)-1)
		Weights[layer] = vcat(hcat(Tracker.data(NN[layer].W), Tracker.data(NN[layer].b)), reshape(zeros(1+size(NN[layer].W,2)),1,:))
		Weights[layer][end,end] = 1
	end
	# last layer weight shouldn't carry forward the bias term. i.e. augmented but with last row removed
	Weights[length(NN)] = hcat(Tracker.data(NN[length(NN)].W), Tracker.data(NN[length(NN)].b))
	return Weights
end


# Evaluate network. Takes care of normalizing inputs and un-normalizing outputs
function eval_net(input, net_dict, copies::Int64)
	copies == -1 ? (return input) : nothing
	NN_out = (input - net_dict["X_mean"]') ./ net_dict["X_std"]'
    for layer = 1:length(net_dict["weights"])-1
        NN_out = relu.(net_dict["weights"][1,layer]*NN_out + net_dict["biases"][1,layer]')
    end
    output = vec( (net_dict["weights"][1,end]*NN_out + net_dict["biases"][1,end]') .* net_dict["Y_std"]' + net_dict["Y_mean"]' )
    return eval_net(output, net_dict, copies-1)
end


### Pyramid test function ###
function test_pyramid()
	W0 = [1 0; -1 0; 2 0; -2 0; 0 1; 0 -1; 0 2; 0 -2]
	b0 = [0; 1; -1; 1; 0; 1; -1; 1]

	W1 = 0.5*[1 1 -1 -1 0 0 0 0; 0 0 0 0 1 1 -1 -1]
	b1 = [0; 0]

	W2 = [-1 0; 1 0; 0 -1; 0 1; 1 -1; -1 1]
	b2 = [0; 0; 0; 0; 0; 0]

	W3 = 0.5*[1 1 1 1 -1 -1]
	b3 = [0]

	W4 = [1]
	b4 = [0]

	pyramid = Chain(Dense(W0,b0,relu), Dense(W1,b1,relu), Dense(W2,b2,relu), Dense(W3,b3,relu), Dense(W4,b4,identity) )
	return pyramid
	# return flux2augmented(pyramid)
end


# NN with input in, output out, hidden dim hdim, and hidden layers layers.
function test_random(in_d, out_d, hdim, layers)
	Weights = Vector{Array{Float64,2}}(undef,layers)
	Weights[1] = sqrt(2/515)*(2*rand(hdim, in_d) - rand(hdim, in_d))
	for i in 2:layers-1
		Weights[i] = sqrt(2/515)*(2*rand(hdim,hdim) - rand(hdim,hdim)) # Kaiming Initialization
	end
	Weights[end] = sqrt(2/515)*(2*rand(out_d, hdim) - rand(out_d, hdim))
	return Weights
end

rand_relu_layer(in_d, out_d)     = Dense(sqrt(2/515)*(2*rand(out_d, in_d) - rand(out_d, in_d)), sqrt(2/515)*(2*rand(out_d, 1) - rand(out_d, 1)) , relu)
rand_identity_layer(in_d, out_d) = Dense(sqrt(2/515)*(2*rand(out_d, in_d) - rand(out_d, in_d)), sqrt(2/515)*(2*rand(out_d, 1) - rand(out_d, 1)) , identity)

function test_random_flux(in_d, out_d, hdim, layers; Aₒ=[], bₒ=[], value=false)
	first_layer   = [rand_relu_layer(in_d, hdim)]
	hidden_layers = [rand_relu_layer(hdim, hdim) for _ in 2:layers-1]
	last_layer    = [rand_identity_layer(hdim, out_d)]
	net = vcat(first_layer, hidden_layers, last_layer)
	if value
		flux_net = Chain(net...)
		flux_netᵥ = Chain(net..., Dense(vcat(Aₒ,zeros(1,size(Aₒ,2))), vcat(bₒ,0), Maxout))
		weights = flux2augmented(flux_net)
		weightsᵥ = flux2augmented_value(flux_net, Aₒ, bₒ)
		return flux_net, flux_netᵥ, weights, weightsᵥ
	else
		flux_net = Chain(net...)
		weights = flux2augmented(flux_net)
		return flux_net, weights
	end
end


# Load ACAS Networks #
#=
net = matread("models/ACAS/ACASXU_run2a_1_1_batch_2000.mat")
"W"
"b"
"Maximum_of_Inputs"
"Minimum_of_Inputs"
"range_for_scaling" - this is len=6
"means_for_scaling" - this is len=6
"layer_sizes"
"size_Of_Largest_Layer"
"is_symmetric"
=#
# input  = [ρ, θ, ψ, v_own, v_int]
# output = [COC, weak right, strong right, weak left, strong left]
# a ∈ [1,...,5],   b ∈ [1,...,9]
function acas_net(a::Int64, b::Int64)
	name = string("models/ACAS/ACASXU_run2a_", a, "_", b, "_batch_2000.mat")
	acas_dict = matread(name)
	layers = [Dense(acas_dict["W"][i], acas_dict["b"][i], relu) for i in 1:length(acas_dict["W"])-1]
	layers = vcat(layers, [Dense(acas_dict["W"][end], acas_dict["b"][end], identity)])
	flux_net = Chain(layers...)
	return flux2augmented(flux_net), acas_dict
end


# Load Pendulum Networks #
#= 
net = matread("models/Pendulum/NN_params_pendulum_0_1s_1e7data_a15_12_2_L1.mat")
"weights"
"biases"
"X_mean"
"X_std"
"Y_mean"
"Y_std"
input  = [θ, θ_dot]_t
output = [θ, θ_dot]_t+1
=#
function pendulum_net(model::String, copies::Int64)
	net_dict = matread(model)
	σ_x = Diagonal(vec(net_dict["X_std"]))
	μ_x = vec(net_dict["X_mean"])
	σ_y = Diagonal(vec(net_dict["Y_std"]))
	μ_y = vec(net_dict["Y_mean"])

	layers = [Dense(net_dict["weights"][1], net_dict["biases"][1]', relu)]
	for net_copy in 1:copies # copies = 0 means just original network
		layers = vcat(layers, [Dense(net_dict["weights"][i], net_dict["biases"][i]', relu) for i in 2:length(net_dict["weights"])-1])
		W = net_dict["weights"][1]*inv(σ_x)*σ_y*net_dict["weights"][end]
		b = net_dict["weights"][1]*inv(σ_x)*(σ_y*net_dict["biases"][end]' + μ_y - μ_x) + net_dict["biases"][1]'
		layers = vcat(layers, [Dense(W, b, relu)])
	end
	layers = vcat(layers, [Dense(net_dict["weights"][i], net_dict["biases"][i]', relu) for i in 2:length(net_dict["weights"])-1])
	layers = vcat(layers, [Dense(net_dict["weights"][end], net_dict["biases"][end]', identity)])
	flux_net = Chain(layers...)
	str = string("pend_net_", copies, ".mat")
	matwrite(str,
	Dict(
	"W" => [Float64.(flux_net[i].W) for i in 1:length(flux_net)],
	"b" => [Float64.(flux_net[i].b) for i in 1:length(flux_net)],
	"range_for_scaling" => [1.0, 1.0, 1.0],
	"means_for_scaling" => [0.0, 0.0, 0.0]))
	return flux2augmented(flux_net), net_dict
end

function gcas_net(model::String)
	net_dict = matread(model)
	σ_x = Diagonal(vec(net_dict["X_std"]))
	μ_x = vec(net_dict["X_mean"])
	σ_y = Diagonal(vec(net_dict["Y_std"]))
	μ_y = vec(net_dict["Y_mean"])

	layers = [Dense(net_dict["weights"][1], net_dict["biases"][1]', relu)]
	layers = vcat(layers, [Dense(net_dict["weights"][2], net_dict["biases"][2]', relu)])
	layers = vcat(layers, [Dense(net_dict["weights"][3], net_dict["biases"][3]', relu)])
	layers = vcat(layers, [Dense(net_dict["weights"][4], net_dict["biases"][4]', identity)])

	flux_net = Chain(layers...)
	str = string("gcas_net.mat")
	matwrite(str,
	Dict(
	"W" => [Float64.(flux_net[i].W) for i in 1:length(flux_net)],
	"b" => [Float64.(flux_net[i].b) for i in 1:length(flux_net)],
	"range_for_scaling" => [1.0, 1.0, 1.0],
	"means_for_scaling" => [0.0, 0.0, 0.0]))
	return flux2augmented(flux_net), net_dict
end

function toy_net(model::String)
	net_dict = matread(model)
	σ_x = Diagonal(vec(net_dict["X_std"]))
	μ_x = vec(net_dict["X_mean"])
	σ_y = Diagonal(vec(net_dict["Y_std"]))
	μ_y = vec(net_dict["Y_mean"])

	layers = [Dense(net_dict["weights"][1], net_dict["biases"][1]', relu)]
	layers = vcat(layers,[Dense(net_dict["weights"][2], net_dict["biases"][2]', relu)])
	layers = vcat(layers, [Dense(net_dict["weights"][3], net_dict["biases"][3]', identity)])

	flux_net = Chain(layers...)
	str = string("toy_net.mat")
	matwrite(str,
	Dict(
	"W" => [Float64.(flux_net[i].W) for i in 1:length(flux_net)],
	"b" => [Float64.(flux_net[i].b) for i in 1:length(flux_net)],
	"range_for_scaling" => [1.0, 1.0, 1.0],
	"means_for_scaling" => [0.0, 0.0, 0.0]))
	return flux2augmented(flux_net), net_dict
end

function toy_simp_net(model::String)
	net_dict = matread(model)
	σ_x = Diagonal(vec(net_dict["X_std"]))
	μ_x = vec(net_dict["X_mean"])
	σ_y = Diagonal(vec(net_dict["Y_std"]))
	μ_y = vec(net_dict["Y_mean"])
	println(net_dict)
	println(net_dict["weights"][1], " ", net_dict["biases"][1], " ", net_dict["weights"][1], " ", net_dict["biases"][1])
	layers = [Dense(net_dict["weights"][1], net_dict["biases"][1]', relu)]
	layers = vcat(layers, [Dense(net_dict["weights"][2], net_dict["biases"][2]', identity)])

	flux_net = Chain(layers...)
	str = string("toy_simp_net.mat")
	matwrite(str,
	Dict(
	"W" => [Float64.(flux_net[i].W) for i in 1:length(flux_net)],
	"b" => [Float64.(flux_net[i].b) for i in 1:length(flux_net)],
	"range_for_scaling" => [1.0, 1.0],
	"means_for_scaling" => [0.0, 0.0]))
	return flux2augmented(flux_net), net_dict
end


function toy_2n2_net(model::String, num_hiddens::Int64, num_units::Int64, random_seed::Int64)
	net_dict = matread(model)
	σ_x = Diagonal(vec(net_dict["X_std"]))
	μ_x = vec(net_dict["X_mean"])
	σ_y = Diagonal(vec(net_dict["Y_std"]))
	μ_y = vec(net_dict["Y_mean"])
# 	println(net_dict)
# 	println(net_dict["weights"][1], " ", net_dict["biases"][1], " ", net_dict["weights"][1], " ", net_dict["biases"][1])
	layers = [Dense(net_dict["weights"][1], net_dict["biases"][1]', relu)]
	for i=1:num_hiddens-1
		layers = vcat(layers, [Dense(net_dict["weights"][i+1], net_dict["biases"][i+1]', relu)])
	end
	layers = vcat(layers, [Dense(net_dict["weights"][num_hiddens+1], net_dict["biases"][num_hiddens+1]', identity)])
	flux_net = Chain(layers...)
	str = string("toy_2n2_net.mat")
	matwrite(str,
	Dict(
	"W" => [Float64.(flux_net[i].W) for i in 1:length(flux_net)],
	"b" => [Float64.(flux_net[i].b) for i in 1:length(flux_net)],
	"range_for_scaling" => [1.0, 1.0],
	"means_for_scaling" => [0.0, 0.0]))
	return flux2augmented(flux_net), net_dict
end


function db_net(model::String)
	net_dict = matread(model)
	σ_x = Diagonal(vec(net_dict["X_std"]))
	μ_x = vec(net_dict["X_mean"])
	σ_y = Diagonal(vec(net_dict["Y_std"]))
	μ_y = vec(net_dict["Y_mean"])
	num_layers = length(net_dict["weights"])-1
	layers=[]
	for i =1:num_layers
		layers = vcat(layers, [Dense(net_dict["weights"][i], net_dict["biases"][i]', relu)])
	end
	layers = vcat(layers, [Dense(net_dict["weights"][num_layers+1], net_dict["biases"][num_layers+1]', identity)])
	flux_net = Chain(layers...)
# 	str = string("db.mat")
# 	matwrite(str,
# 	Dict(
# 	"W" => [Float64.(flux_net[i].W) for i in 1:length(flux_net)],
# 	"b" => [Float64.(flux_net[i].b) for i in 1:length(flux_net)],
# 	"range_for_scaling" => [1.0, 1.0],
# 	"means_for_scaling" => [0.0, 0.0]))
	return flux2augmented(flux_net), net_dict
end

function gcas_net(model::String)
		net_dict = matread(model)
	σ_x = Diagonal(vec(net_dict["X_std"]))
	μ_x = vec(net_dict["X_mean"])
	σ_y = Diagonal(vec(net_dict["Y_std"]))
	μ_y = vec(net_dict["Y_mean"])
	num_layers = length(net_dict["weights"])-1
	layers=[]
	for i =1:num_layers
		layers = vcat(layers, [Dense(net_dict["weights"][i], net_dict["biases"][i]', relu)])
	end
	layers = vcat(layers, [Dense(net_dict["weights"][num_layers+1], net_dict["biases"][num_layers+1]', identity)])
	flux_net = Chain(layers...)
	return flux2augmented(flux_net), net_dict
end