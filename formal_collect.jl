using Plots
# include("reach.jl")
include("reach_new.jl")
using JSON

function minmax_to_A_b(s_mins, s_maxs, C, d)
    dim = length(s_mins)
    A = zeros(dim*2, dim)
	for i=1:dim
		A[i*2-1, i] = -1.0
		A[i*2, i] = 1.0
	end
    b = zeros(dim*2,)
    for i=1:dim
       b[i*2-1] = -s_mins[i]
       b[i*2] = s_maxs[i]
    end
#     return A, b
    newA = A * C
	newb = b - A * d


	for i=1:dim
	    newb[i*2-1] = newb[i*2-1] / abs(newA[i*2-1,i])
	    newA[i*2-1,:] = newA[i*2-1,:] / abs(newA[i*2-1,i])
		newb[i*2] = newb[i*2] / abs(newA[i*2,i])
	    newA[i*2,:] = newA[i*2,:] / abs(newA[i*2,i])
	end

	println(A)
	println(b)
	println(newA)
	println(newb)
	return newA, newb
end

#TODO Exp-1 Ground Robot Navigation
function input_robot(weights; net_dict=[])
    nt = 50
    dt = 0.05
    s_mins=[-1.8, -1.8, 0.0,  1.0, 0.00]
    s_maxs=[-1.2, -1.2, 1.57, 1.5, (nt-1) * dt]
	A, b = minmax_to_A_b(s_mins, s_maxs, Float64.(Diagonal(vec(net_dict["X_std"]))), vec(net_dict["X_mean"]))
	return A, b
end

function output_robot(weights; net_dict=[])
    rhomin = 0.0
	rhomax = 100000.0
    s_mins=[rhomin, -2.0, -2.0, -0.0035, 0.0596]
    s_maxs=[rhomax,  2.0,  2.0, 1.6063, 3.5290]
	A, b = minmax_to_A_b(s_mins, s_maxs, Diagonal(vec(net_dict["Y_std"])), vec(net_dict["Y_mean"]))
	return A, b
end

#TODO Exp-2 Double Integrator
function input_dint(weights; net_dict=[])
    nt = 10
    dt = 1.0
	# condtion-0
	s_mins=[-0.5, -1.0, 0.00]
    s_maxs=[4.0, 1.0, (nt-1) * dt]

	# condition-1
	s_mins=[-0.5, -1.0, 0.00]
    s_maxs=[4.0, 1.0, (nt-1) * dt]

	# condition-2

	A, b = minmax_to_A_b(s_mins, s_maxs, Float64.(Diagonal(vec(net_dict["X_std"]))), vec(net_dict["X_mean"]))
	return A, b
end

function output_dint(weights; net_dict=[])
    rhomin = 0.0
	rhomax = 100000.0
    s_mins=[rhomin, -2.9111, -1.5602]
    s_maxs=[rhomax,  6.4872, 2.9066]
	A, b = minmax_to_A_b(s_mins, s_maxs, Diagonal(vec(net_dict["Y_std"])), vec(net_dict["Y_mean"]))
	return A, b
end

#TODO Exp-3 FACTEST Car-Model
function input_car(weights; net_dict=[])
    nt = 50
    dt = 0.1
	# cond-0
#     s_mins=[-2.1, -2.1, -0.1, 0.0, 0.00]
#     s_maxs=[2.1, 2.1, 0.1, 5.0, (nt-1) * dt]

# 	# cond-1
# 	s_mins=[-1.1, -1.1, -0.1, 0.0, 0.00]
#     s_maxs=[1.1, 1.1, 0.1, 2.0, (nt-1) * dt]

# 	# cond-2
# 	s_mins=[-0.5, -0.5, -0.1, 0.0, 0.00]
#     s_maxs=[0.5, 0.5, 0.1, 1.0, (nt-1) * dt]

# 	# cond-3
# 	s_mins=[-2.1, -2.1, -0.1, 0.0, 0.00]
#     s_maxs=[2.1, 2.1, 0.1, 0.5, (nt-1) * dt]

#     # cond-4
# 	s_mins=[-1.1, -1.1, -0.1, 0.0, 0.00]
#     s_maxs=[1.1, 1.1, 0.1, 0.6, (nt-1) * dt]

## cond-5
	s_mins=[-1.1, -1.1, -0.1, 0.0, 0.00]
    s_maxs=[1.1, 1.1, 0.1, 0.5, (nt-1) * dt]

	A, b = minmax_to_A_b(s_mins, s_maxs, Float64.(Diagonal(vec(net_dict["X_std"]))), vec(net_dict["X_mean"]))
	return A, b
end

function output_car(weights; net_dict=[])
    rhomin = 0.0
	rhomax = 100000.0
    s_mins=[rhomin, -2.0996, -3.7112, -2.4299, 0.0000]
    s_maxs=[rhomax, 4.4255, 3.7996, 2.2531, 0.4999]
	A, b = minmax_to_A_b(s_mins, s_maxs, Diagonal(vec(net_dict["Y_std"])), vec(net_dict["Y_mean"]))
	return A, b
end

#TODO Exp-4 Quadrotor ReachLP
function input_quad(weights; net_dict=[])
    nt = 12
    dt = 0.1
    s_mins=[4.6500, 4.6500, 2.9500, 0.9400, -0.0500, -0.4999, 0.00]
    s_maxs=[4.7500, 4.7500, 3.0500, 0.9600, 0.0500, 0.4997, (nt-1) * dt]
	A, b = minmax_to_A_b(s_mins, s_maxs, Float64.(Diagonal(vec(net_dict["X_std"]))), vec(net_dict["X_mean"]))
	return A, b
end

function output_quad(weights; net_dict=[])
    rhomin = 0.0
	rhomax = 100000.0
    s_mins=[rhomin, 4.6050, 3.6878, -2.5585, -1.1009, -1.7322, -9.8067]
    s_maxs=[rhomax, 5.0652, 4.7570, 3.0984, 0.9600, 0.0500, 0.4997]
	A, b = minmax_to_A_b(s_mins, s_maxs, Diagonal(vec(net_dict["Y_std"])), vec(net_dict["Y_mean"]))
	return A, b
end

#TODO Exp-5 Inverted Pendulum
function input_pend(weights; net_dict=[])
    nt = 50
    dt = 0.02
    s_mins=[-2.1, -5.5, -2.0, -2.0, 0.00]
    s_maxs=[2.1, 5.5, 2.0, 2.0, (nt-1) * dt]
# 	s_mins=[-1.1, -1.5, -1.0, -1.0, 0.00]
#     s_maxs=[1.1, 1.5, 1.0, 1.0, (nt-1) * dt]

#     s_mins=[-0.1, -0.5, -0.4, -0.1, 0.00]
#     s_maxs=[0.1, 0.5, 0.4, 0.1, (nt-1) * dt]

	A, b = minmax_to_A_b(s_mins, s_maxs, Float64.(Diagonal(vec(net_dict["X_std"]))), vec(net_dict["X_mean"]))
	return A, b
end

function output_pend(weights; net_dict=[])
    rhomin = 0.0
	rhomax = 100000.0
    s_mins=[rhomin, -4.9239, -8.6915, -1.9996, -1.9997]
    s_maxs=[rhomax, 4.9786, 8.8851, 1.9991, 1.9989]
	A, b = minmax_to_A_b(s_mins, s_maxs, Diagonal(vec(net_dict["Y_std"])), vec(net_dict["Y_mean"]))
	return A, b
end

#TODO Exp-6 Platoon
function input_toon(weights; net_dict=[])
    nt = 50
    dt = 0.15
    s_mins=[-0.1000, -0.1000, -0.0999, -0.1000, -0.0999, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.0100, 0.00]
    s_maxs=[0.1000, 0.1000, 0.1000, 0.0999, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.0100, (nt-1) * dt]
	A, b = minmax_to_A_b(s_mins, s_maxs, Float64.(Diagonal(vec(net_dict["X_std"]))), vec(net_dict["X_mean"]))
	return A, b
end

function output_toon(weights; net_dict=[])
    rhomin = 0.0
	rhomax = 100000.0
    s_mins=[rhomin, -0.2179, -1.0363, -0.2990, -0.1319, -0.1178, -0.1972, -0.1825, -0.7121, -0.3046, -0.1208, -0.1000, -0.1627, -0.1664, -1.7694, -0.7851, -0.0100]
    s_maxs=[rhomax, 1.0758, 0.1476, 0.1144, 0.1732, 0.1510, 0.1305, 0.1334, 0.1760, 0.1782, 0.1273, 0.1179, 0.7441, 0.3757, 0.1339, 0.1039, 0.0100]
	A, b = minmax_to_A_b(s_mins, s_maxs, Diagonal(vec(net_dict["Y_std"])), vec(net_dict["Y_mean"]))
	return A, b
end

#TODO Exp-7 ACC
#TODO needs normalization
function input_acc(weights; net_dict=[])
    nt = 50
    dt = 0.1
    s_mins=[59.0006, 26.0003, -0.0100, 30.0000, -0.0100, -10.1000, -1.9996, 0.00]
    s_maxs=[61.9995, 29.9995, 0.0100, 30.4999, 0.0100, -9.9000, 1.9991, (nt-1) * dt]
	A, b = minmax_to_A_b(s_mins, s_maxs, Float64.(Diagonal(vec(net_dict["X_std"]))), vec(net_dict["X_mean"]))
	return A, b
end

function output_acc(weights; net_dict=[])
    rhomin = 0.0
	rhomax = 100000.0
    s_mins=[rhomin, 36.0363, 14.6485, -7.9906, 11.4882, -4.4711, -15.7584, -1.9996]
    s_maxs=[rhomax, 81.6502, 32.8234, 9.9802, 30.5005, 0.0100, 12.6333, 1.9991]
	A, b = minmax_to_A_b(s_mins, s_maxs, Diagonal(vec(net_dict["Y_std"])), vec(net_dict["Y_mean"]))
	return A, b
end

#TODO Exp-8 F-16 GCAS
#TODO needs normalization
function input_gcas(weights; net_dict=[])
    nt = 106
    dt = 0.03333
#     s_mins=[560.0040, 0.0750, -0.0100, -0.1000, -0.9996, -0.0100, -0.0100, -0.0100, -0.0100, -0.0100, -0.0100, 1150.0158, 0.0002, 0.00]
#     s_maxs=[599.9902, 0.1000, 0.0100, -0.0750, -0.5002, 0.0100, 0.0100, 0.0100, 0.0100, 0.0100, 0.0100, 1199.9896, 0.9994, (nt-1) * dt]
	s_mins=[595.0040, 0.0750, -0.0100, -0.1000, -0.9996, -0.0100, -0.0100, -0.0100, -0.0100, -0.0100, -0.0100, 1155.0158, 0.0002, 0.00]
    s_maxs=[599.9902, 0.1000, 0.0100, -0.0750, -0.5002, 0.0100, 0.0100, 0.0100, 0.0100, 0.0100, 0.0100, 1159.9896, 0.9994, (nt-1) * dt]
	A, b = minmax_to_A_b(s_mins, s_maxs, Float64.(Diagonal(vec(net_dict["X_std"]))), vec(net_dict["X_mean"]))
	return A, b
end

function output_gcas(weights; net_dict=[])
    rhomin = 0.0
	rhomax = 100000.0
    s_mins=[rhomin, 504.5229, 0.0042, -0.0101, -0.1001, -1.0334, -0.1000, -0.0100, -0.3550, -0.0236, -0.0100, -98.0891, -157.6262, 0.0002]
    s_maxs=[rhomax, 614.8140, 0.2720, 0.0100, -0.0293, 0.3185, 0.0265, 0.1478, 0.6173, 0.0274, 1942.5127, 12.7566, 1199.9896, 8.8157]
	A, b = minmax_to_A_b(s_mins, s_maxs, Diagonal(vec(net_dict["Y_std"])), vec(net_dict["Y_mean"]))
	return A, b
end


function input_vdp(weights; net_dict=[])
	nt = 50
    #dt = 0.05
	dt = 0.10
	s_mins=[-2.5, -2.5, 0.00]
    s_maxs=[2.5, 2.5, (nt-1) * dt]
	A, b = minmax_to_A_b(s_mins, s_maxs, Float64.(Diagonal(vec(net_dict["X_std"]))), vec(net_dict["X_mean"]))
	return A, b
end

function output_vdp(weights; net_dict=[])
	rhomin = 0.0
	rhomax = 100000.0
    s_mins=[rhomin, -5, -5.0]
    s_maxs=[rhomax, 5.0, 5.0]
	A, b = minmax_to_A_b(s_mins, s_maxs, Diagonal(vec(net_dict["Y_std"])), vec(net_dict["Y_mean"]))
	return A, b
end

function input_circ(weights; net_dict=[])
	nt = 50
    dt = 0.05
	s_mins=[0.5, 0.5, 0.00]
    s_maxs=[1.0, 1.0, (nt-1) * dt]
	A, b = minmax_to_A_b(s_mins, s_maxs, Float64.(Diagonal(vec(net_dict["X_std"]))), vec(net_dict["X_mean"]))
	return A, b
end

function output_circ(weights; net_dict=[])
	rhomin = 0.0
	rhomax = 100000.0
    s_mins=[rhomin, -2.0, -2.0]
    s_maxs=[rhomax, 2.0, 2.0]
	A, b = minmax_to_A_b(s_mins, s_maxs, Diagonal(vec(net_dict["Y_std"])), vec(net_dict["Y_mean"]))
	return A, b
end

function input_kop(weights; net_dict=[])
	nt = 80
    dt = 0.125
	s_mins=[0.0, -2.0, -2.0, 0.0]
    s_maxs=[2.0, 2.0, 2.0, (nt-1) * dt]
	A, b = minmax_to_A_b(s_mins, s_maxs, Float64.(Diagonal(vec(net_dict["X_std"]))), vec(net_dict["X_mean"]))
	return A, b
end

function output_kop(weights; net_dict=[])
	rhomin = 0.0
	rhomax = 100000.0
    s_mins=[rhomin, -2.0, -2.0, -2.0]
    s_maxs=[rhomax, 2.0, 2.0, -2.0]
	A, b = minmax_to_A_b(s_mins, s_maxs, Diagonal(vec(net_dict["Y_std"])), vec(net_dict["Y_mean"]))
	return A, b
end

function write_to_json(state2constraints, filepath; exists_c=false)
	A_list=[]
	b_list=[]
	if exists_c
		c_list=[]
	end
	for state in keys(state2constraints)
		if exists_c
			A, b, c = state2constraints[state]
			push!(c_list, c)
		else
			A, b = state2constraints[state]
		end
		push!(A_list, A)
		push!(b_list, b)
	end
	if exists_c
		dict1 = Dict("A" => A_list, "b" => b_list, "c" => c_list)
	else
		dict1 = Dict("A" => A_list, "b" => b_list)
	end
	stringdata = JSON.json(dict1)
	open(filepath, "w") do f
	    write(f, stringdata)
    end
end


###########################
######## SCRIPTING ########
###########################
if length(ARGS)==3
	println("using ARGS~")
    model = ARGS[1]
	exp_mode=ARGS[2]
	prefix = ARGS[3]
else
	exit()
	model = "./models/g0531-011427_DBnew_lr2k_32x2_preT_rho_t.mat"
	exp_mode="robot"
	prefix = "debug_"
end
weights, net_dict = db_net(model)
if exp_mode == "robot"
    Aᵢ, bᵢ = input_robot(weights, net_dict=net_dict)
    Aₒ, bₒ = output_robot(weights, net_dict=net_dict)
elseif exp_mode == "dint"
    Aᵢ, bᵢ = input_dint(weights, net_dict=net_dict)
    Aₒ, bₒ = output_dint(weights, net_dict=net_dict)
elseif exp_mode == "car"
    Aᵢ, bᵢ = input_car(weights, net_dict=net_dict)
    Aₒ, bₒ = output_car(weights, net_dict=net_dict)
elseif exp_mode == "quad"
    Aᵢ, bᵢ = input_quad(weights, net_dict=net_dict)
    Aₒ, bₒ = output_quad(weights, net_dict=net_dict)
elseif exp_mode == "pend"
    Aᵢ, bᵢ = input_pend(weights, net_dict=net_dict)
    Aₒ, bₒ = output_pend(weights, net_dict=net_dict)
elseif exp_mode == "toon"
    Aᵢ, bᵢ = input_toon(weights, net_dict=net_dict)
    Aₒ, bₒ = output_toon(weights, net_dict=net_dict)
elseif exp_mode == "acc"
    Aᵢ, bᵢ = input_acc(weights, net_dict=net_dict)
    Aₒ, bₒ = output_acc(weights, net_dict=net_dict)
elseif exp_mode == "gcas"
	Aᵢ, bᵢ = input_gcas(weights, net_dict=net_dict)
    Aₒ, bₒ = output_gcas(weights, net_dict=net_dict)
elseif exp_mode == "vdp"
    Aᵢ, bᵢ = input_vdp(weights, net_dict=net_dict)
    Aₒ, bₒ = output_vdp(weights, net_dict=net_dict)
elseif exp_mode == "circ"
    Aᵢ, bᵢ = input_circ(weights, net_dict=net_dict)
    Aₒ, bₒ = output_circ(weights, net_dict=net_dict)
elseif exp_mode == "kop"
	Aᵢ, bᵢ = input_kop(weights, net_dict=net_dict)
    Aₒ, bₒ = output_kop(weights, net_dict=net_dict)
end


# Run algorithm
@time begin
# state2input, state2output, state2map, state2backward = compute_reach_ym(weights, Aᵢ, bᵢ, [Aₒ], [bₒ], reach=true, back=true, verification=false)
state2input, state2output, state2map, state2backward = compute_reach(weights, Aᵢ, bᵢ, [Aₒ], [bₒ], reach=true, back=true, verification=false)
end
@show length(state2input)

write_to_json(state2input, string("data/models/exp_",exp_mode,"_",prefix,"state2input.json"))
write_to_json(state2output, string("data/models/exp_",exp_mode,"_",prefix,"state2output.json"), exists_c=true)
write_to_json(state2map, string("data/models/exp_",exp_mode,"_",prefix,"state2map.json"))
write_to_json(state2backward[1], string("data/models/exp_",exp_mode,"_",prefix,"state2backward.json"))