# %%
using CellListMap
using BenchmarkTools
using Plots
using ProgressBars
using Random
using JLD
using Printf

# %%
function P_NP_nonloc_int(pos_P, pos_NP, g, r, rcomp, domaincell)
    nP = size(pos_P, 2)
    nNP = size(pos_NP, 2)

    P_pg_benefit = zeros(nP)
    NP_pg_benefit = zeros(nNP)

    pos_P_NP = cat(pos_P, pos_NP, dims=2)

    nbs = neighborlist(pos_P_NP, r, unitcell=domaincell)

    nbs_P = zeros(nP)
    nbs_NP = zeros(nNP)

    for i in nbs
        if i[1] <= nP
            nbs_P[i[1]] += 1
            if i[2] <= nP    
                nbs_P[i[2]] += 1
            else
                nbs_NP[i[2]-nP] += 1
            end
        else
            nbs_NP[i[1]-nP] += 1
            nbs_NP[i[2]-nP] += 1
        end
    end

    for i in 1:nP
        P_pg_benefit[i] += g/(nbs_P[i]+1)
    end
    
    for i in nbs
        if i[1] <= nP
            if i[2] <= nP
                P_pg_benefit[i[1]] += g/(nbs_P[i[2]]+1)
                P_pg_benefit[i[2]] += g/(nbs_P[i[1]]+1)
            else
                NP_pg_benefit[i[2]-nP] += g/(nbs_P[i[1]]+1)
            end
        end
    end

    if rcomp != r
        nbs = neighborlist(pos_P_NP, rcomp, unitcell=domaincell)

        nbs_comp_P = zeros(nP)
        nbs_comp_NP = zeros(nNP)

        for i in nbs
            if i[1] <= nP
                nbs_comp_P[i[1]] += 1
                if i[2] <= nP    
                    nbs_comp_P[i[2]] += 1
                else
                    nbs_comp_NP[i[2]-nP] += 1
                end
            else
                nbs_comp_NP[i[1]-nP] += 1
                nbs_comp_NP[i[2]-nP] += 1
            end
        end

    else
        nbs_comp_P = nbs_P
        nbs_comp_NP = nbs_NP
    end

    return P_pg_benefit, NP_pg_benefit, nbs_comp_P, nbs_comp_NP
end

# %%
function pv_field(pos, pvs, strengths, domaincell, periodic_repeats=2)
    L = domaincell[1,1]
    
    v = zeros(size(pos))
    v_temp = zeros(2)
    
    for p in eachindex(pos[1,:])
        for n in eachindex(pvs[1,:])
            v_temp *= 0

            for i in -periodic_repeats:periodic_repeats
                dx = pos[1,p]-pvs[1,n]
                dy = pos[2,p]-pvs[2,n]
                
                if dx-i*L != 0 || dy != 0
                    v_temp[1] -= sin(2*pi*dy/L)/(cosh(2*pi*dx/L-2*pi*i)-cos(2*pi*dy/L))
                end
                if dy-i*L != 0 || dx != 0
                    v_temp[2] += sin(2*pi*dx/L)/(cosh(2*pi*dy/L-2*pi*i)-cos(2*pi*dx/L))
                end
            end

            v[:,p] += strengths[n]/(2*L)*v_temp
        end
    end
                
    return v
end

# %%
function pv_pvs(pvs, strengths, domaincell, periodic_repeats=2)
    
    L = domaincell[1,1]
    
    v = zeros(size(pvs))
    v_temp = zeros(2)
    
    for p in eachindex(pvs[1,:])
        for n in eachindex(pvs[1,:])
            if n == p
                continue
            end
            
            v_temp *= 0
            for i in -periodic_repeats:periodic_repeats
                dx = pvs[1,p]-pvs[1,n]
                dy = pvs[2,p]-pvs[2,n]

                if dx-i*L != 0 || dy != 0
                    v_temp[1] -= sin(2*pi*dy/L)/(cosh(2*pi*dx/L-2*pi*i)-cos(2*pi*dy/L))
                end
                if dy-i*L != 0 || dx != 0
                    v_temp[2] += sin(2*pi*dx/L)/(cosh(2*pi*dy/L-2*pi*i)-cos(2*pi*dx/L))
                end
            end

            v[:,p] += strengths[n]/(2*L)*v_temp
        end
    end
                
    return v
end

# %%
function pv_pos_pvs(pospvs, params, t)
    strengths, domaincell, periodic_repeats = params
    
    n_pv = size(strengths, 1)
    n_pos = size(pospvs, 2) - n_pv
    
    pvs = copy(pospvs[n_pos+1:n_pos+n_pv])
    
    L = domaincell[1,1]
    
    v = zeros(size(pospvs))
    v_temp = zeros(2)
    
    for p in 1:n_pos
        for n in 1:n_pv          
            v_temp *= 0
            for i in -periodic_repeats:periodic_repeats
                dx = pospvs[1,p]-pvs[1,n]
                dy = pospvs[2,p]-pvs[2,n]

                if dx-i*L != 0 || dy != 0
                    v_temp[1] -= sin(2*pi*dy/L)/(cosh(2*pi*dx/L-2*pi*i)-cos(2*pi*dy/L))
                end
                if dy-i*L != 0 || dx != 0
                    v_temp[2] += sin(2*pi*dx/L)/(cosh(2*pi*dy/L-2*pi*i)-cos(2*pi*dx/L))
                end
            end

            v[:,p] += strengths[n]/(2*L)*v_temp
        end
    end
    
    for p in n_pos+1:n_pos+n_pv
        for n in 1:n_pv
            if n == p
                continue
            end
            
            v_temp *= 0
            for i in -periodic_repeats:periodic_repeats
                dx = pospvs[1,p]-pvs[1,n]
                dy = pospvs[2,p]-pvs[2,n]

                if dx-i*L != 0 || dy != 0
                    v_temp[1] -= sin(2*pi*dy/L)/(cosh(2*pi*dx/L-2*pi*i)-cos(2*pi*dy/L))
                end
                if dy-i*L != 0 || dx != 0
                    v_temp[2] += sin(2*pi*dx/L)/(cosh(2*pi*dy/L-2*pi*i)-cos(2*pi*dx/L))
                end
            end

            v[:,p] += strengths[n]/(2*L)*v_temp
        end
    end
                
    return v
end

# %%
function next_reaction_time(pos_P, pos_NP, b, d, k, g, a, r, rcomp, domaincell)
    n_P = size(pos_P, 2)
    n_NP = size(pos_NP, 2)
    
    P_pg_benefit, NP_pg_benefit, nbs_P, nbs_NP = P_NP_nonloc_int(pos_P, pos_NP, g, r, rcomp, domaincell)

    rates = [d*(n_P+n_NP), (b-k)*n_P+sum(P_pg_benefit), b*n_NP+sum(NP_pg_benefit), a*sum(nbs_P), a*sum(nbs_NP)]

    rates_sum = sum(rates)

    tau = -log(rand())/rates_sum
    
    return tau
end

# %%
function exec_reaction(pos_P, pos_NP, b, d, k, g, a, r, rcomp, domaincell)
    nP = size(pos_P, 2)
    nNP = size(pos_NP, 2)
    
    P_pg_benefit, NP_pg_benefit, nbs_P, nbs_NP = P_NP_nonloc_int(pos_P, pos_NP, g, r, rcomp, domaincell)

    rates = [d*(nP+nNP), (b-k)*nP+sum(P_pg_benefit), b*nNP+sum(NP_pg_benefit), a*sum(nbs_P), a*sum(nbs_NP)]

    rates_sum = sum(rates)
    rates_cumsum = cumsum(rates)/sum(rates)
    
    reaction = searchsortedfirst(rates_cumsum, rand())
    
    if reaction == 1
        idx = rand(1:nP+nNP)
        if idx <= nP
            pos_P = pos_P[:,1:nP .!= idx]
        else
           pos_NP = pos_NP[:,1:nNP .!= idx-nP] 
        end

    elseif reaction == 2
        birth_P_cumsum = cumsum((b-k).+P_pg_benefit)/((b-k)*nP+sum(P_pg_benefit))
        new_P = pos_P[:,searchsortedfirst(birth_P_cumsum, rand())]
        pos_P = hcat(pos_P, new_P)
    
    elseif reaction == 3
        birth_NP_cumsum = cumsum(b.+NP_pg_benefit)/(b*nNP+sum(NP_pg_benefit))
        new_NP = pos_NP[:,searchsortedfirst(birth_NP_cumsum, rand())]
        pos_NP = hcat(pos_NP, new_NP)
                
    elseif reaction == 4
        comp_P_cumsum = cumsum(nbs_P)/sum(nbs_P)
        idx = searchsortedfirst(comp_P_cumsum, rand())
        pos_P = pos_P[:,1:nP .!= idx]
    
    elseif reaction == 5
        comp_NP_cumsum = cumsum(nbs_NP./sum(nbs_NP))
        idx = searchsortedfirst(comp_NP_cumsum, rand())
        pos_NP = pos_NP[:,1:nNP .!= idx]
    end
    
    return pos_P, pos_NP
end

# %%
function diffusion(pos, dt, D)
    return pos + randn(size(pos))*sqrt(2*D*dt)    
end

# %%
global n_pv = 5
global nP = 1000
global nNP = 0

seed = 0

Random.seed!(seed)

global pos_P = rand(2, nP)
global pos_NP = rand(2, nNP)
global pvs = rand(2, n_pv)

global strength = 1.
global strengths = rand(-1:2:1, n_pv)*strength

global D = 1e-2

global b = 2.0
global d = 1.0
global g = 0.
global k = 0.

global a = (b-d+g-k)/19

global r = 0.05
global rcomp = 0.1

global L = 1.
global domaincell = [L 0.0; 0.0 L]

global periodic_repeats = 2;

# %%
global t = [0.]
global pop_P = [size(pos_P, 2)]
global pop_NP = [size(pos_NP, 2)]

global t_ = 0

global save_step = 1000

global n_steps = 100000

folder = "abm_pg_pv_julia"

if folder ∉ readdir()
    mkdir(folder)
end

runname = @sprintf("b%.1fd%.1fg%.1fk%.1f_r%f_rcomp%f_v%f_D%f_seed%i", b, d, g, k, r, rcomp, strength, D, seed)

path = folder*"/"*runname

save("$path/pos_P_t=0=0.0.jld", "pos_P", pos_P)
save("$path/pos_NP_t=0=0.0.jld", "pos_NP", pos_NP)

macro_pop_file = open("$path/macro_pop.csv", "w")
write(macro_pop_file, "$(t[1]),$(pop_P[1]),$(pop_NP[1])\n")

global dt_vel = 0.01/mean(sum(pv_field(hcat(pos_P, pos_NP), pvs, strengths, domaincell, periodic_repeats).^2, dims=1).^0.5)

for i in ProgressBar(2:save_step*10)
    global reac_dt = next_reaction_time(pos_P, pos_NP, b, d, k, g, a, r, rcomp, domaincell)
    global t_ += reac_dt    
    
    if dt_vel > reac_dt
        global dt = reac_dt
        
        global pos_P += pv_field(pos_P, pvs, strengths, domaincell, periodic_repeats)*dt
        global pos_P = diffusion(pos_P, dt, D)
        global pos_NP += pv_field(pos_NP, pvs, strengths, domaincell, periodic_repeats)*dt
        global pos_NP = diffusion(pos_NP, dt, D)
        global pos_P = mod.(pos_P, L)
        global pos_NP = mod.(pos_NP, L)
        global pvs += pv_pvs(pvs, strengths, domaincell, periodic_repeats)*dt
        global pvs = mod.(pvs, L)
        
    else
        global dt = dt_vel
        
        for i in 1:dt_vel/reac_dt
            global pos_P += pv_field(pos_P, pvs, strengths, domaincell, periodic_repeats)*dt
            global pos_P = diffusion(pos_P, dt, D)
            global pos_NP += pv_field(pos_NP, pvs, strengths, domaincell, periodic_repeats)*dt
            global pos_NP = diffusion(pos_NP, dt, D)
            global pos_P = mod.(pos_P, L)
            global pos_NP = mod.(pos_NP, L)
            global pvs += pv_pvs(pvs, strengths, domaincell, periodic_repeats)*dt
            global pvs = mod.(pvs, L)
        end
        
        dt = mod(dt_vel, reac_dt)
        
        global pos_P += pv_field(pos_P, pvs, strengths, domaincell, periodic_repeats)*dt
        global pos_P = diffusion(pos_P, dt, D)
        global pos_NP += pv_field(pos_NP, pvs, strengths, domaincell, periodic_repeats)*dt
        global pos_NP = diffusion(pos_NP, dt, D)
        global pos_P = mod.(pos_P, L)
        global pos_NP = mod.(pos_NP, L)
        global pvs += pv_pvs(pvs, strengths, domaincell, periodic_repeats)*dt
        global pvs = mod.(pvs, L)
    end
    
    global pos_P, pos_NP = exec_reaction(pos_P, pos_NP, b, d, k, g, a, r, rcomp, domaincell)
    
    if i % save_step == 0
        global dt_vel = 0.01/mean(sum(pv_field(hcat(pos_P, pos_NP), pvs, strengths, domaincell, periodic_repeats).^2, dims=1).^0.5)
        append!(t, t_)
        append!(pop_P, size(pos_P, 2))
        append!(pop_NP, size(pos_NP, 2))
        
        write(macro_pop_file, "$(t[size(t,1)]),$(pop_P[size(t,1)]),$(pop_NP[size(t,1)])\n")
        
        save("$path/pos_P_t=$(trunc(Int, i/save_step))-$(t_).jld", "pos_P", pos_P)
        save("$path/pos_NP_t=$(trunc(Int, i/save_step))-$(t_).jld", "pos_NP", pos_NP)
    end
    
    if size(pos_P, 2) == 0 || size(pos_NP, 2) == 0
        if t_ != t[size(t, 1)]
            append!(t, t_)
            append!(pop_P, size(pos_P, 2))
            append!(pop_NP, size(pos_NP, 2))
        end
        break
    end
end

close(macro_pop_file)