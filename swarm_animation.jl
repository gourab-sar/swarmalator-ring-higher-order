# Importing the required packages


using DelimitedFiles
#using LightGraphs
using LinearAlgebra 
using PyPlot
using Random
#using BenchmarkTools
using Distributions
using StatsBase
using  OrdinaryDiffEq ###DifferentialEquations
using DiffEqCallbacks


# System size and intrinsic frequencies


N=100000;
#Random.seed!(123)
d=Cauchy()
omega=rand(d,N);
omega=omega-ones(N)*mean(omega)
v=rand(d,N);
v=v-ones(N)*mean(v);

# defining the function for simulation


function kuramoto(du, u, pp, t)

       u1 = @view u[1:N]
       du1 = @view du[1:N]
       u2 = @view u[N+1:2*N]
       du2 = @view du[N+1:2*N]
       
       j1,j2,k1,k2=pp

        z11 = Array{Complex{Float64},1}(undef, 1)
        z11= mean(exp.((u1+u2)*1im))
        z11c=conj(z11)
         
        z12 = Array{Complex{Float64},1}(undef, 1)
        z12= mean(exp.((u1-u2)*1im))
        z12c=conj(z12)
   
        z21 = Array{Complex{Float64},1}(undef, 1)
        z21= mean(exp.((u1+u2)*2im))

        z22 = Array{Complex{Float64},1}(undef, 1)
        z22= mean(exp.((u1-u2)*2im))
        

        ####### equ of motion\n",
        @. du1  = v + j1/2 * ( ( imag(z11*exp((-1im)*(u1+u2)) ))+( imag(z12*exp((-1im)*(u1-u2)) )) ) + j2/2*( ( imag(z21 * z11c * exp((-1im) * (u1+u2)) )) 
        + ( imag(z22 * z12c * exp((-1im) * (u1-u2)) )))
        @. du2 =  omega + k1/2 * ( ( imag(z11*exp((-1im)*(u1+u2)) ))-( imag(z12*exp((-1im)*(u1-u2)) )) ) + k2/2*( ( imag(z21 * z11c * exp((-1im) * (u1+u2)) )) 
        - ( imag(z22 * z12c * exp((-1im) * (u1-u2)) )))
       end;


       # Integration times 
       
       
#dt = 0.01 # time step
dts = 1 # save time
ti = 0.0
tt = 150
tf = 200
nt = Int(div(tt,dts))
nf = Int(div(tf,dts))

tspan = (ti, tf); # time interval


# Initial Conditions

#u0=[-pi*ones(N)+rand(N)*2*pi;-pi*ones(N)+rand(N)*2*pi]; ## for forward 
#u0=[pi*1.99*ones(N)+rand(N)*0.01*pi;pi*1.99*ones(N)+rand(N)*0.01*pi]; ## for backward
#u0=[(-pi*ones(N)+rand(N)*2*pi)*0.001;-(-pi*ones(N)+rand(N)*2*pi)*0.001] ; ## for backward

x0=(-pi*ones(N)+rand(N)*2*pi);
θ0=-x0;
u0=[x0;θ0];

# Parameter values
k1 =6.5 
j1= 1
k2= 9
j2= 5

# Integration


    
@time begin
    # defining problem for ODE solver
    prob = ODEProblem(kuramoto, u0, tspan, (j1,j2,k1,k2))
    
      """  saved_values1= SavedValues(Float64, Float64)
       function saver1(u,t,integrator)
          _re1=mean(cos.(u[1:N]+u[N+1:2*N]))
          _im1=mean(sin.(u[1:N]+u[N+1:2*N]))
          out1=sqrt(_re1^2 + _im1^2)
       end
     cb1 = SavingCallback(saver1, saved_values1,saveat=ti:dts:tf)

      saved_values2= SavedValues(Float64, Float64)
       function saver2(u,t,integrator)
          _re2=mean(cos.(u[1:N]-u[N+1:2*N]))
          _im2=mean(sin.(u[1:N]-u[N+1:2*N]))
          out2=sqrt(_re2^2 + _im2^2)
       end
     cb2 = SavingCallback(saver2, saved_values2,saveat=ti:dts:tf)
     
      cbs = CallbackSet(cb1, cb2)
    

     sol = solve(prob, Tsit5(),progress=true,callback = cbs,saveat=ti:dts:tf);
     """
     sol = solve(prob, Tsit5(),progress=true,saveat=[tf]); 
     
     #sp_fwd_t= saved_values1.saveval;
     #sm_fwd_t= saved_values2.saveval;

        
end;     


# Saving Last time point

writedlm("j1=$j1,j2=$j2,k1=$k1,k2=$k2,N=$N,backward_last_time_point.txt",-pi*ones(2*N)+mod2pi.(sol[:,end]))

# Visualizing the last time instance

x = -pi*ones(N)+mod2pi.(sol[1:N,end])
θ = -pi*ones(N)+mod2pi.(sol[N+1:2*N,end])

wpe= mean(exp.((x+θ)*1im))
wme= mean(exp.((x-θ)*1im)) 
rpe=abs.(wpe) 
ψpe=angle.(wpe)
rme=abs.(wme)
ψme=angle.(wme);



clf()
figure(figsize=(6, 12))
##################################    
subplot(311)
xlim(-1.2,1.2)
ylim(-1.2,1.2) 
xlabel("x", fontsize=20)
ylabel("y", fontsize=20)
ticks = [-1, 0, 1]
tick_labels = ["-1", "0", "1"]
PyPlot.xticks(ticks, tick_labels, fontsize=15)
PyPlot.yticks(ticks, tick_labels, fontsize=15)
scatter(cos.(x),sin.(x),c=θ,cmap="jet",vmin=-π, vmax=π,s=50)
colorbar_obj = colorbar(label="θ", ticks=[-π, 0, π])
colorbar_obj.ax.set_yticklabels(["-π", "0", "π"], fontsize=15)
colorbar_obj.set_label("θ", fontsize=20)
pte1x=[0,rpe*cos(ψpe)]
pte1y=[0,rpe*sin(ψpe)]
pte2x=[0,rme*cos(ψme)]
pte2y=[0,rme*sin(ψme)]
scatter(pte1x[2],pte1y[2],c="black",s=100)
scatter(pte2x[2],pte2y[2],c="magenta",s=100)
plot(pte1x,pte1y, c="black" )
plot(pte2x,pte2y, c="magenta" )
####################################################

subplot(312)
p2=scatter(x,θ,c="blue",s=0.1)
xlabel("x")
ylabel("θ")
ylim(-π,π)
xlim(-π,π)
title("N=$N,j1=$j1,k1=$k1,j2=$j2,k2=$k2")
ticks = [-π, 0, π]
tick_labels = ["-π", "0", "π"]
PyPlot.xticks(ticks, tick_labels)
PyPlot.yticks(ticks, tick_labels)
####################################################
#subplot(313)
#p3=plot(sp_fwd_t,c="red")
#plot(sm_fwd_t,c="blue")
#ylim(-0.1,1)
#xlabel("t")
#ylabel("S±")
subplots_adjust(hspace=0.5)
gcf()


# Unpacking the solution


l=length(sol)
x1=Array{Vector{Float64}}(undef,l)
θ1=Array{Vector{Float64}}(undef,l)
wp=Array{Complex{Float64}}(undef,l)
wm=Array{Complex{Float64}}(undef,l)
for j in 1:l
x1[j] = -pi*ones(N)+mod2pi.(sol[1:N,j])
θ1[j] = -pi*ones(N)+mod2pi.(sol[N+1:2*N,j])
wp[j]= mean(exp.((x1[j]+θ1[j])*1im))
wm[j]= mean(exp.((x1[j]-θ1[j])*1im))        
end

rp1=abs.(wp)
ψp1=angle.(wp)
rm1=abs.(wm)
ψm1=angle.(wm);


# Making animation and save as mp4 video

using PyCall
@pyimport matplotlib.animation as anim
using PyPlot

using Base64

function showmp4(filename)
    open(filename) do f
        base64_video = base64encode(f)
        display("text/html", """<video controls src="data:video/mp4;base64,$base64_video"></video>""")
    end
end


fig = figure(figsize=(21, 6))
ax = PyPlot.gca()


function plotw(t)
       
    clf()
        
    subplot(131)
    title("\$t = $t\$",fontsize=20)
    xlim(-1.2,1.2)
    ylim(-1.2,1.2) 
    xlabel("x", fontsize=20)
    ylabel("y", fontsize=20)
    ticks = [-1, 0, 1]
    tick_labels = ["-1", "0", "1"]
    PyPlot.xticks(ticks, tick_labels, fontsize=15)
    PyPlot.yticks(ticks, tick_labels, fontsize=15)
    scatter(cos.(x1[Int(t)+1]),sin.(x1[Int(t)+1]),c=θ1[Int(t)+1],cmap="jet",vmin=-π, vmax=π,s=50)
    colorbar_obj = colorbar(label="θ", ticks=[-π, 0, π])
    colorbar_obj.ax.set_yticklabels(["-π", "0", "π"], fontsize=15)
    colorbar_obj.set_label("θ", fontsize=20)
    pt1x=[0,rp1[t+1]*cos(ψp1[t+1])]
    pt1y=[0,rp1[t+1]*sin(ψp1[t+1])]
    pt2x=[0,rm1[t+1]*cos(ψm1[t+1])]
    pt2y=[0,rm1[t+1]*sin(ψm1[t+1])]
    scatter(pt1x[2],pt1y[2],c="black",s=100)
    scatter(pt2x[2],pt2y[2],c="magenta",s=100)
    plot(pt1x,pt1y, c="black" )
    plot(pt2x,pt2y, c="magenta" )
    gca()[:set_aspect]("equal")
    
    subplot(132)
    title("\$t = $t\$",fontsize=20)
    xlim(-π,π)
    ylim(-π,π) 
    xlabel("x", fontsize=20)
    ylabel("θ", fontsize=20)
    ticks = [-π, 0, π]
    tick_labels = ["-π", "0", "π"]
    PyPlot.xticks(ticks, tick_labels, fontsize=15)
    PyPlot.yticks(ticks, tick_labels, fontsize=15)
    scatter(x1[Int(t)+1],θ1[Int(t)+1],c="blue",s=2.0)
    gca()[:set_aspect]("equal")
    
    
    subplot(133)
    title("\$t = $t\$",fontsize=20)
    xlim(-π,π)
    ylim(-π,π)
    xlabel("ξ", fontsize=20)
    ylabel("η", fontsize=20)
    ticks = [-π, 0, π]
    tick_labels = ["-π", "0", "π"]
    PyPlot.xticks(ticks, tick_labels, fontsize=15)
    PyPlot.yticks(ticks, tick_labels, fontsize=15)
    scatter(x1[Int(t)+1]+θ1[Int(t)+1],x1[Int(t)+1]-θ1[Int(t)+1],c="blue",s=2.0)
    gca()[:set_aspect]("equal")
    
    subplots_adjust(wspace=0.5) ## for horizontal spacing 
    #subplots_adjust(hspace=0.8) ## for vertical spacing
    plot()
end


n=l # frame
interval=200 # time between two frames in milli seconds

# k=0,1,...,frames-1
function animate(k)
    plotw(k)
end

function init()
    plotw(0)
end



# Increase the bitrate and DPI for high resolution
bitrate = -1
dpi = 100

withfig(fig) do
    global myanim = anim.FuncAnimation(fig, animate, frames=n, init_func=init, interval=interval, blit=true)
    myanim[:save]("test5.mp4", dpi=dpi, bitrate=bitrate, extra_args=["-vcodec", "libx264"])
end

showmp4("test5.mp4")



#withfig(fig) do
#    global myanim = anim.FuncAnimation(fig, animate, frames=l*n+1, init_func=init, interval=interval, blit=true)
#    myanim[:save]("test4.mp4", bitrate=-1, extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv480p"])
#end

#showmp4("test4.mp4")


