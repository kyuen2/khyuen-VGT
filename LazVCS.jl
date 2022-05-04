module LazVCS

# Core Modules
using HDF5,PyPlot,Statistics,LsqFit,PyCall,FFTW,StatsBase,Images

# Gaussian Quadrature integration, needed since we need the numerical integration in julia fitting.
using QuadGK    # one-dimensional quick GK, for anything inside D_vz
#using Cubature  # multi-dimensional cubature, for P_v

# LazTech-VGT
using LazCore,LazType,LazThermal,LazIO,LazThermal_Kritsuk
using LazRHT_investigation,LazCFA

# LazTech-VGT-new
using LazNewCore 
using LazDDA
using LazFilaments

# Multithreading
using Base.Threads # enables @threads

	##############################################################################
	#
	# Copyright (c) 2020
	# Alexey Chepurnov and Alex Lazarian
	# Based on A. Chepurnov's Mathematica 4 code and Chepurnov et.al (2015) paper
	# Coded and tested by Ka Ho Yuen 
	# All Rights Reserved.
	#
	# This program is free software: you can redistribute it and/or modify
	# it under the terms of the GNU General Public License as published by
	# the Free Software Foundation, either version 3 of the License, or
	# (at your option) any later version.
	# 
	# This program is distributed in the hope that it will be useful,
	# but WITHOUT ANY WARRANTY; without even the implied warranty of
	# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	# GNU General Public License for more details.
	# You should have received a copy of the GNU General Public License
	# along with this program.  If not, see <https://www.gnu.org/licenses/>.
	#
	##############################################################################

	#==
	 KH: This is v1.3+ only
	==#

	function amp(k::Vec;dims=2)
		l=length(k)
		a=0;
		for i in 1:l
			a+=k^dims
		end
		return a^(1.0./dims)
	end

	## We are following Alexey's code and definitions in `P1em_plot.nb`


	function I1(r::Number,L_inj::Number,α_v::Number;infvalue=1e9)
	    k_inj = 2*pi/L_inj
	    x,ϵ=quadgk(q->exp(-k_inj^2*r^2/q^2)/q^(α_v-2)*(1-sin(q)/q),0,infvalue)
	    return x
	end

	function I2(r::Number,L_inj::Number,α_v::Number;infvalue=1e9)
	    k_inj = 2*pi/L_inj
	    x,ϵ=quadgk(q->exp(-k_inj^2*r^2/q^2)/q^(α_v-2)*(1-3/q^2*(sin(q)/q-cos(q))),0,infvalue)
	    return x
	end

	function I1_0(α_v::Number;smallvalue=1e-5,infvalue=1e12)
	    x,ϵ=quadgk(q->1/q^(α_v-2)*(1-sin(q)/q),smallvalue,infvalue)
	    return x
	end

	function I2_0(α_v::Number;smallvalue=1e-5,infvalue=1e12)
	    x,ϵ=quadgk(q->1/q^(α_v-2)*(1-3/q^2*(sin(q)/q-cos(q))),smallvalue,infvalue)
	    return x
	end

	function Is(r::Number,L_inj::Number,α_v::Number;infvalue=1e9,γ=0,c=0.000020944)
	    k_inj = 2*pi/L_inj
	    if ((k_inj*r)>c)
	        x=2*I1(r,L_inj,α_v,infvalue=infvalue)-2/3*(1-γ)*I2(r,L_inj,α_v,infvalue=infvalue)
	    else
	        x=2*I1_0(α_v,infvalue=infvalue)-2/3*(1-γ)*I2_0(α_v,infvalue=infvalue)
	    end
	    return x
	end

	function Ic(r::Number,L_inj::Number,α_v::Number;infvalue=1e9,γ=0,c=0.000020944)
	    k_inj = 2*pi/L_inj
	    if ((k_inj*r)>c)
	        x=2*γ*I1(r,L_inj,α_v,infvalue=infvalue)+4/3*(1-γ)*I2(r,L_inj,α_v,infvalue=infvalue)
	    else
	        x=2*γ*I1_0(α_v,infvalue=infvalue)+4/3*(1-γ)*I2_0(α_v,infvalue=infvalue)
	    end
	    return x
	end

	function D_z(R::Number,z::Number,V_0::Number,L_inj::Number,α_v::Number;r_0=1e-10,infvalue=1e9,γ=0,c=0.000020944)
	    r=sqrt(R^2+z^2);
	    r>r_0   ? rr=r : rr=r_0;
	    r_lim=3*L_inj; # r_lim = 3*(2pi/k_inj)
	    if (r<r_lim) 
	        RR=R;
	        zz=z;
	    else
	        rr=r_lim;
	        RR=0;
	        zz=r_lim;
	    end
	    return 2*pi*V_0*rr^(α_v-5)*(RR^2*Is(r,L_inj,α_v,infvalue=infvalue,γ=γ,c=c)+zz^2*Ic(r,L_inj,α_v,infvalue=infvalue,γ=γ,c=c))
	end


	function P_v(ν::Number,V_0::Number,L_inj::Number,α_v::Number;r_0=1e-10,infvalue=1e9,γ=0,c=0.000020944)
	    # Using successive integration
	    function P_v_θ(r::Number,ν::Number,V_0::Number,L_inj::Number,α_v::Number;r_0=1e-10,infvalue=1e9,γ=0,c=0.000020944)
			x,ϵ_x=quadgk(θ->4*pi*r^2*sin(θ)*exp(-ν^2*D_z(r*sin(θ),r*cos(θ),V_0,L_inj,α_v,r_0=r_0,infvalue=infvalue,γ=γ,c=c)),0,pi/2)
			return x
		end
	    y,ϵ_y=quadgk(r->P_v_θ(r,ν,V_0,L_inj,α_v,r_0=r_0,infvalue=infvalue,γ=γ,c=c),0,infvalue)
	    return y
	end

    function dP_vdV_0(ν,V_0,L_inj,α_ν;dx=1e-5,dV_0=dx*V_0,infvalue=1e9,γ=0,c=0.000020944)
        # 2nd order differentiatoion
        Pp=P_v_gk(ν,V_0+dV_0,L_inj,α_ν,infvalue=infvalue,γ=γ,c=c);
        Pm=P_v_gk(ν,V_0-dV_0,L_inj,α_ν,infvalue=infvalue,γ=γ,c=c);
        P0=P_v_gk(ν,V_0     ,L_inj,α_ν,infvalue=infvalue,γ=γ,c=c);
        return (Pp.+Pm.-2.0.*P0)./2.0./dV_0;
    end
    #
    function dP_vdα_ν(ν,V_0,L_inj,α_ν;dx=1e-5,dα_0=dx*α_ν,infvalue=1e9,γ=0,c=0.000020944)
        # 2nd order differentiatoion
        Pp=P_v_gk(ν,V_0,L_inj,α_ν+dα_0,infvalue=infvalue,γ=γ,c=c);
        Pm=P_v_gk(ν,V_0,L_inj,α_ν-dα_0,infvalue=infvalue,γ=γ,c=c);
        P0=P_v_gk(ν,V_0,L_inj,α_ν     ,infvalue=infvalue,γ=γ,c=c);
        return (Pp.+Pm.-2.0.*P0)./2.0./dα_0;
    end
    #
    function dP_vdL_inj(ν,V_0,L_inj,α_ν;dx=1e-5,dL_inj=dx*L_inj,infvalue=1e9,γ=0,c=0.000020944)
        # 2nd order differentiatoion
        Pp=P_v_gk(ν,V_0,L_inj+dL_inj,α_ν,infvalue=infvalue,γ=γ,c=c);
        Pm=P_v_gk(ν,V_0,L_inj-dL_inj,α_ν,infvalue=infvalue,γ=γ,c=c);
        P0=P_v_gk(ν,V_0,L_inj       ,α_ν,infvalue=infvalue,γ=γ,c=c);
        return (Pp.+Pm.-2.0.*P0)./2.0./dL_inj;
    end

    # Basic GD algorithm to minimize the error function:
    # f(x) = sum_ν [(P_predicted(ν,x) - P_real(ν))^2]
    # ∇f = \sum_ν 2∇P_predicted(ν,x)*(P_predicted(ν,x) - P_real(ν))
    
    function f(ν_0::Vector,P_0::Vector,V_0,L_inj,α_ν;infvalue=1e9,γ=0,c=0.000020944)
        ϵp=0;
        for i in 1:length(ν_0)
            P_mock=P_v_gk(ν_0[i],V_0,L_inj,α_ν,infvalue=infvalue,γ=γ,c=c);
            ϵp+=(P_0[i].-P_mock).^2.0
        end
        return ϵp
    end
    
    function ∇f(ν_0::Vector,P_0::Vector,V_0,L_inj,α_ν;dx=1e-5,infvalue=1e9,γ=0,c=0.000020944)
        eep=zeros(3)
        for i in 1:length(ν_0)
            ν=ν_0[i];
            P_mock=P_v_gk( ν,V_0,L_inj,α_ν,infvalue=infvalue,γ=γ,c=c);
            ∇f1=dP_vdV_0(  ν,V_0,L_inj,α_ν,infvalue=infvalue,γ=γ,c=c);
            ∇f2=dP_vdα_ν(  ν,V_0,L_inj,α_ν,infvalue=infvalue,γ=γ,c=c)
            ∇f3=dP_vdL_inj(ν,V_0,L_inj,α_ν,infvalue=infvalue,γ=γ,c=c)
            eep[1]+=2*∇f1*(P_mock.-P_0[i])
            eep[2]+=2*∇f2*(P_mock.-P_0[i])
            eep[3]+=2*∇f3*(P_mock.-P_0[i])
        end
        return eep
    end

	function fit_P1_gd(ν_0::Vector,P_0::Vector,V_0::Number,L_inj::Number,α_ν::Number;dx=1e-5,ϵ_0=1e-10,infvalue=1e9,γ=0,c=0.000020944)
	    # KH: Here we have to perform a 3-variable gradient descent
	    #     which possibly involves the numerical differentiation.
	    #     We here uses the integrand value to perform the differentations.
	  
	    # Initial values : V_0,L_inj,α_v
	    
	  
	    # Vectorize the initial condition
	    global x=[V_0,L_inj,α_ν]
	    global ii=0;
	    n_ν=length(ν_0)
	    
	    ϵ=f(ν_0,P_0,x...,infvalue=infvalue,γ=γ,c=c) 
	    while (ϵ>ϵ_0)
	        ∇ϵ=∇f(ν_0,P_0,x...,dx=dx,infvalue=infvalue,γ=γ,c=c);
	        x-=∇ϵ.*dx
	        ϵ=f(ν_0,P_0,x...,infvalue=infvalue,γ=γ,c=c) 
	        ii+=1;
	        println("Iteration ",ii,":",x[1]," ",x[2]," ",x[3],"ϵ=",ϵ,"∇ϵdx",∇ϵ.*dx)
	    end
	    return x
	end



#==

	#==
	KH: Re-implementation of the VCS technique that takes into account 
	     of the regular motions. See Chepurnov et.al 2015
	     https://iopscience.iop.org/article/10.1088/0004-637X/810/1/33/pdf
 
	The key is to fit Eqs (12), (16), (20) in fitting the one dimensional V-axis power spectrum

	See also
		Chepurnov & Lazarian (2006) : https://arxiv.org/pdf/astro-ph/0611463.pdf
	==#


	#==
	Eq(12):
	P(k_v) ~ f^2 int dr g(r) C(r) exp(-k_v^2/2*D-ik_v*bz)

	where g(r) is the geometric factor, which assumed to be a composite gaussian
	      b    is the regular shear
	==#


	function sf1d_np(pp::Vec)

	end

	function cf_spectral(p::Cube)
		# assuming ppv
		nx,ny,nv=size(p)
		pcf=zeros(typeof(1+im),nx,ny,nv);
		for i in 1:nx, j in 1:ny
			pp=reshape(p[i,j,:],length(p[i,j,:]));
			pcf[i,j,:]=sf1d_np(pp)
		end
		return pcf
	end

	#==
	Eq(16):
	D = 2 int dk (1-e^ikr) z_i z_j F_ij (sum over repeated indices)
	  KH (Aug 24): should be r_i r_j from the raw code?
	where
          F_ij = V_inj^2/k^α_v exp(-k_inj^2/k^2) (δ_ij-k_i k_j/k^2)
          which requires three fitting parameters:

          V_inj : Injection Velocity
          k_inj : (Wavenumber) of the injection scale
          α_v   : (3D) velocity spectral index

	==#

	#==
    Eq(20):
    C ~ int δ^2 F^2 e^ikr dk +1
    where 
          F = 1/k^α_v exp(-k_inj^2/k^2)
          δ = (ΔS/S)^2 1/(|FFT(g)|^2 F)

    From Alexey:
          when density contribution is zero -> we only need a constant ϵ_0 (aka 1 here)
          i.e. δ=0 in Eq.(19)
	==#

	#function F2_ϵ(L_inj,α_v,kx,ky,kz)
#	#	k_inj = (2*pi)./L_inj;
#	#	k_abs=amp(k);
#	#	C=exp(-k_inj^2/k_abs^2)/k_abs^α_v
#	#	return C
#	#end
#
#	#function δ()
#
#	#	
#
	#end

	function ij_init(k::Vec;dims=length(k))
		l=length(k)
		F=zeros(l,l)
		k_abs=amp(k)
		for i in 1:l, j in 1:l
			if (i==j)
				F[i,j]+=1
			end
			F[i,j]+=-k[i]*k[j]/k_abs^2;
		end
		return F
	end

	function F_ij(V_inj,L_inj,α_v,kx,ky,kz)
		k=[kx,ky,kz];
		k_inj = (2*pi)./L_inj;
		k_abs=amp(k)
		F=ij_init(k)
		C=V_inj^2/k_abs^α_v*exp(-k_inj^2/k_abs^2);
		return C.*F
	end

==#






end #module LazVCS