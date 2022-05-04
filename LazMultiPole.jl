module LazMultiPole
using HDF5,Statistics,LsqFit,PyCall,FFTW,StatsBase,Images
using LazCore,LazType,LazIO
using LazRHT # Filament moduile
using LazThermal,LazThermal_Kritsuk # Thermal broadening module
using LazDust # The core package for PvB
using LazGAC # Gradient aand Curvature
using LazCh5 # the ch5 functions
using Base.Threads # Parallelism

##############################################################################
#
# Copyright (c) 2020
# Ka Ho Yuen, Alex Lazarian and Dmitri Pogosyan
# All Rights Reserved.
#
# ​This program is free software: you can redistribute it and/or modify
# ​it under the terms of the GNU General Public License as published by
# ​the Free Software Foundation, either version 3 of the License, or
# ​(at your option) any later version.
# ​
# This program is distributed in the hope that it will be useful,
# ​but WITHOUT ANY WARRANTY; without even the implied warranty of
# ​MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# ​GNU General Public License for more details.
# ​You should have received a copy of the GNU General Public License
# ​along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
##############################################################################

#==
 KH: The aim of this package is implement two of KH's idea in acquiring the inclination angles, namely the "Kandel method" and the "Planck method"
 To achieve this we need to implement some of the functions
 ==#

export multipole_2d,multipole_2d_Mat,kandel_method,Gaensler_SPG,planck_method,multipole_2D_3ptsf


# Kandel method
# MNRAS 464, 3617-3635 (2017)
# Uses D2/D0, D4/D0 to obtain gamma!


function multipole_2d(A::Mat;n=10)
	Af=fftshift(sf(A))
	nx,ny=size(A);
	nr=round(Int,sqrt(nx^2+ny^2)/2)
	Dn=zeros(Complex{Float64},nr,div(n,2)+1)
	for nnn in 1:div(n,2)+1, i in 1:nx, j in 1:ny
		nn=(nnn-1)*2
		idx=i-div(nx,2);
		jdx=j-div(ny,2);
		r=round(Int,sqrt(idx.^2.0.+jdx.^2.0));
        if ((r>0) & (r<=nr))
        	phi = atan(jdx./idx)
        	Dn[r,nnn] += Af[i,j]*exp(-im.*nn*phi)/2/pi
        end
	end
	# KH: Output is D0,D2,D4. Default to D10.
	return Dn
end

function multipole_2D_3ptsf(A::Mat;n=10)
	nx,ny=size(A)
	Af=threept_sf2_bruteforce(A;nnx=nx,nny=ny)
	nr=round(Int,sqrt(nx^2+ny^2)/2)
	Dn=zeros(Complex{Float64},nr,div(n,2)+1)
	for nnn in 1:div(n,2)+1, i in 1:nx, j in 1:ny
		nn=(nnn-1)*2
		idx=i-div(nx,2);
		jdx=j-div(ny,2);
		r=round(Int,sqrt(idx.^2.0.+jdx.^2.0));
        if ((r>0) & (r<=nr))
        	phi = atan(jdx./idx)
        	Dn[r,nnn] += Af[i,j]*exp(-im.*nn*phi)/2/pi
        end
	end
	# KH: Output is D0,D2,D4. Default to D10.
	return Dn
end


function multipole_2D_3ptsf_montecarlo(A::Mat;n=10,fac=1e-3)
	nx,ny=size(A)
	Af=threept_sf2_bruteforce_montecarlo(A;nnx=nx,nny=ny,fac=fac)
	nr=round(Int,sqrt(nx^2+ny^2)/2)
	Dn=zeros(Complex{Float64},nr,div(n,2)+1)
	for nnn in 1:div(n,2)+1, i in 1:nx, j in 1:ny
		nn=(nnn-1)*2
		idx=i-div(nx,2);
		jdx=j-div(ny,2);
		r=round(Int,sqrt(idx.^2.0.+jdx.^2.0));
        if ((r>0) & (r<=nr))
        	phi = atan(jdx./idx)
        	Dn[r,nnn] += Af[i,j]*exp(-im.*nn*phi)/2/pi
        end
	end
	# KH: Output is D0,D2,D4. Default to D10.
	return Dn
end

function kandel_method(Ch::Mat;n=10)
	Dn=multipole_2d(Ch;n=10)
	# second indices 1->0, 2->2, 3->4
	D0=Dn[:,1]
	D2=Dn[:,2]
	D4=Dn[:,3]
	return D2./D0, D4./D0
end



# Planck method
# Planck 2018 L11, King et.al 2018, Chen et.al 2019
# uses Eq E45 to obtain gamma!

function Gaensler_SPG(I::Mat,Q::Mat,U::Mat)
	# Find per-pixel |∇Φ| from IQU, using Gaenlser et.al (2011) method
	QI=Q./I;
	UI=U./I;
	qx=(circshift(QI,[1,0]).+circshift(QI,[-1,0]).-2.0.*QI)./2
	qy=(circshift(QI,[0,1]).+circshift(QI,[0,-1]).-2.0.*QI)./2
	ux=(circshift(UI,[1,0]).+circshift(UI,[-1,0]).-2.0.*UI)./2
	uy=(circshift(UI,[0,1]).+circshift(UI,[0,-1]).-2.0.*UI)./2
	return sqrt.(qx.^2.0.+qy.^2.0.+ux.^2.0.+uy.^2.0)
end

function planck_method(I::Mat,Q::Mat,U::Mat;dn=20,angular_scale=1)
	# KH: There are several assumptions in this particular algorithm
	# (1) p_max ~ 0.26 (Planck 2018 XII)
	# (2) N=7, which is the average number of layers along the line of sight as fitted by Planck 2016, 2018 
	# (3) I replaced f(δ) in Planck 2018 XII to Ma without a proportionality constant
	# From these three assumptions, we have
	# X = 2/sqrt(21) M_A(δ)/δ|∇Φ|  
	# (4) Chen et.al (2019): sin(γ)^2 = 5X/(4+X)
    delphi = Gaensler_SPG(I,Q,U)
    phi=0.5.*atan.(U,Q);
    nx,ny=size(delphi)
    nnx=div(nx,dn)
    nny=div(ny,dn)
    delphi_dn=zeros(nnx,nny);
    Ma_dn=zeros(nnx,nny)
    for i in 1:nnx, j in 1:nny
    	is=(i-1)*dn+1
    	ie=i*dn
    	js=(j-1)*dn+1
    	je=j*dn
    	delphi_dn[i,j]=mean(delphi[is:ie,js:je])
    	Ma_dn[i,j]=cstd(phi[is:ie,js:je])
    end
    X = (2/(sqrt(21))/dn).*Ma_dn./delphi_dn
    X2= (5.0.*X)./(4.0.+X)
    X2[X2.<0].=0
    Y = sqrt.(X2);
    gamma =zeros(size(Y))
    for i in 1:nnx, j in 1:nny
    	if (abs(Y[i,j])<=1)
    		gamma[i,j]=asin(Y[i,j])
    	else
    		gamma[i,j]=NaN
    	end
    end
    #gamma=asin.(Y);
    return gamma
end

function planck_method_per_pixel(I::Mat,Q::Mat,U::Mat;dn=20,angular_scale=1)
	# KH: There are several assumptions in this particular algorithm
	# (1) p_max ~ 0.26 (Planck 2018 XII)
	# (2) N=7, which is the average number of layers along the line of sight as fitted by Planck 2016, 2018 
	# (3) I replaced f(δ) in Planck 2018 XII to Ma without a proportionality constant
	# From these three assumptions, we have
	# X = 2/sqrt(21) M_A(δ)/δ|∇Φ|  
	# (4) Chen et.al (2019): sin(γ)^2 = 5X/(4+X)
    delphi = Gaensler_SPG(I,Q,U)
    phi=0.5.*atan.(U,Q);
    nx,ny=size(delphi)
    nnx=div(nx,dn)
    nny=div(ny,dn)
    delphi_dn=zeros(nx,ny);
    Ma_dn=zeros(nx,ny)
    for i in 1:nx, j in 1:ny
    	delphix=circshift(delphi,[i-1+div(dn,2),j-1+div(dn,2)])
    	phix=circshift(phi,[i-1+div(dn,2),j-1+div(dn,2)])
    	delphi_dn[i,j]=mean(delphix[1:dn,1:dn])
    	Ma_dn[i,j]=cstd(phix[1:dn,1:dn])
    end
    X = (2/(sqrt(21))/dn).*Ma_dn./delphi_dn
    X2= (5.0.*X)./(4.0.+X)
    X2[X2.<0].=0
    Y = sqrt.(X2);
    gamma =zeros(size(Y))
    for i in 1:nx, j in 1:ny
    	if (abs(Y[i,j])<=1)
    		gamma[i,j]=asin(Y[i,j])
    	else
    		gamma[i,j]=NaN
    	end
    end
    #gamma=asin.(Y);
    return gamma
end






end # LazMultiPole