module LazThermal_Kritsuk
using PyCall,FITSIO,FFTW,Statistics
using FastGaussQuadrature
using LazType,LazPyWrapper

	##############################################################################
	#
	# Copyright (c) 2019
	# Ka Ho Yuen, Ka Wai Ho and Alex Lazarian
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
	 KH: This is v1.0+
	==#

#==
  KH: Thermal broadening module specifically for Alexei Kritsuk's cube
  May 13 2019
==#

#==
  Sound speed = sqrt(dP/d\rho)
  In cold and warm regime the adiabatic EoS is valid, i.e. c_s^2 = (gamma)P/\rho
  Here P = \rho kT/\mu mH
  However 
==#

export cs,intensity_rescaling,intensity_rescaling_GL
export thermal_ppv_thin,thermal_ppv_cube


function cs(Pressure::Cube,Density::Cube,gamma::Number)
	# U = C nkT = rho kT/mu(gamma-1)
	# cs^2 = kT
	return sqrt.(Pressure./Density.*gamma)
end


function intensity_rescaling(Density::Cube,V_I::Number,V_T::Cube,c_s::Cube)
	return Density.*exp.(.-(V_I.-V_T).^2.0./c_s.^2.0./2.0)
end

function intensity_rescaling_GL(Density::Cube,V_I::Vec,Weights::Vec,V_T::Cube,c_s::Cube)
	ll=length(V_I)
	a=zeros(size(Density))
	for i in 1:ll
		# A normalization factor is required when temperature is not a constant.
		a+=Weights[i].*Density.*exp.(.-(V_I[i].-V_T).^2.0./c_s.^2.0./2.0)./c_s./sqrt.(2.0.*pi)
	end
	return a
end

function thermal_ppv_thin(d::Cube,p::Cube,v::Cube;v_0=0,dv=0.1*std(v),gamma=5/3,dx=1,node_number=10,const_dens=false)
	#==
	Input 
		d 	  : density
		p 	  : pressure
		v 	  : velocity
	Parameter
		v_0   : Position of the channel
		dv    : channel width
		gamma : EoS index
	Assumptions here
	1. dv <<1 -> Approximate the integral by simply the mid-pt theorem.
	==#
	nodes,weights=gausslegendre(node_number);
	

	vi=v_0.+dv./2.0.*nodes;
	if (const_dens)
		px=intensity_rescaling_GL(ones(size(d)),vi,weights,v,cs(p,d,gamma));
	else
		px=intensity_rescaling_GL(d,vi,weights,v,cs(p,d,gamma));
	end
	pp=reshape(sum(px,dims=3),size(d)[1:2]).*dx.*dv./2.0;
	return pp
end

function thermal_ppv_cube(d::Cube,p::Cube,v::Cube,vmin::Number,vmax::Number;
		dv=0.1*std(v),gamma=5/3,dx=1,node_number=10,const_dens=false)
	vmat=Array(vmin:dv:vmax);
	vlen=length(vmat);
	nx,ny,nz=size(d);
	pp=zeros(nx,ny,vlen);
	for nv in 1:vlen
		vv=vmat[nv];
		pp[:,:,nv]=thermal_ppv_thin(d,p,v,v_0=vv,dv=dv,gamma=gamma,dx=dx,node_number=node_number,const_dens=const_dens)
	end
	return pp
end

function EB_mode(Q::Mat,U::Mat)
	#==
	From Kritsuk+2018
	E(k) = (k_1^2-k_2^2)/k^2 Q(k) + (2k_1k_2)/k^2 U(k)
	B(k) = -(2k_1k_2)/k^2 Q(k) + (k_1^2-k_2^2)/k^2 U(k)
	==#

	Qf=fftshift(fft(Q))
	Uf=fftshift(fft(U));
	nx,ny=size(Q);
	X=zeros(nx,ny);
	Y=zeros(nx,ny);
	for i in 1:nx, j in 1:ny
		ii=i-div(nx,2)
		jj=j-div(ny,2)
		rr2=ii^2+jj^2
		if (rr2>0)
			X[i,j]=(ii^2-jj^2)/rr2
			Y[i,j]=2*ii*jj/rr2
		end
	end
	# Orthogonal product.
	Ef=  X.*Qf.+Y.*Uf;
	Bf=.-Y.*Qf.+X.*Uf;
	E=real(ifft(ifftshift(Ef)));
	B=real(ifft(ifftshift(Bf)));
	return E,B
end

function PlanckBD(lambda::Number,T::Cube)
	h = 6.626e-34;
	c = 3e8;
	kb= 1.38e-23;
	B_L=2.0.*h.*c.*c./lambda^5.0./(exp.(h.*c./kb./lambda./T).-1);
	return B_L
end




end # module LazThermal_Kritsuk