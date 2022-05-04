module LazThermal_Stochastic
using PyCall,FITSIO,FFTW
using LazRHT,LazThermal,LazCore,LazType
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
  KH: Stochastic Thermal Noise Adding module
  Mar 22 2019
==#
export adiabatic_cs,ensemble_creation
export stochastic_ppv

export continuum_stochastic_ppv



function adiabatic_cs(Energy::Number,Density::Number,mean_molecular_weight::Number,gamma::Number)
	# U = C nkT = rho kT/mu(gamma-1)
	# cs^2 = kT
	return sqrt.(Energy/Density*mean_molecular_weight*(gamma-1))
end

function adiabatic_cs(Energy::Cube,Density::Cube,mean_molecular_weight::Number,gamma::Number)
	# U = C nkT = rho kT/mu(gamma-1)
	# cs^2 = kT
	return sqrt.(Energy./Density.*mean_molecular_weight.*(gamma.-1))
end

function ensemble_creation(iv::Number,cs::Number,n_sample::Number)
	# KH creates a 1D array that obeys the normal distribution
	# Stochastic variable X with mean(X) = 0, std(X) = 1
	# in this scenario we want std(c_T) = cs
	v_array = randn(n_sample);
	v_new = iv .+ cs.*v_array;
	return v_new
end


function stochastic_ppv(dens::Cube,iv::Cube,gas_energy::Cube,mean_molecular_weight::Number,
						gamma::Number,n_sample::Number,channel_number::Number)
	# Idea: Create an ensemble for each position 
	# Use the thermal functions
	#==
	function ppv(d::Cube,v::Cube,binnum)
	nx,ny,nz=size(d);
	offset=1e-9;
	binrange=linspace(minimum(v),maximum(v),binnum+1);
	bindiff=maximum(diff(binrange));
	minv=minimum(v);
	p=zeros(nx,ny,binnum);
	for k in 1:nz,j in 1:ny,i in 1:nx
	vb=round(Int,div(v[i,j,k]-minv,bindiff))+1;
	if (vb>binnum) 
	vb=binnum;
	end
	p[j,k,vb]+=d[i,j,k];
	end
	return p
	end
	==#
	nx,ny,nz=size(iv);
	offset=1e-9;
	vmin=minimum(iv);
	vmax=maximum(iv);
	vrange=lspace(vmin,vmax,channel_number+1);
	vdiff=maximum(diff(vrange))
	p=zeros(ny,nz,channel_number);
	for k in 1:nz, j in 1:ny, i in 1:nx
		cs=adiabatic_cs(gas_energy[i,j,k],dens[i,j,k],mean_molecular_weight,gamma);
		v_array=ensemble_creation(iv[i,j,k],cs,n_sample)
		for cv in 1:n_sample
			vb=round(Int,div(v_array[cv]-vmin,vdiff))+1;
			if ((vb>0) & (vb<channel_number+1))
				# partial PPV within range to be out only.
				p[j,k,vb]+=dens[i,j,k]./n_sample
			end
		end
	end
	return p
end

function continuum_stochastic_ppv(dens::Cube,iv::Cube,gas_energy::Cube,mean_molecular_weight::Number,
      gamma::Number,n_sample::Number,channel_number::Number)
       # Idea: Create an ensemble for each position 
 # Use the thermal functions
 #==
 function ppv(d::Cube,v::Cube,binnum)
 nx,ny,nz=size(d);
 offset=1e-9;
 binrange=linspace(minimum(v),maximum(v),binnum+1);
 bindiff=maximum(diff(binrange));
 minv=minimum(v);
 p=zeros(nx,ny,binnum);
 for k in 1:nz,j in 1:ny,i in 1:nx
 vb=round(Int,div(v[i,j,k]-minv,bindiff))+1;
 if (vb>binnum) 
 vb=binnum;
 end
 p[j,k,vb]+=d[i,j,k];
 end
 return p
 end
 ==#
 nx,ny,nz=size(iv);
 offset=1e-9;
 vmin=minimum(iv);
 vmax=maximum(iv);
 vrange=linspace(vmin,vmax,channel_number+1);
 vdiff=maximum(diff(vrange));
 vchannel = collect(vmin:vdiff:vmax);
 p=zeros(ny,nz,channel_number);
 for k in 1:nz, j in 1:ny, i in 1:nx
  cs=adiabatic_cs(gas_energy[i,j,k],dens[i,j,k],mean_molecular_weight,gamma);
     ppvd_P = dens[i,j,k].*Prob_Function(iv[i,j,k],cs,vchannel).*vdiff;
  for vh = 1:channel_number
   p[j,k,vh] += ppvd_P[vh];
  end
 end
 return p
end

end # module LazThermal_Stochastic