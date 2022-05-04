module LazCore_v2
using PyCall,FITSIO,FFTW,Statistics,StatsBase
using FastGaussQuadrature
using LazType,LazPyWrapper

	##############################################################################
	#
	# Copyright (c) 2019
	# Ka Ho Yuen and Alex Lazarian
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
	Acknowledgments to Junda (Mike) Chen in discussing the possibiltiy of Bayesian Gradients
	==#

export gradient_histogram


function gradient_histogram(d::Mat;cutoff=10,dx=1,weighting_index=2)
	nx,ny=size(d);
	dd=zeros(nx,ny);
	weight=ones(nx,ny)
	for i in 1:nx, j in 1:ny
		idx=i-div(nx,2);
		jdx=j-div(ny,2);
		r=sqrt(idx^2+jdx^2);
		if (r>0)
			weight[i,j]=exp(weighting_index*log(r))
		end
	end
	ang=zeros(nx,ny);
	for i in 1:nx, j in 1:ny
		dcenter=d[i,j];
		dl=fftshift(circshift(d.-dcenter,[i-1,j-1]));
		dl=dl./weights;
		ang[i,j]=atan(j./i)

	end	
	aa=fit(Histogram,log.(T)[:]./log(10),0:0.1:4)
end


# KH: Nov 19 2019
#==
 The core filtering algorithm in according to Li+15
 The correct calculation of energies is required to proceed
==#

"""
TODO:
  1. Migrating the 2014-era code from me to here
  2. Check the units and see if results agrees with the MHD output
  3. Implement the maximum-FoF algorithm as follows
  	(a) grab all neighboring pixels (diagonals included)
  	(b) see if that increases the virial number
  	(c) yes -> add in else throw away.
"""

function ke(d::Cube,iv::Cube,jv::Cube,kv::Cube;dx=1)
	dV=dx^3;
	return 0.5.*d.*(iv.^2.0.+jv.^2.0.+kv.^2.0);
end

function U(p::Cube;dx=1)
	dV=dx^3;
	return p*dV
end

function ME(ib::Cube,jb::Cube,kb::Cube;dx=1)
	# ZEUS/ATHENA has sqrt(4pi included)
	dV=dx^3;
	return 0.5.*(ib.^2.0.+jb.^2.0.+ik.^2.0).*dV
end

function GE(d::Cube,gp::Cube;G=1,dx=1)
	# Take 2nd order approximation
	gpx=(circshift(gp,[1,0,0])-circshift(gp,[-1,0,0]))./(2*dx);
	gpy=(circshift(gp,[0,1,0])-circshift(gp,[0,-1,0]))./(2*dx);
	gpz=(circshift(gp,[0,0,1])-circshift(gp,[0,0,-0]))./(2*dx);
	gp2=gpx.^2.0.+gpy.^2.0.+gpz.^2.0;
	dX=dx^4;
	return gp2./(8*pi*G).*dX
end

function pixelwise_energy_computation(d::Cube,iv::Cube,jv::Cube,kv::Cube,ib::Cube,jb::Cube,kb::Cube,p::Cube;gp=zeros(size(d)),G=1,dx=1);
	kevalue=ke(d,iv,jv,kv,dx=dx);
	ievalue=U(p,dx=dx);
	mevalue=ME(ib,jb,kb,dx=dx);
	gevalue=GE(d,gp,G=G,dx=dx);
	return kevalue.+ievalue.+mevalue.-gevalue
end

function coor2id(seedx::Number,seedy::Number,seedz::Number,nx::Number,ny::Number,nz::Number)
	return seedx+(seedy-1)*nx+(seedz-1)*nx*ny;
end

function id2coor(id::Number,nx::Number,ny::Number,nz::Number)
	seedz=div(id,nx*ny)+1;
	seedy=div(id-(seedz-1)*nx*ny,nx)+1;
	seedx=id-(seedy-1)*nx-(seedz-1)*nx*ny;
	return seedx,seedy,seedz
end

function maximum_fof_algorithm(E::Cube,seedx::Number,seedy::Number,seedz::Number;periodic=false)
	id_bucket_tbc=zeros(0);
	id_bucket_checked=zeros(0);
	nx,ny,nz=size(E);
	seed=seed2id(seedx,seedy,seedz,nx,ny,nz);
	push!(id_bucket_tbc,seed)

	if (periodic)
		while(length(id_bucket_tbc)>0)
			# Get the element out
			a=pop!(id_bucket_tbc);
			push!(id_bucket_checked,a)
			x,y,z=id2coor(a);
			for i in -1:1,j in -1:1,k in -1:1
				if ~((i=0) && (j==0) && (k==0))
					xx=mod(x+i-1,nx)+1;
					yy=mod(y+j-1,ny)+1;
					zz=mod(z+k-1,nz)+1;
					scoord=coor2id(xx,yy,zz,nx,ny,nz);
					if ((length(findall(id_bucket_checked.==scoord))==0) && (length(findall(id_bucket_tbc.==scoord))==0)
						if (E[scoord]<0)
							push!(id_bucket_tbc,scoord)
						end
					end
				end
			end
		end
	else
		error("KH: not implemented yet.")			
	end
	return id_bucket_checked
end




function mfof(E::Cube,seedx::Number,seedy::Number,seedz::Number;periodic=false)
	# Declare an "Any" type to store the matrix
	 dict_id = Dict{Int,Array{Float64,1}}()
	 nx,ny,nz=size(E);
	 maxcoord=findall(E.==minimum(E))[1]
	 maxcoordid=coord2id(maxcoord[1],maxcoord[2],maxcoord[3],nx,ny,nz);
	 Ec=zeros(size(E));
	 Ec[maxcoord]=1;

	 """
	 Intended use

	 Find a list of "maximum points" that 
	 id_bucket=maximum_fof_algorithm(E,seedx,seedy,seedz)
	 dict_id[Some_Integer]=id_bucke
	 """


end

end #module LazCore_V2