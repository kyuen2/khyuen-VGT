module ath2h5
using HDF5
#using PyCall
#glob = pyimport("glob");
const Cube{T}=Array{T,3};
const Mat{T}=Array{T,2};
const Vec{T}=Array{T,1};

##############################################################################
#
# Copyright (c) 2018 
# Ka Ho Yuen, Ka Wai Ho, Yue Hu, Junda Chen and Alex Lazarian
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

###############################################################################
#									      #
# 				   Code      			  	      # 
#				   for                                        #
#		     Converting Athena athdf to hdf5			      #
#									      #
###############################################################################
#   
#
#		Version     : v1.0.1(16/11/2019)
#       Author		: KW HO @ Lazarian Technology
#	Description : Converting Athena athdf to hdf5
#       Input       : *.athdf
#       Output      : function*(h5): ConvertAthdf ath2h5
#       Data Format : "ConvertAthdf"  : Converting Athdf inisde the julia  environment without save.
#                      Type: dictationary
#		       "ath2h5"     : Converting Athdf to h5 and saving
#                      Type: N/A
#
######################################################################################
export ConvertAthdf,ath2h5save,DecompressionMode


function puzzleX(prim,LL,k,Nx,Ny,Nz)
	c=map(Float32,zeros((Nx,Ny,Nz)));
	#Nx,Ny,Nz = [maximum(LL[1,:]),maximum(LL[2,:]),maximum(LL[3,:])].*???
	dN = [size(prim)[1];size(prim)[2];size(prim)[3]];
	for i = 1:size(LL)[2]
		x1s, x2s, x3s = (LL[:,i].*dN).+1;
		x1f, x2f, x3f = (LL[:,i].+1).*dN;
		c[x1s:x1f, x2s:x2f, x3s:x3f] = prim[:,:,:,i,k];
	end
	c
end

struct DecompressionMode
	EOS
	B
	G
end

function ConvertAthdf(db,Nx::Int,Ny::Int,Nz::Int,mode)
	f = h5open(db,"r");
	prim = read(f,"prim");
 	LL   = read(f,"LogicalLocations");
 	Cubes = Dict("d"=>zeros(Nx,Ny,Nz) ,"p"=>zeros(Nx,Ny,Nz) ,"G"=>zeros(Nx,Ny,Nz),
 		         "ib"=>zeros(Nx,Ny,Nz),"jb"=>zeros(Nx,Ny,Nz),"kb"=>zeros(Nx,Ny,Nz),
 		         "iv"=>zeros(Nx,Ny,Nz),"jv"=>zeros(Nx,Ny,Nz),"kv"=>zeros(Nx,Ny,Nz));
 	Cubes["d"] = puzzleX(prim,LL,1,Nx,Ny,Nz);
 	if (mode.EOS == "Isothermal" )
		Cubes["iv"] = puzzleX(prim,LL,2,Nx,Ny,Nz);
		Cubes["jv"] = puzzleX(prim,LL,3,Nx,Ny,Nz);
		Cubes["kv"] = puzzleX(prim,LL,4,Nx,Ny,Nz);
 	elseif (mode.EOS == "Adiabatic")
		Cubes["p"]  = puzzleX(prim,LL,2,Nx,Ny,Nz);
		Cubes["iv"] = puzzleX(prim,LL,3,Nx,Ny,Nz);
		Cubes["jv"] = puzzleX(prim,LL,4,Nx,Ny,Nz);
		Cubes["kv"] = puzzleX(prim,LL,5,Nx,Ny,Nz);
	end
	if ( mode.B == true )
 		B = read(f,"B");
		Cubes["ib"] = puzzleX(B,LL,1,Nx,Ny,Nz);
		Cubes["jb"] = puzzleX(B,LL,2,Nx,Ny,Nz);
		Cubes["kb"] = puzzleX(B,LL,3,Nx,Ny,Nz);
	end
	if (mode.G == true)
		Cubes["G"] = puzzleX(prim,LL,size(prim)[end],Nx,Ny,Nz);
	end
	close(f)
	return Cubes
end


function ath2h5save(file,Nx::Int,Ny::Int,Nz::Int,mode)
	Case = file[1:10]
	Time= file[end-10:end-6]
	Name = Case*"_T"*Time*".h5";
	fw = h5open(Name,"w");
	Cubes = ConvertAthdf(file,Nx,Ny,Nz,mode)
	GC.gc();
	write(fw,"gas_density",Cubes["d"]);
	write(fw,"i_velocity",Cubes["iv"]);
	write(fw,"j_velocity",Cubes["jv"]);
	write(fw,"k_velocity",Cubes["kv"]);
	if (mode.EOS == "Adiabatic")
		write(fw,"gas_pressure",Cubes["p"]);
	end
	if (mode.G == true)
		write(fw,"grav_pot",Cubes["G"]);
	end
	if ( mode.B == true )
		write(fw,"i_mag_field",Cubes["ib"]);
		write(fw,"j_mag_field",Cubes["jb"]);
		write(fw,"k_mag_field",Cubes["kb"]);
	end
	close(fw);
end
end

#Exmple to Use the code
#1. EOS  = (Adaibatic->γ ≠ 1, Isothermal = 1 )
#2. B-field = true/false
#3. Gravity = true/false
#mode = DecompressionMode("Isothermal",true,true)
#Nx=Ny=Nz=120;
#db = "Turb.out2.00000.athdf";
#Cubes = ConvertAthdf(db,Nx,Ny,Nz,mode);
