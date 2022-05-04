module LazDDA
using HDF5,PyPlot,Statistics,LsqFit,PyCall,FFTW,StatsBase,Images
#using LazCore,LazType,LazThermal,LazIO,LazThermal_Kritsuk
#using LazRHT_investigation,LazCFA


	##############################################################################
	#
	# Copyright (c) 2020
	# Ka Ho Yuen, Ka Wai Ho, Alex Lazarian and Dmitri Pogosyan
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

	# Nickname "Velocity Decomposition Algorithm", referring to the algorithm I developed in April 2020
	# when trying to decompose the velocity channel map into densities and velocities contribution.

#	export Vec 	# Abstract 1D Array
#	export Mat 	# Abstract 2D Array
#	export Cube	# Abstract 3D Array

	const  Vec{T}  = Array{T,1};
	const  Mat{T}  = Array{T,2};
	const Cube{T}  = Array{T,3};
	const Vecx{T}  = LinRange{T};

	export pdec,DDA,DDA_dispersion,vbin

	function re(p::Cube;dims=3,s=size(p)[1:2])
		I=sum(p,dims=dims);
		I=reshape(I,s);
		return I
	end

	function channelproduct(A::Mat,B::Mat)
	    return mean((A.-mean(A)).*(B.-mean(B)))
	end

	function vbin(v)
		vv=zeros(2*length(v)-1);
		vv[1:2:end]=v;
		vv[2:2:end]=0.5.*(v[1:end-1].+v[2:end]);
		return vv
	end


	function ncp(A::Mat,B::Mat)
	    return channelproduct(A,B)./std(A)./std(B)
	end

	function normalized_channel(A::Mat)
	    return (A.-mean(A))./std(A)
	end

	function pd(p::Mat,I::Mat)
	    Ix=normalized_channel(I);
	    return channelproduct(p,Ix).*Ix
	end

	function pv(p::Mat,I::Mat)
	    return p.-pd(p,I)
	end

	function pdec(p::Mat,I::Mat)
	    return pd(p,I),pv(p,I)
	end

	function DDA(p::Cube)
		nx,ny,nv=size(p);
		pdd=zeros(size(p));
		pvv=zeros(size(p));
		I=re(p)
		for k in 1:nv
			pdd[:,:,k]=pd(p[:,:,k],I)
			pvv[:,:,k]=pv(p[:,:,k],I)
		end
		return pdd,pvv
	end

	function DDA_dispersion(p::Cube,vv)
		pd,pv=DDA(p)
		aaaa=zeros(0)
		bbbb=zeros(0)
		cccc=zeros(0)
		for i in 1:length(vv)
		    push!(aaaa,vv[i])
		    push!(bbbb,std(pd[:,:,i]))
		    push!(cccc,std(pv[:,:,i]))
		end
		return aaaa,bbbb,cccc
	end

end # LazDDA
