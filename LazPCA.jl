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

"""
	module LazPCA

PCA Method for VGT
Include:


Reference: Brunt & Heyer 2002b, Heyer et al.2008, Hu et. al 2018a
Author: Ka Ho Yuen, Yue Hu, Dora Ho, Junda Chen

Changelog:
	- Mike Initiate module LazPCA
	- Ka Ho review the code and compare with PCA_v10.jl

"""
module LazPCA

# using PyCall
using LsqFit
using FFTW # fft, ifft
using Statistics # var, std
using LinearAlgebra
using LazType,LazThermal


export compute_PCA,ACF,linspace_float,isotropy_index,compute_PCA_2002

# Below are debug line
export pvslice,covar_pvslice,line,find_char_scale,Gaus,find_char_scale_with_gaussian
export find_char_scale_matrix,covar_matrixdot,twopix_approx,covar_matrixcorr,covar_pvcorr
export eig,PCA_channel_dot_product

e=exp(1);

# """@internal"""
function linspace(a,b,c)
	# KH : Construct a 1d linspace 
	#      TODO: linspce is called LinRange in Julia v1.0
	width=mod(b-a,c);
	return [x for i in a:width:b]
end


function twopix_approx(x1,x2,y1,y2,y)
	# y-y1 = (y2-y1)/(x2-x1)*(x-x1)
	x=(y-y1)/(y2-y1)*(x2-x1)+x1;
	return x
end

function project_x( A ::Cube )
	return sum(A, dims=1)[1, :, :]
end

function project_y( A ::Cube )
	return sum(A, dims=2)[:, 1, :]
end


function pvslice(p::Cube,thickness::Number)
	# KH: In my cube the thickness is 1/nx
	# nx,ny,nv=size(p);
	if (thickness<=0)
		thickness=1/nx;
	end
	Wx = project_x( p ) / thickness
	Wy = project_y( p ) / thickness

	# Wx = reshape(sum(p,dims=2), nx, nv) / thickness;
	# Wy = reshape(sum(p,dims=1), ny, nv) / thickness;
	return Wx,Wy
end

function covar_pvslice(W::Mat)
	nx,nv=size(W);
	C=zeros(nv,nv);
	for j in 1:nv, i in 1:nv
		C[i,j]=sum(W[:,i].*W[:,j])/nx;
	end
	return C
end

function covar_pvcorr(W::Mat)
	nx,nv=size(W);
	C=zeros(nv,nv);
	for j in 1:nv, i in 1:nv
		C[i,j]=sum(W[:,i].*W[:,j])./nx.-sum(W[:,i])*sum(W[:,j])/(nx.^2);
	end
	return C
end

function covar_matrixdot(p::Cube)
	nx,ny,nv=size(p);
	C=zeros(nv,nv);
	for j in 1:nv, i in 1:nv
		C[i,j]=sum((p[:,:,i].*p[:,:,j])[:])/nx/ny;
	end
	return C
end

function covar_matrixcorr(p::Cube)
	nx,ny,nv=size(p);
	C=zeros(nv,nv);
	for j in 1:nv, i in 1:nv
		C[i,j]=sum((p[:,:,i].*p[:,:,j])[:])/nx/ny-sum(p[:,:,i])*sum(p[:,:,j])/(nx*ny)^2
	end
	return C
end

function ACF(A::Mat)
	# This is the normalized autocorrelation function
	# notice: CF R(x) = <A(r)A(r+x)> = real(ifft(abs(fft(A)).^2))[end:-1:1,end:-1:1];
	# ACF=CF/var(A)
	return real(ifft(abs.(fft(A)).^2))[end:-1:1,end:-1:1]/var(A);
end

function ACF(A::Vec)
	# This is the normalized autocorrelation function
	# notice: CF R(x) = <A(r)A(r+x)> = real(ifft(abs(fft(A)).^2))[end:-1:1,end:-1:1];
	# ACF=CF/var(A)
	return real(ifft(abs.(fft(A)).^2))/var(A);
end

function line(x,p)
	return p[1].+p[2].*x
end

function Gaus(x,p)
	return p[1].*exp.(.-(x.-p[2]).^2.0.*p[3]);
end

function find_char_scale(A::Vec,Scale::Vec)
	Ax=ACF(A);
	Ax=Ax.-mean(Ax);
	Ax=Ax./Ax[1];
	# Abusing the findmix function
	# Definition of delta v and L
	# Ax(delta v) = 1/e
	pos=0;
	for i in 1:length(Scale)
	 if (Ax[i]>1/e)
	  pos=twopix_approx(Scale[i],Scale[i+1],Ax[i],Ax[i+1],1/e)
	 else
	  break;
	 end
	end
	#if (pos < div(nx,2))
	return pos
	#else
		#return Scale[nx+1-pos];
	#end
end

function find_char_scale_matrix(A::Mat,Scale::Vec)
	Ax=ACF(A);
	Ax=Ax.-mean(Ax);
	Ax=Ax./Ax[1,1];
	# Abusing the findmix function
	# Definition of delta v and L
	# Ax(delta v) = 1/e
	nx,ny=size(Ax);
	posx=0;posy=0;
	for i in 1:nx
	 if (Ax[i,1]>1/e)
	  posx=i;
	 else
	  break;
	 end
	end
	for i in 1:ny
	 if (Ax[1,i]>1/e)
	  posy=i;
	 else
	  break;
	 end
	end
	#if (pos < div(nx,2))
	return Scale[posx],Scale[posy];
	#else
		#return Scale[nx+1-pos];
	#end
end

function find_char_scale_with_gaussian(A::Vec,Scale::Vec)
	Ax=fftshift(ACF(A));
	nx=size(Ax)[1];
	fit=curve_fit(Gaus,Scale,Ax,[rand(3)...]);
	return 1/fit.param[3]
end

function linspace_float(a,b,number)
	A=linspace(a,b,number)
	nx=size(A)[1];
	Ax=zeros(nx);
	for i in 1:nx
		Ax[i]=Float64(A[i]);
	end
	return Ax
end

function eig(A::Mat)
	eigv=eigvecs(A);
	eiglambda=eigvals(A);
	return eiglambda,eigv
end

function compute_PCA(p::Cube,pscale::Vec,sample_number::Number)
	nx,ny,nv=size(p);
	xscale=lspace(0,10,nx);
	Wx,Wy=pvslice(p,1/nx);
	Cx=covar_pvcorr(Wx);
	Cy=covar_pvcorr(Wy);
	Cx_eig_value,Cx_eig_vectors=eig(Cx);
	Cy_eig_value,Cy_eig_vectors=eig(Cy);
	# Typically C?_eig_value only has ~10 dominant part
	Ix=Wx*Cx_eig_vectors;
	Iy=Wy*Cy_eig_vectors;
	# Abusing the sort and sortperm
	# 1st indice for eig_vector (2nd indice for IxIy) is the sort-perm indice
	# From big eigenvalue to small
	px=sortperm(Cx_eig_value,rev=true)[1:sample_number];
	py=sortperm(Cy_eig_value,rev=true)[1:sample_number];
	# With px,py we now know the biggest contributions, denoted by px[1...] and py[1...]
	delta_vx=zeros(sample_number);
	delta_vy=zeros(sample_number);
	tau_x=zeros(sample_number);
	tau_y=zeros(sample_number);
	pscalex=pscale[pscale.>0];
	for i in 1:sample_number
		delta_vx[i]=abs(find_char_scale(reshape(Cx_eig_vectors[:,px[i]],nv),pscalex));
		delta_vy[i]=abs(find_char_scale(reshape(Cy_eig_vectors[:,py[i]],nv),pscalex));
		tau_x[i]=find_char_scale(Ix[:,px[i]],xscale);
		tau_y[i]=find_char_scale(Iy[:,py[i]],xscale);
	end
	fitx=curve_fit(line,log.(tau_x),log.(delta_vx),[rand(2)...]);
	fity=curve_fit(line,log.(tau_y),log.(delta_vy),[rand(2)...]);
	ax=fitx.param[2];
	ay=fity.param[2];
	return ax,ay,tau_x,tau_y,delta_vx,delta_vy
end

function PCA_channel_dot_product(p::Cube)
	## KH: Simply produce the first few channels of PCA-channels
	nx,ny,nv=size(p);
	xscale=lspace(0,10,nx);
	C=covar_matrixcorr(p);
	# TODO: [#eigs] Add support to older version
	Cv,Ce=eig(C);
	# Typically C?_eig_value only has ~10 dominant part
	pp=zeros(nx,ny,nv);
	for i in 1:nv,j in 1:nv
		pp[:,:,i]+=p[:,:,j]*Ce[j,i]
	end
	# The 
	return pp,Cv
end

# TODO: Add support for old version
#==
function eig(A::Mat)
	return eigvals(A), eigvecs(A)
end
==#
function compute_PCA_2002(p::Cube,pscale::Vec,sample_number::Number)
	nx,ny,nv=size(p);
	xscale=linspace_float(0,10,nx);
	C=covar_matrixcorr(p);
	# TODO: [#eigs] Add support to older version
	Cv,Ce=eig(C);
	pp=sortperm(Cv,rev=true)[1:sample_number];
	dv=zeros(sample_number);
	dx=zeros(sample_number);
	dy=zeros(sample_number);
	pscalex=pscale[pscale.>0]
	for i in 1:sample_number
		I=zeros(nx,ny);
		for kk in 1:nv, jj in 1:ny, ii in 1:nx
			I[ii,jj]+=p[ii,jj,kk].*Ce[kk,pp[i]];
		end
		dv[i]=abs(find_char_scale(reshape(Ce[:,pp[i]],nv),pscalex));
		dx[i],dy[i]=find_char_scale_matrix(I,xscale);
	end
	dl=sqrt(dx.^2+dy.^2);
	fit=curve_fit(line,log(dl),log(dv),[rand(2)...]);
	a=fit.param[2];
	return a,dl,dv
end

function isotropy_index(ax,ay)
	if (ax*ay>0)
		return 1-abs(ax-ay)/sqrt(ax*ay)
	end
	return NaN

end

function isotropy_average(ax,ay)
	return (ax-ay)/(ax+ay)
end

end # End Module LazPCA
