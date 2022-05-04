# Copyright (C) <2018> 
# <Ka Ho Yuen, Ka Wai Ho, Yue Hu, Junda Chen and Alex Lazarian>

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

"""
	module LazCore

Embeded core functionalities for calculation.
Include:
	Image Processing Tools: Gaussian Filter, Gaussian Fitting, sband, sobel, convolution
	Math Tools: structure function, coorelation function
	Advance Cube Operation: Cube rotation, PPV Generator

Author: Ka Ho Yuen, Yue Hu, Dora Ho, Junda Chen

Changelog:
	- Mike Initiate module LazCore
	- Ka Ho review the code and add some native functions

Todo:

"""
module LazCore

# using PyCall
using LsqFit
using StatsBase
using LinearAlgebra
using Statistics

# TODO: [linspace] Might change this for linspace
# linspace = LinRange

# There are some modules requried upon the update of v1.0
using FFTW
# using ImageFiltering


using LazType
using LazIO

# @pyimport numpy as np

############################
# Export Symbols
############################

# Math Module
	export dotproductangle, dot_product_3d,atan2
	# """internal hist"""
	# """internal maxid, rmaxid"""
	# """internal atan2, linspace, meshgrid"""

# Image Processing
	# Gaussian Fitting
	export fit_gaussian_2d
	
	export sban2d
	
	# sobel family
	export sobel_conv_2d, sobel_conv_3d
	export sobel_kernel_2d, sobel_kernel_3d
	
	# sobel helper methods
	# """internal sobel_parallel, sobel_perpendicular"""
	export convoluting_kernel

# Cube Operation
	export ppv

# Alignment measure
	export AM

############################
# Implementation
############################

##
# Math Module
	# export dotproductangle, bitwise_filter, dot_product_3d, sban_dotproduct

function dotproductangle(a::Mat,b::Mat)
	a1=cos.(a);  a2=sin.(a);
	b1=cos.(b);  b2=sin.(b);

	# Notice: matnorm() : from package `LinearAlgebra`
	# (matnorm(a1, a2) .* matnorm(b1, b2)) is usually be one
	# but Julia sometimes will create stupid values that will
	# return NaN while computing the dot product.

	cab = (a1.*b1 + a2.*b2) ./ (matnorm(a1, a2) .* matnorm(b1, b2));

	# Warn: Brutal calculation
	# Putting all values taht are stupidly out of range to be normal
	cab[cab.> 1] .=  1;
	cab[cab.<-1] .= -1;
	ab  = acos.(cab);
	return ab
end

function bitwise_filter(A::Cube,threshold)
	# KH : The bitwise filter for Smith cloud (Hu et.al 2018c)
	A =  A.-threshold;
	B = (A.+abs.(A))./(2.0 .* abs.(A));
	B[isnan.(B)].=0;
	return B;
end

function dot_product_3d(A::Cube,B::Cube)
	# KH : The bitwise dot product
	Am = mean(A);
	Bm = mean(B);
	ratio = Am/Bm;
	Ax = bitwise_filter(A,Am);
	Bx = bitwise_filter(B,Bm);
	len= length(Am);
	return (sum(Ax.*Bx) / len);
end



# """@internal"""
if (VERSION > v"0.6.0")
	function hist_new(data,range)
		# Linear histogram function
		# Wrapper for the `hist` function for functions written for julia v0.5-
		h=fit(Histogram,data,range)
		ax=h.edges[1];
		ac=h.weights;
		return ax,ac
	end
end

# """@internal"""
function maxid(ax::Vec)
 return findall(ax.==maximum(ax));
end

# """@internal"""
function maxid(ax::Mat)
 return findall(ax.==maximum(ax));
end

# """@internal"""
function meshgrid(X,Y)
	# KH : native implementation of meshgrid
	#      The order of the meshgrid Zis **different from** python
	#      X,Y are ranges.
	return [ i for i=X, j=Y ], [ j for i=X, j=Y ]
end

# """@internal"""
#==
function atan2(X::Mat,Y::Mat)
	# Native matrix atan2 operator
	return [atan(Y[i,j],X[i,j]) for i=1:size(X)[1], j in 1:size(X)[2]]
end
==#

function atan2(X::Mat,Y::Mat)
	# Native matrix atan2 operator
	return atan.(Y,X)
end

	# # Gaussian Fitting
		# export fit_gaussian_2d

function fit_gaussian_2d(Ax::Mat,Ay::Mat,binsize;type=1)
	# KH : The subblock Gaussian fitting algorithm
	#      this algorithm was ignoring the periodicity of the angle
	#      and use the Real space defined Gaussian to fit instead.
	#      a fftshift is required to play for the two cases.
	phi=atan.(Ay./Ax)
	Gaus(x,p)=p[1]*exp.(.-(x.-p[2]).^2 .*p[3]);
	# KH: This modiification is made because of the new julia-v1.1 error called
	# `ERROR: Data contains `Inf` or `NaN` values and a fit cannot be performed`
	# 
	if (type==0)
		ax,ac=hist_new(phi[.~isnan.(phi)][:],-pi/2:0.01:pi/2)
	else
		ax,ac=hist_new(phi[.~isnan.(phi)][:],linspace(-pi/2,pi/2,binsize+1));
	end
	ax=.5*(ax[1:end-1]+ax[2:end]);
	if (abs.(ax[maxid(ac)])[1]<pi/4)
		fit=curve_fit(Gaus,ax,ac/sum(ac),[maximum(ac/sum(ac)),0.0,1.0])
	else
		ax=ax.-pi./2;
		ac=fftshift(ac);
		fit=curve_fit(Gaus,ax,ac/sum(ac),[maximum(ac/sum(ac)),-pi/2,1.0]);
	end
	sigma=estimate_errors(fit,0.95);
	return fit.param[2],sigma[2];
end

	# # sban family
	# # the Sub-Block-Average-New family
	# # Created for Yuen & Lazarian (2017a) and subsequent papers
		# export sban2d


function sban2d(Ax::Mat,Ay::Mat,dn;type=1)
	nx,ny=size(Ax)
	Ana=zeros(div(nx,dn),div(ny,dn));
	Ans=zeros(div(nx,dn),div(ny,dn));

	for  j in 1:div(ny,dn),i in 1:div(nx,dn)
	   is=(i-1)*dn+1;
	   ie=i*dn;
	   js=(j-1)*dn+1;
	   je=j*dn;f=
	   Axx=Ax[is:ie,js:je];
	   Ayy=Ay[is:ie,js:je];
	   binsize=dn;
	   Apeak,Adisp=fit_gaussian_2d(Axx,Ayy,binsize;type=1);
	   Ana[i,j]=Apeak;
	   Ans[i,j]=Adisp;
	end
	return Ana,Ans
end


	# # sobel family
	# # KH : The Sobel derivative used in Soler et.al 2013
	# #      A local implementation is much quicker than calling from python
		# export sobel_conv_2d, sobel_conv_3d
		# export sobel_kernel_2d, sobel_kernel_3d

function sobel_conv_2d(A::Mat)
	Kx,Ky=sobel_kernel_2d(A);
	Ax=convoluting_kernel(A,Kx);
	Ay=convoluting_kernel(A,Ky);
	return Ax,Ay
end

function sobel_conv_3d(A::Cube)
	Kx,Ky,Kz=sobel_kernel_3d(A);
	Ax=convoluting_kernel(A,Kx);GC.gc()
	Ay=convoluting_kernel(A,Ky);GC.gc()
	Az=convoluting_kernel(A,Kz);GC.gc()
	return Ax,Ay,Az
end

function sobel_kernel_2d(A::Mat)
	nx,ny=size(A);
	Ax=zeros(size(A));
	Ay=zeros(size(A));
	vp=sobel_parallel(3);
	vl=sobel_perpendicular(3);
	Axx=zeros(3,3);
	Ayy=zeros(3,3);
	for j in 1:3, i in 1:3
		Axx[i,j]=vp[i]*vl[j];
		Ayy[i,j]=vl[i]*vp[j];
	end
	Ax[1:3,1:3]=circshift(Axx,(1,1));
	Ay[1:3,1:3]=circshift(Ayy,(1,1));
	Ax=circshift(Ax,(-1,-1));
	Ay=circshift(Ay,(-1,-1));
	return Ax,Ay
end

function sobel_kernel_3d(A::Cube)
	nx,ny,nz=size(A);
	Ax=zeros(size(A));
	Ay=zeros(size(A));
	Az=zeros(size(A));
	vp=sobel_parallel(3);
	vl=sobel_perpendicular(3);
	Axx=zeros(3,3,3);
	Ayy=zeros(3,3,3);
	Azz=zeros(3,3,3);
	for  k in 1:3,j in 1:3, i in 1:3
		Axx[i,j,k]=vp[i]*vl[j]*vl[k];
		Ayy[i,j,k]=vl[i]*vp[j]*vl[k];
		Azz[i,j,k]=vl[i]*vl[j]*vp[k];
	end
	Ax[1:3,1:3,1:3]=circshift(Axx,(1,1,1));
	Ay[1:3,1:3,1:3]=circshift(Ayy,(1,1,1));
	Az[1:3,1:3,1:3]=circshift(Azz,(1,1,1));
	Ax=circshift(Ax,(-1,-1,-1));
	Ay=circshift(Ay,(-1,-1,-1));
	Az=circshift(Az,(-1,-1,-1));
	return Ax,Ay,Az
end


	# # sobel helper
		# internal sobel_parallel, sobel_perpendicular
		# export convoluting_kernel


function sobel_parallel(size::Int)
	if (size>=3)
		v=zeros(size);
		v[2]=-1
		v[end]=1
		return v
	else
		error("LazCore.sobel_parallel: Size < 3 not supported")
	end
end

function sobel_perpendicular(size::Int)
	if (size>=3)
		v=zeros(size);
		v[2]=1
		v[1]=2
		v[end]=1
		return v
	else
		error("LazCore.sobel_perpendicular: Size < 3 not supported")
	end
end

function convoluting_kernel(A::Cube,B::Cube)
	Af=fft(A);
	Bf=fft(B);
	Cf=Af.*Bf
	C=real(ifft(Cf));
	return C
end

function convoluting_kernel(A::Mat,B::Mat)
	Af=fft(A);
	Bf=fft(B);
	Cf=Af.*Bf
	C=real(ifft(Cf));
	return C
end

# Cube Operation
	# export ppv

# """@internal"""
function linspace(a,b,c)
	# KH: a Lower limit, B Upper limit, C number of points
	# KH : Construct a 1d linspace
	width =(b-a)/c;
	x=zeros(c+1);
	xx=a:width:b
	for i in 1:c+1
	x[i]=xx[i]
	end
	return x
end


function ppv(d::Cube,v::Cube,binnum)
 nx,ny,nz=size(d);
 offset=1e-9;
 # KH : There is no linspace function further more
 #binrange=linspace(minimum(v),maximum(v),binnum+1);
 bindiff=(maximum(v)-minimum(v))/(binnum)
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

function AM(a::Mat,b::Mat)
    z=a.*b;
    ab=findall(.~isnan.(z))
    ca=cos.(a[ab][:]);
    sa=sin.(a[ab][:]);
    cb=cos.(b[ab][:]);
    sb=sin.(b[ab][:]);
    c=ca.*cb+sa.*sb;
    return mean(2 .*c.^2 .-1);
end

end # End module LazCore
	
