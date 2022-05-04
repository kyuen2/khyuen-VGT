module LazThermal
using PyCall,FITSIO,LazRHT,LazCore,LazType,FFTW,LsqFit
using Base.Threads
  gc=GC.gc; 
	##############################################################################
	#
	# Copyright (c) 2019
	# Ka Ho Yuen and Alex Lazarian
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
	 KH: This is v1.0+ only
	==#

#==
  KH: Extra functions to deal with Clark+19's issue
  Mar 21 2019
==#
export lspace,onedconvolution,ppv_thermal,ppv_deconvolution,wiener_deconvolution
export ppv_thermal_from_ppv
export cf,sf,psf,radial_average,spectrallines,spectralindex,line

# custom functions
line(x,p)=p[1].+p[2].*x;

function cf(A::Cube)
  return real(ifft(abs.(fft(A)).^2.0))./length(A);
end

function sf(A::Cube)
  asf=cf(A);
  return 2.0.*(asf[1,1,1].-asf);
end

function psf(A::Cube)
  asf=sf(A);
  apsf=reshape(sum(asf,dims=1),size(A)[2:3]);
  return apsf;
end

function cf(A::Mat)
  return real(ifft(abs.(fft(A)).^2.0))./length(A);
end

function sf(A::Mat)
  asf=cf(A);
  return 2.0.*(asf[1,1].-asf);
end

function radial_average(A::Mat)
 nx,ny=size(A);
 nl=round(Int,sqrt(nx^2+ny^2));
 kn=zeros(nl);
 kl=zeros(nl);
 kr=zeros(nl);
 for i in 1:nx, j in 1:ny
  idx=i-div(nx,2);
  jdx=j-div(ny,2);
  rr=round(Int,sqrt(idx^2+jdx^2));
  if ((rr<=nl) & (rr>0))
   kn[rr]+=A[i,j];
   kl[rr]+=1;
   kr[rr]=rr;
  end
 end
 return kr,kn./kl
end

function spectrallines(A::Mat)
 nx,ny=size(A);
 nl=round(Int,sqrt(nx^2+ny^2));
 kn=zeros(nl);
 kl=zeros(nl);
 Ax2=fftshift((abs.(fft(A))).^2.0);
 for i in 1:nx, j in 1:ny
  idx=i-div(nx,2);
  jdx=j-div(ny,2);
  rr=round(Int,sqrt(idx^2+jdx^2));
  if ((rr<=nl) & (rr>0))
   kn[rr]+=Ax2[i,j];
   kl[rr]=rr;
  end
 end
 return kn,kl
end

function spectralindex(A::Mat,xmin::Number,xmax::Number)
  xran2=xmin:xmax
  kvn,kvl=spectrallines(A)
  fitv=curve_fit(line,log10.(xran2),log10.(kvn[xran2]),[rand(2)...]);
  return fitv.param[2];
end

function lspace(a,b,c)
           # KH : Construct a 1d linspace
           width = (b-a)/c;
           x=zeros(round(Int,c));
           for i in 1:round(Int,c)
               x[i]=a+width*(i-1)
           end
           return x
       end

function onedconvolution(A::Vec,B::Vec)
  #==
  KH: The convolution formula is X[v] = \int v' A[v']B[v+v']
  We would not use the convolution theorem.
  ==#
  nx=size(A)[1];
  Nx=2*nx+1;
  AA=zeros(Nx);
  BB=zeros(Nx);
  AA[1:nx]=A;
  BB[1:nx]=B;
  CC=zeros(nx);
  for i in 1:nx 
    shift_constant=i-div(nx,2)-1
    CC[i]=sum(AA.*circshift(BB,shift_constant))/sum(BB);
  end
  return CC
end

function thermal_convolutionN(A::Cube,B::Cube)
  #==
  DH: optimized convolution for CPU
  A : ppv Cubes
  B : Thermal Function
  ==#
  CC = real(fftshift(ifft(fft(A,3).*conj(fft(B,3)),3),3));
  return CC
end

function thermal_convolution(A::Vec,B::Vec)
  #==
  KH: The convolution formula is X[v] = \int v' A[v']B[v+v']
  Convolution theorem applies
  ==#
  nx=size(A)[1];
  Nx=2*nx-1;
  AA=zeros(Nx);
  BB=zeros(Nx);
  AA[1:nx]=A;
  BB[1:nx]=B;
  CC=real(fftshift(ifft(fft(AA).*conj(fft(BB)))));
  AA=0;BB=0;
  return CC
end

function wiener_deconvolution(C::Vec,B::Vec;N=0)
  #==
  KH: There is a need for deconvoluting the PPV cubes
  https://en.wikipedia.org/wiki/Wiener_deconvolution

  Formulation:
  Assume y = (h*x)(t)+n(t)
  target is to find x=(g*y)(t)
  which that g in fourier space, G, is
  G = (H*) S/(H^2 S+N)
  where
  S=power spectrum of x 
  N=power spectrum of n

  in the no-noise case
  G = 1/H
  which we will assume here.

  For our case, we are doing
  C=fftshift(A*conj(B))
  so a = ifft(ifftshift(C)./conj(B))
  ==#
  nx=size(C)[1];
  CC=C;
  BB=zeros(size(C));
  BB[1:length(B)] .= B;
  nnx=div(nx+1,2);
  if (N==0)
    A=real(ifft(fft(ifftshift(CC))./conj(fft(BB)))[1:nnx])
  else
    A=real((fft(BB)[1:nnx].*fft(ifftshift(CC)))./((abs.(fft(BB)[1:nnx])).^2.0.+N));
  end
  return A
end

#==
function thermal_function(velocity_scale::Vec,cs::Number)
  # KH : Returns the thermal Gaussian according to the thermal speed cs
    return exp.(-velocity_scale.^2.0./(2.0.*cs.^2))./sqrt.(2.0.*pi.*cs.^2)
end
KH: A normalization factor is missing due to discretization. Fixed below.
==#

function thermal_function(velocity_scale::Vec,cs::Number)
  # KH : Returns the thermal Gaussian according to the thermal speed cs
    a=exp.(-velocity_scale.^2.0./(2.0.*cs.^2))./sqrt.(2.0.*pi.*cs.^2)
    return a./sum(a);
end

function thermal_function_Array(p::Array,velocity_scale::Array,cs::Number)
    # KH : Returns the thermal Gaussian according to the thermal speed cs
    # DH : Array versio of thermal Gaussian problem
    Nx,Ny,Nv = size(p);
    a = exp.(-velocity_scale.^2.0./(2.0.*cs.^2))./sqrt.(2.0.*pi.*cs.^2);
    a = reshape(vcat(a./sum(a),zeros(Nv-1)),(1,1,2*Nv-1));
    A = repeat(a,outer=[Nx,Ny,1]);
    return A
end

function ppv_thermal(d::Cube,v::Cube,binnum::Number,cs::Number;multithreading=false)
  # KH : using Blakesley's smoothing  
  p=ppv(d,v,binnum);
  vmin=minimum(v);
  vmax=maximum(v);
  vrange=lspace(vmin,vmax,binnum)
  nx,ny,nv=size(p);
  pp=zeros(nx,ny,2*nv-1);
  if (multithreading && (nthreads()>1))
    @threads for i in 1:nx
      for j in 1:ny 
        pp[i,j,:].=thermal_convolution(p[i,j,:],thermal_function(vrange,cs))
      end
    end
  else 
    thFunc= thermal_function_Array(p,vrange,cs); #Intialization process for convolution
    p     = cat(p,zeros(nx,ny,nv-1);dims=3); #Intialization process for convolution
    pp    = thermal_convolutionN(p,thFunc);  #Single Core Optimized Code
  end
  return pp,vrange
end

function ppv_thermal_from_ppv(p::Cube,vmin::Number,vmax::Number,binnum::Number,cs::Number)
  # KH : using Blakesley's smoothing, input to be ppv
  vrange=lspace(vmin,vmax,binnum)
  nx,ny,nv=size(p);
  pp=zeros(nx,ny,2*nv-1);
  for i in 1:nx, j in 1:ny 
    pp[i,j,:].=thermal_convolution(p[i,j,:],thermal_function(vrange,cs))
  end
  return pp,vrange
end

function ppv_deconvolution(p::Cube,vrange::Vec,cs::Number)
  nx,ny,Nv=size(p)
  pp=zeros(nx,ny,div(Nv+1,2));
  for i in 1:nx, j in 1:ny 
    pp[i,j,:].=wiener_deconvolution(p[i,j,:],thermal_function(vrange,cs))
  end 
  return pp
end

end # module LazThermal
