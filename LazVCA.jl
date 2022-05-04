module LazVCA

# using PyCall
using LsqFit
using FFTW # fft, ifft
using Statistics # var, std
using LazType
using LazCore
using LazThermal

##############################################################################
#
# Copyright (c) 2019
# Ka Ho Yuen, Yue Hu and Alex Lazarian
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
module LazVCA
calling spectrallines,spectralindex

## VCA Equation
Pv(k) propto k^-β
where β=1+2*(minimum(2,β_I)-β_Ch)

## VCA table: Assuming P(ρ) ~ k^(-3+r)
2D
| Slice thickness   |		r>0			|		r<0		|
|:-----------------:|:-----------------:|:-------------:|
| thin 			 	|  k^(-3+r+m/2)		|	k^(-3+m/2)	|
| thick 			|	k^(-3+r)		|	k^(-3-m/2)	|
| very thick 		|	k^(-3+r)		|	k^(-3+r)	|


"""



export VCA,spectrallines_3d,spectralindex_3d,spectrallines_3d_vector

line(x,p)=p[1].+x.*p[2];

function VCA(I::Mat,Ch::Mat;xmin=10,xmax=40)
	β_I =spectralindex(I ,xmin,xmax);
	β_Ch=spectralindex(Ch,xmin,xmax);
	return 1+2*(minimum([2,β_I])-β_Ch)
end


function spectrallines_3d(A::Cube)
 nx,ny,nz=size(A);
 nl=round(Int,sqrt(nx^2+ny^2+nz^2));
 kn=zeros(nl);
 kl=zeros(nl);
 Ax2=fftshift((abs.(fft(A))).^2.0);
 for i in 1:nx, j in 1:ny, k in 1:nz
  idx=i-div(nx,2);
  jdx=j-div(ny,2);
  kdx=k-div(nz,2);
  rr=round(Int,sqrt(idx^2+jdx^2+kdx^2));
  if ((rr<=nl) & (rr>0))
   kn[rr]+=Ax2[i,j,k];
   kl[rr]=rr;
  end
 end
 return kn,kl
end

function spectralindex_3d(A::Cube,xmin::Number,xmax::Number)
  xran2=xmin:xmax
  kvn,kvl=spectrallines_3d(A)
  fitv=curve_fit(line,log10.(xran2),log10.(kvn[xran2]),[rand(2)...]);
  return fitv.param[2];
end

function plot_spectrum(A::Cube)
    kn,kl=spectrallines3D(A);
    scatter(kl[kn.>0],kn[kn.>0])
    xscale("log")
    yscale("log")
    axis([1,300,minimum(kn[kn.>0])*0.5,maximum(kn[kn.>0])*2])
    xxx=curve_fit(line,log10.(kl[10:50]),log10.(kn[10:50]),[rand(2)...]).param;
    plot(10:50,10.0.^(line(log10.(10:50),xxx)),color="r")
    title(round(xxx[2],digits=4));
    return 0;
end

function spectrallines_3d_vector(A::Cube,B::Cube,C::Cube)
 nx,ny,nz=size(A);
 nl=round(Int,sqrt(nx^2+ny^2+nz^2));
 kn=zeros(nl);
 kl=zeros(nl);
 Ax2=fftshift((abs.(fft(A))).^2.0.+(abs.(fft(B))).^2.0.+(abs.(fft(C))).^2.0);
 for i in 1:nx, j in 1:ny, k in 1:nz
  idx=i-div(nx,2);
  jdx=j-div(ny,2);
  kdx=k-div(nz,2);
  rr=round(Int,sqrt(idx^2+jdx^2+kdx^2));
  if ((rr<=nl) & (rr>0))
   kn[rr]+=Ax2[i,j,k];
   kl[rr]=rr;
  end
 end
 return kn,kl
end



end # module LazVCA