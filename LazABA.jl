module LazABA
using PyCall,FITSIO,FFTW,Statistics,LsqFit,StatsBase,LinearAlgebra
using LazRHT,LazThermal,LazCore,LazType

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



export recursive_gaussian, fit_supergaussian_2d
# export ,aba2d

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

function recursive_gaussian(number_of_gaussian::Number);
 Gaus(x,p)=p[1]*exp.(.-(x.-p[2]).^2 .*p[3]);
 Gausnew(x,p)=p[1]*exp.(.-(x.-p[2]).^2 .*p[3]);
 for i in 2:number_of_gaussian
  Gausnew = let Gausold=Gausnew
   (x,p)->Gaus(x,p[(i-1)*3+1:i*3]).+Gausold(x,p[1:(i-1)*3]);
  end
 end
 return Gausnew
end


function fit_supergaussian_2d(Ax::Mat,Ay::Mat,binsize,gaussize)
 phi=atan.(Ay./Ax)
 Gausnew=recursive_gaussian(gaussize);
 ax,ac=hist_new(phi[.~isnan.(phi)][:],-pi/2:0.01:pi/2);
 ax=.5*(ax[1:end-1]+ax[2:end]);
 if (abs(ax[maxid(ac)])[1]<pi/4)
  fit=curve_fit(Gausnew,ax,ac/sum(ac),[rand(3*gaussize)...])
 else
  ax=ax-pi/2;
  ac=fftshift(ac);
  fit=curve_fit(Gausnew,ax,ac/sum(ac),[rand(3*gaussize)...]);
 end
 #sigma=estimate_errors(fit,0.95);
 return fit.param     #,sigma;
end

#==
function aba2d(Ax::Mat,Ay::Mat,dn::Number)
 # KH: The idea is to create a multiple-Gaussian function
 # And fit against the curve
 # The linearity is controlled by dn
 
 # Let's define the Maximum linear Gaussian number to be the block size!
 nx,ny=size(Ax)
 Ana=zeros(div(nx,dn),div(ny,dn));
 Ans=zeros(div(nx,dn),div(ny,dn));

 for  j in 1:div(ny,dn),i in 1:div(nx,dn)
  is=(i-1)*dn+1;
  ie=i*dn;
  js=(j-1)*dn+1;
  je=j*dn;
  Axx=Ax[is:ie,js:je];
  Ayy=Ay[is:ie,js:je];
  binsize=dn;
  amp=zeros(gauss);
  amperr=zeros(dn);
  ori=zeros(dn);
  orierr=zeros(dn);
  oritc=0;
  orits=0;
  oriterr=0;
  gaussize=round(Int,sqrt(dn));
  fitp=fit_supergaussian_2d(Axx,Ayy,binsize,gaussize);
  # For convenience we will use two for-loops
  # Esp for error calculation
  for k in 1:gaussize
   amp[k]=fitp[(k-1)*3+1];
   ori[k]=fitp[(k-1)*3+2];
   #amperr[k]=sigma[(k-1)*3+1];
   #orierr[k]=sigma[(k-1)*3+1];
   oritc+=amp[k]*cos(ori[k]);
   orits+=amp[k]*sin(ori[k]);
  end
  #for k in 1:gaussize
  # cfac=oritc/(oritc^2+orits^2);
  # sfac=oritc/(oritc^2+orits^2);
  # ds=amperr[k]*cos(ori[k])-amp[k]*sin(ori[k])*orierr[k];
  # dc=amperr[k]*sin(ori[k])+amp[k]*cos(ori[k])*orierr[k];
  # oriterr+=cfac*ds-sfac*dc
  #end
  orit=atan(orits/oritc);
  Ana[i,j]=orit
  #Ans[i,j]=oriterr
 end
 return Ana,Ans
end


==#


end # module advanced_block_averaging