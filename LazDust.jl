module LazDust
using PyCall,FITSIO,FFTW,Statistics
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
  KH: Dust emission module
  Jul 15 2019
==#
export dust_emissions,synchrotron_emissions,dust_emission_chen
export cmean,cstd,density_dist_generator

function dust_emissions(d::Cube,ib::Cube,jb::Cube,kb::Cube;field_order=2.0)
	# KH: Assuming the mean field is along k direction while projecting along i direction
	theta=atan.(jb./kb);
	cbn=cos.(field_order.*theta);
	sbn=sin.(field_order.*theta);
	mag=sqrt.(jb.^2.0.+kb.^2.0);
	nx,ny,nz=size(d);
	In=reshape(sum(d.*mag.^field_order,dims=1),ny,nz);
	Qn=reshape(sum(d.*cbn.^field_order,dims=1),ny,nz);
	Un=reshape(sum(d.*sbn.^field_order,dims=1),ny,nz);
	return In,Qn,Un
end

function synchrotron_emissions(d::Cube,ib::Cube,jb::Cube,kb::Cube;field_order=2.0)
	# KH: Assuming the mean field is along k direction while projecting along i direction
	theta=atan.(jb./kb);
	cbn=cos.(field_order.*theta);
	sbn=sin.(field_order.*theta);
	mag=sqrt.(jb.^2.0.+kb.^2.0);
	nx,ny,nz=size(d);
	In=reshape(sum(d.*mag.^field_order,dims=1),ny,nz);
	Qn=reshape(sum(d.*(mag.*cbn).^field_order,dims=1),ny,nz);
	Un=reshape(sum(d.*(mag.*sbn).^field_order,dims=1),ny,nz);
	return In,Qn,Un
end



function dust_emission_chen(d::Cube,ib::Cube,jb::Cube,kb::Cube;field_order=2.0,mmw=1,pmax=0.2)
	# Reference: Chen+2019, MNRAS 485, 3499–3513 (2019)
	# KH: One is assuming d/mmw = n. Here we simply assume mmw=1
	# Here Chen's gamma = 90 - P's gamma
	n=d./mmw;
	theta=atan.(jb./kb);
	magp=sqrt.(jb.^2.0.+kb.^2.0);
	mag=sqrt.(ib.^2.0.+jb.^2.0.+kb.^2.0);
	cosgamma=magp./mag;
	nx,ny,nz=size(d);

	# IQU generation
	cbn=cos.(field_order.*theta);
	sbn=sin.(field_order.*theta);
	Qn=reshape(sum(n.*cbn.^field_order.*cosgamma.^2.0,dims=1),ny,nz);
	Un=reshape(sum(n.*sbn.^field_order.*cosgamma.^2.0,dims=1),ny,nz);

	# pmax -> p0
	p0=3.0.*pmax./(3.0.+pmax);

	# correction term
	N2=reshape(sum(n.*(cosgamma.^2.0.-2.0./3.0),dims=1),ny,nz);
	N=reshape(sum(n,dims=1),ny,nz)

	# polarization percentage
	p=p0.*sqrt.(Qn.^2.0.+Un.^2.0)./(N.-p0.*N2);

	# gamma_obs, Eq(10) of Chen+19
	gamma_obs=acos.(sqrt.(p.*(1.0.+2.0./3.0.*p0)./(p0.*(1.0.+p))));
	return Qn,Un,p,gamma_obs
end

# circular statistics functions


function cmean(theta::Cube)
	stheta=sin.(theta);
	ctheta=cos.(theta);
	mtheta=atan.(mean(stheta)./mean(ctheta))
	return mtheta
end

function cstd(theta::Cube)
	stheta=sin.(theta);
	ctheta=cos.(theta);
	R=sqrt(mean(stheta)^2+mean(ctheta)^2);
	return sqrt(-2*log(R))
end

function cmean(theta::Mat)
	stheta=sin.(theta);
	ctheta=cos.(theta);
	mtheta=atan.(mean(stheta)./mean(ctheta))
	return mtheta
end

function cstd(theta::Mat)
	stheta=sin.(theta);
	ctheta=cos.(theta);
	R=sqrt(mean(stheta)^2+mean(ctheta)^2);
	return sqrt(-2*log(R))
end

##

function density_dist_generator(sigma::Number,rho_c::Number,m::Number,s::Number)
    # KH (Aug 30): a Monte-Carlo based random number generator for a pdf
    # in the form of log-Gaussian (sigma) plus a power law (-m), with a cut-off at rho_c
    # Basic logic:
    # (1) Infinite uniform generator x to select a real number
    # (2) [0,Infty) uniform generator to select a y
    # (3) If (x,y) stays below the curve, throws out the value rho=exp.(x)
    # (4) In any case go back to (1)
    # Caution: Here we select [-5,5] for the Gaussian and power-law log(rho) cutoff
    rho=zeros(0);
    ii=0;
    while (ii<s)
     x=rand(Uniform(-5*sigma,5*sigma));
     y=rand(Uniform(0,1));
     if (exp(x)<rho_c)
      if (y<exp(-x^2/2/sigma/sigma))
       push!(rho,exp(x))
       ii+=1;
      end
     else
      # The amplitude has to be corredted
      if (y<exp(m*(x-log(rho_c)))*exp(-log(rho_c)^2/2/sigma))
       push!(rho,exp(x));
       ii+=1;
      end
     end
    end
    return rho
end


function sbam2d(Ax::Mat,Ay::Mat,dn)
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
	   Aaa=atan.(Ayy,Axx);
	   binsize=dn;
	   Ana[i,j]=cmean(Aaa);
	   Ans[i,j]=cstd(Aaa);
	end
	return Ana,Ans
end


#==
function atann(U::Mat,Q::Mat;n=2)

end
==#
end # module LazDust