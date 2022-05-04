module LazTurbStat
using StatsBase,HDF5, LsqFit, PyCall, FITSIO, Images, PyPlot, FFTW, Statistics
using LazCore, LazType, LazThermal
using Base.Threads
np=pyimport("numpy");

##############################################################################
#
# Copyright (c) 2020
# Ka Wai Ho, Ka Ho Yuen, Yue Hu, Junda Chen and Alex Lazarian
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
#                                                                             #
#                        	  Code                                        # 
#                                                                             #
#                          Turbulence Statistics                              #
#                                                                             #
###############################################################################
#   
#
#	Version     : v1.0.1(20/2/2020)
#       Author	    : KW HO , KH Yuen@ Lazarian Technology
#       Description : Providing basic turbulence statistics function
#       Capability  : Work for julia version 1.2.0
#	Caution     : Work for Periodic Map or Cube only!
#
###############################################################################


# Auto correlation function
export CF
# 2 point structure function related
export SF3D,SFr,SFr_vec
# multithread version (experimental)
export SFr_vec_mt
# contour extraction algorithm
export Getcontour,fitline

function s(vx::Cube,vy::Cube,vz::Cube)
	return vx.*vx.+vy.*vy.+vz.*vz;
end

function s(vx::Number,vy::Number,vz::Number)
	return vx.*vx.+vy.*vy.+vz.*vz;
end


function t(a1,a2,a3,b1,b2,b3)
	x=dot_product(a1,a2,a3,b1,b2,b3)/sqrt(s(a1,a2,a3))/sqrt(s(b1,b2,b3));
	if (x>1) 
		x=1 
	end;
	if (x<-1)
		x=-1
	end;
	return acos(x)
end

function dot_product(a1,a2,a3,b1,b2,b3)
	return a1*b1+a2*b2+a3*b3
end

function dot_product(a::Vec,b::Vec)
	return sum(a.*b);
end

function cross_product(a1,a2,a3,b1,b2,b3)
	#==
	| x̂  ŷ  ẑ|
	|a1 a2 a3|
	|b1 b2 b3|
	==#
	return [a2*b3-a3*b2,a3*b1-a1*b3,a1*b2-a2*b1]
end

function RoundUpInt(x::Number)
   return Int(round(x,RoundUp)) 
end

function RoundInt(x::Number)
   return Int(round(x)) 
end


function getind(kll::Number,kpp::Number,Rx::Number,Ry::Number)
    rpar  = RoundInt(kll);
    rperp = RoundInt(kpp);
    return rpar,rperp 
end

#Auto-Correlation function for 3D **periodic** Cube
CF(V::Cube) = fftshift((real(ifft(abs.(fft(V)).^2))));

#Auto-Correlation function for 2D **periodic** Map
CF(V::Mat) = fftshift((real(ifft(abs.(fft(V)).^2))));

function SFC(V::Cube) 
    CF(V::Cube) = fftshift((real(ifft(abs.(fft(V)).^2))));
    sf = 2*(mean(V).-CF(V));
    return sf
end

#provding 3D 2 point structure point function for scalar 
#functional form: SF(R) = <|V(x+R)-V(x)|^2>
#Note : R is 3D vector 
function SF3D(V::Cube) 
    CF(V::Cube) = fftshift((real(ifft(abs.(fft(V)).^2))));
    sf = 2*(mean(V).-CF(V));
    return sf
end

#provding 2D 2 point structure point function for scalar 
#functional form: SF(R) = <|V(x+R)-V(x)|^2>
#Note : X-axis means prallel to B-field, Y-axis means perpendicular to B-field
function SFr(V::Cube,bx::Cube,by::Cube,bz::Cube)
	# get the vector size
    Nx,Ny,Nz = size(V);
    R        = sqrt(s(div(Nx,2),div(Ny,2),div(Nz,2)));

    # get the mean field
    mbx= mean(bx);
    mby= mean(by);
    mbz= mean(bz);
    
    # get the structure function 
    SFV = SFC(V);
    Rx,Ry = RoundUpInt(R), RoundUpInt(R);
    # declaring the output 
    Mask = zeros((Rx+1,Ry+1));
    SFVr = zeros((Rx+1,Ry+1));
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        # get the k vector
	idx = i-div(Nx,2);
	jdx = j-div(Ny,2);
	kdx = k-div(Nz,2);
	kk  = sqrt(s(idx,jdx,kdx));
	if kk > 0
		#get the θ between b unit vector and k vector
		θ  = t(mbx,mby,mbz,idx,jdx,kdx)

		#get the 2D vector parallel and perpendicular to B-field
		kll = abs(kk*cos(θ));
		kpp = abs(kk*sin(θ));

		rpar,rperp       = getind(kll,kpp,Rx,Ry).+1; #prob2
		Mask[rpar,rperp] += 1; 
		SFVr[rpar,rperp] += SFV[i,j,k];
	end
    end
    SFVr./=Mask;
    SFVr
end

#provding 2D 2 point structure point function for vector 
#functional form: SF(R) = <|Vec(x+R)-Vec(x)|^2>
#Note : X-axis means prallel to B-field, Y-axis means perpendicular to B-field
function SFr_vec(Vx::Cube,Vz::Cube,Vy::Cube,bx::Cube,by::Cube,bz::Cube)
    Nx,Ny,Nz = size(Vx);
    R        = sqrt(s(div(Nx,2),div(Ny,2),div(Nz,2)));

    #get the mean field
    mbx= mean(bx);
    mby= mean(by);
    mbz= mean(bz);
    
    #get the structure function 
    SFVx  = SFC(Vx);
    SFVy  = SFC(Vy);
    SFVz  = SFC(Vz);
    #get the vector structure function
    SFV   = SFVx.+SFVy.+SFVz;
    #declaring the output
    Rx,Ry = RoundUpInt(R), RoundUpInt(R);
    Mask = zeros((Rx+1,Ry+1));
    SFVr = zeros((Rx+1,Ry+1));

    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        # get the k vector
	idx = i-div(Nx,2);
	jdx = j-div(Ny,2);
	kdx = k-div(Nz,2);
	kk  = sqrt(s(idx,jdx,kdx));
	if kk>0
		#get the θ between b unit vector and k vector
		θ  = t(mbx,mby,mbz,idx,jdx,kdx)

		#get the 2D vector parallel and perpendicular to B-field
		kll = abs(kk*cos(θ));
		kpp = abs(kk*sin(θ));

		rpar,rperp       = getind(kll,kpp,Rx,Ry).+1; #prob2
		Mask[rpar,rperp] += 1; 
		SFVr[rpar,rperp] += SFV[i,j,k];
	end
    end
    SFVr./=Mask;
    SFVr
end

#==
 KH: Multithread version
 Not tested, used with caution (usually it's the memory overflow)
==#

function SFr_vec_mt(Vx::Cube,Vz::Cube,Vy::Cube,bx::Cube,by::Cube,bz::Cube)
    Nx,Ny,Nz = size(Vx);
    R        = sqrt(s(div(Nx,2),div(Ny,2),div(Nz,2)));

    #get the mean field
    mbx= mean(bx);
    mby= mean(by);
    mbz= mean(bz);
    
    #get the structure function 
    SFVx  = SFC(Vx);
    SFVy  = SFC(Vy);
    SFVz  = SFC(Vz);
    #get the vector structure function
    SFV   = SFVx.+SFVy.+SFVz;
    #declaring the output
    Rx,Ry = RoundUpInt(R), RoundUpInt(R);
    Mask = zeros((Rx+1,Ry+1));
    SFVr = zeros((Rx+1,Ry+1));

    @threads for k in 1:Nz
     for j in 1:Ny, i in 1:Nx
        # get the k vector
    idx = i-div(Nx,2);
    jdx = j-div(Ny,2);
    kdx = k-div(Nz,2);
    kk  = sqrt(s(idx,jdx,kdx));
    if kk>0
        #get the θ between b unit vector and k vector
        θ  = t(mbx,mby,mbz,idx,jdx,kdx)

        #get the 2D vector parallel and perpendicular to B-field
        kll = abs(kk*cos(θ));
        kpp = abs(kk*sin(θ));

        rpar,rperp       = getind(kll,kpp,Rx,Ry).+1; #prob2
        Mask[rpar,rperp] += 1; 
        SFVr[rpar,rperp] += SFV[i,j,k];
    end
    end
    end
    SFVr./=Mask;
    SFVr
end

# Contour gettting function, by Ka Wai Ho.

function Getcontour(V::Mat,levels)
    #==
    Example to use
    rllsHBp,rpersHBp = Getcontour(SFVrHBs⊥[1:120,1:120],50);
    subplot(122)
    Ns,Nl = 1,20
    loglog((rpersHBp[Ns:Nl]),(rllsHBp[Ns:Nl]),"o");
    fitline(log10.(rpersHBp[Ns:Nl]),log10.(rllsHBp[Ns:Nl]),"β >> 1");
    ==#
    imshow(V,cmap="Blues_r")
    Nx,Ny = size(V);
    iac = contour(V,levels=levels,cmap="Dark2")
    axis([1,Nx,1,Ny])
    Conlevel = length(A.allsegs)
    x = zeros(Float64,Conlevel);
    y = zeros(Float64,Conlevel);
    for i = 1:Conlevel
        tor=1e-3
        if (length(A.allsegs[i])>0)
            if ((A.allsegs[i][1][1,1]<tor) && (A.allsegs[i][1][end,end]<tor))
                x[i],y[i]  = NaN,NaN;
            end
            x[i],y[i] = A.allsegs[i][1][1,end],A.allsegs[i][1][end,1]
        else
        x[i],y[i]  = NaN,NaN;
        end
    end
    # output is r_ll,r_pp.
    scatter(x[.~isnan.(y) .& (x.>0) .& (x.<Nx)],y[.~isnan.(y).& (x.>0) .& (x.<Ny)])
end


# Fitting function, by Ka Wai Ho.
function fitline(x::Array,y::Array,label)
    line(x,p)=p[1].+x.*p[2];
    p = [0,y[2]-y[1]];
    xxx=curve_fit(line,x,y,p).param;
    m,C = round(xxx[2],digits=2),round(10^(xxx[1]),digits=3)
    plot([10.0].^x,[10.0].^line(x,xxx),label="m = $m ,C = $C, "*label);
    legend()
    return xxx
end


end
