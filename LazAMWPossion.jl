module NMW_Beta
using PyCall
@pyimport numpy as np
@pyimport scipy.signal as signal
typealias Cube{T} Array{T,3}
typealias Mat{T} Array{T,2}
typealias Vec{T} Array{T,1}

export MovingWindows

##############################################################################
#
# Copyright (c) 2019
# Ka Ho Yuen, Ka Wai Ho, Yue Hu and Alex Lazarian
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
#
# # This Code is now under the format of Julia 0.4~0.5
# KW HO @ LazTech 
# April 4 2019
# Suppose:This Code produce the MW Map by resolving the 2D Vector Map B and output a div free Map.  
# Method: Suppose Vector Map B = B' + C
#         Where div(B) = div(B') + div(C) and div(B') = 0, div(B)=f(x,y);
#         If we consider C => Grad(Phi) , Where div(Grad(Phi)) => Delta^2 (Phi) = div(B) = f(x,y)
#         So, The Code would:
#         1. The Code is solving Possion Eqation of Phi using the fourier method 
#         2. returning the div(B') Map by Calculating B - Grad(Phi)
# 
# Some of the Implementation in the code from https://github.com/bhatiaharsh/naturalHHD by: Harsh Bhatia (bhatia4@llnl.gov)
#          
#
##################################################################################



#===== THIS CODE IS TO SOLVE POSSION EQUATION IN 2D CASE  ========#

function Curl(u::Mat,v::Mat,dx::Number,dy::Number)
	dudy, dudx = np.gradient(u, dx, dy);
	dvdy, dvdx = np.gradient(v, dx, dy);
	Curl = dvdx - dudy;
	Curl;
end

function Div(u::Mat,v::Mat,dx::Number,dy::Number)
   	dudy, dudx = np.gradient(u, dx, dy);
	dvdy, dvdx = np.gradient(v, dx, dy);
	divF = dvdy + dudx;
	divF;
end

function PossionSolver2D(SolMap::Mat,dx::Number,dy::Number)
    GreenMap = GreensFunctiionMap(SolMap);
    p   = signal.fftconvolve(SolMap,GreenMap, mode="same");
    Sol = p.*(dx*dy);
    Sol
end

function GreensFunctiionMap(Map::Mat)
	nx,ny = size(Map);
	Nx,Ny = 2*nx-1,2*ny-1;
	dx,dy = 1, 1;
    GreensMap = zeros(Nx,Ny);
    halfLx, halfLy = div(Nx,2),div(Ny,2);
    for j in 1:Ny
    	for i in 1:Nx
    		x = (i-halfLx)*dx;
    		y = (j-halfLy)*dy;
    		GreensMap[i,j] = sqrt(x*x+y*y);  
    	end
    end
    GreensMap[halfLx,halfLy] = 0.5*mean([dx,dy]);
    GreensMap = log(GreensMap);
    return 0.5.*GreensMap./pi
end

function MovingWindows(GradMap::Mat)
	u = cos(GradMap);
	v = sin(GradMap);
	DivB = Div(u,v,1,1) ;
	Phi  = PossionSolver2D(DivB,1,1);
	Gradu,Gradv = np.gradient(Phi,1,1);
	U = u.- Gradu;
	V = v.- Gradv;
	return atan2(U,V);   
end
end

#====================================Testing Code ===================================#
#Under Construction
using PyPlot
function plotangle(A::Mat,B::Mat)
	nx,ny = size(A);
	dn = 1;
	Xd,Yd=np.meshgrid(div(dn,2)+1:dn:div(dn,2)+ny,div(dn,2)+1:dn:div(dn,2)+nx);
	quiver(Yd,Xd,sin(A.+pi/2),cos(A.+pi/2),headwidth=0,scale=50,color="b",label="A Map")
	quiver(Yd,Xd,-sin(A.+pi/2),-cos(A.+pi/2),headwidth=0,scale=50,color="b")
	quiver(Yd,Xd,sin(B.+pi/2),cos(B.+pi/2),headwidth=0,scale=50,color="r",label="B Map")
	quiver(Yd,Xd,-sin(B.+pi/2),-cos(B.+pi/2),headwidth=0,scale=50,color="r")
	legend(loc="best")
end

#Case 1.
GM = fill(pi/4,(30,30));
GM[10,20] = GM[16,3] = GM[22,5] = 0.5;
GM[6,7]   = GM[29,21]= GM[15,24]= 1.2;
B = MovingWindows(GM);
plotangle(GM,B)
#Case 2
function testfield1(N)
	Field = zeros(N,N)
	nx,ny = size(Field);
	for j in 1:ny
		for i in 1:nx
			Field[i,j] = atan2(i-div(N,2)-1,j-div(N,2)-1);
		end
	end
	Field;
end
A = testfield1(100);
nx=ny=100;
dn = 1;
