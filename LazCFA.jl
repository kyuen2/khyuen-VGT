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
  module LazCFA
The correlation function anisotopy module 
Include:
  The old modules of rotate2d.jl, GreensFunction.jl, ellipse.jl
Author: Ka Ho Yuen, Yue Hu, Dora Ho, Junda Chen
Changelog:
  - Ka Ho initialize LazCFA
  - 5/3/2020: KW HO introduce new CFA method, the ellipse fitting method.
Todo:
"""

module LazCFA

# using PyCall
using LsqFit
using StatsBase
using LinearAlgebra
using FFTW

using LazType
using LazIO
using LazCore
using PyPlot
using PyCall
np = pyimport("numpy");
pygui(false)

############################
# Export Symbols
############################

# internal functions
# """ internal meshgrid """ 

# rotate2d.jl
# """ internal New_Map,rotate_2d """

# Supermodule.jl
# + == Future
# + Move to CFA (temporarily removed)
# + structfunc, corrfunc, SF2
# + structfunc_rand, corrfunc_rand
# + cal2ndmoment, anisocal_quick


# GreensFunction.jl
export crosscorr,autocorr

# ellipse.jl
# """ internal angle_back,dydx """
export ellipse_axis,sban_cfa
export rotate_2d



############################
# Implementation
############################


# """@internal"""

function meshgrid(X,Y)
  # KH : native implementation of meshgrid
  #      The order of the meshgrid is **different from** python
  #      X,Y are ranges.
  return [ i for i=X, j=Y ], [ j for i=X, j=Y ]
end


  # # rotate2d.jl
    # New_Map,rotate_2d
    # KH : Written by Dora Ho (CUHK) to perform the CFA ellipse fitting algorithm
    #      Used in Lazarian et. al (2018)

function New_Map(Block::Mat,angle::Number)
    new_x,new_y=size(Block)
  Y,X=meshgrid(1:new_x,1:new_y)
    #shift the coordinates system to the centre first
    X=X.-round(Int,new_x/2)   
    Y=Y.-round(Int,new_y/2)
  X_dot= X.*cos(angle)-Y.*sin(angle)
  Y_dot= X.*sin(angle)+Y.*cos(angle)
  return X_dot,Y_dot
end


function rotate_2d(I::Mat,angle::Number)
    nx,ny=size(I);
    cut_2=round(Int,nx/2);
    # KH : The periodic rotation function developed by Dora Ho (CUHK)
    # His original comment: 
    # this is a puzzle game, you divid the amp into 1,2,3,4 block and using this four to expand it. 
    # It looks like this
    # [1,2]        #  [3,4,3,4]
    # [3,4]    =>  #  [2,1,2,1]
    #             #  [4,3,4,3]
    #             #  [2,1,2,1]
    BLock=Dict( "Block_2"=>I[1:cut_2,cut_2+1:end],
                "Block_4"=>I[cut_2+1:end,cut_2+1:end],
                "Block_1"=>I[1:cut_2,1:cut_2],
                "Block_3"=>I[cut_2+1:end,1:cut_2],)
    Left_Block  =vcat(BLock["Block_2"],BLock["Block_4"]) 
    Right_Block =vcat(BLock["Block_1"],BLock["Block_3"])
    upper_Block =hcat(BLock["Block_4"],BLock["Block_3"],BLock["Block_4"],BLock["Block_3"])
    buttom_Block=hcat(BLock["Block_2"],BLock["Block_1"],BLock["Block_2"],BLock["Block_1"])
    #menrge those blocks into a big map
    Big_Block    =vcat(upper_Block,hcat(Left_Block,I,Right_Block),buttom_Block) 
    Rn_x,Rn_y=New_Map(I,angle)
    # find the new coordinates after rotating the small map
    R_Map=zeros(nx,ny);
    for i in 1:ny, j in 1:nx
        R_Map[i,j]=Big_Block[round(Int,Rn_x[i,j]+nx),round(Int,Rn_y[i,j]+ny)] #this line may have problem
    end
    return R_Map'
end

#==
    Rotation module from v0.5-R2.jl
@noinline function rodrigue_rotation_s(iv,jv,kv,angle,rotation_axis_number)
 # KH: This is for angle
 # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formul
 if (rotation_axis_number==1)
  jvp=jv.*cos(angle)-kv.*sin(angle)
  kvp=kv.*cos(angle)+jv.*sin(angle)
  return iv,jvp,kvp
 elseif (rotation_axis_number==2)
  kvp=kv.*cos(angle)-iv.*sin(angle)
  ivp=iv.*cos(angle)+kv.*sin(angle)
  return ivp,jv,kvp
 elseif (rotation_axis_number==3)
  ivp=iv.*cos(angle)-jv.*sin(angle)
  jvp=jv.*cos(angle)+iv.*sin(angle)
  return ivp,jvp,kv;
 else
  println("KH: WTF are you projecting?");
  return 0; 
 end
end
@noinline nmeshgrid(nx,ny,nz)=np.meshgrid(1:nx,1:ny,1:nz);
function meshgrid(nx,ny,nz)
  X,Y,Z=zeros((nx,ny,nz)),zeros((nx,ny,nz)),zeros((nx,ny,nz));
  for i=1:nx
    X[i,:,:]=i;
  end
  for j=1:ny
    Y[:,j,:]=j;
  end
  for k=1:nz
    Z[:,:,k]=k;
  end
  return X,Y,Z
end
@noinline function New_Map(Block::Cube,anglex::Number,angley::Number,anglez::Number)
    nx,ny,nz=size(Block);
      Y,X,Z=nmeshgrid(nx,ny,nz);
    gc();
    #shift the coordinates system to the centre first
    X.-=Int(nx/2);  
    Y.-=Int(ny/2);
    Z.-=Int(nz/2);
    X,Y,Z=rodrigue_rotation_s(X,Y,Z,anglex,1);
    X,Y,Z=rodrigue_rotation_s(X,Y,Z,angley,2);
    X,Y,Z=rodrigue_rotation_s(X,Y,Z,anglez,3);
    return X,Y,Z
end
function Vector_New_Map(Vx::Cube,Vy::Cube,Vz::Cube,anglex::Number,angley::Number,anglez::Number)
  iVx,jVx,kVx=rodrigue_rotation_s(Vx,Vy,Vz,anglex,1);
  Vx,Vy,Vz=0,0,0;gc();
  iVy,jVy,kVy=rodrigue_rotation_s(iVx,jVx,kVx,angley,2);
  iVx,jVx,kVx=0,0,0;gc();
  iVz,jVz,kVz=rodrigue_rotation_s(iVy,jVy,kVy,anglez,3);
  iVy,jVy,kVy=0,0,0;gc();
  return iVz,jVz,kVz
end
function pointer_function(value::Cube,nx::Int)
    #pointer the coordinates back to the small Cube
    half_nx=div(nx,2)
    for k in eachindex(value)
      if (value[k]>half_nx)
          n=div(value[k],nx)
          value[k]-=(n+1)*nx
      elseif -1*value[k]>half_nx
          n=div(abs(value[k]),nx)
          value[k]+=(n+1)*nx
      end
      value[k]=round(Int,value[k]+half_nx)
      if ( value[k]==0 )
          value[k]+=nx
      end
    end
    return convert(Array{Int64,3},value) # in 2d ,i make the block big , so it is nx but here is nx/2
end
function rotate_3d_scalar(d::Cube,anglex::Number,angley::Number,anglez::Number)
  nx,ny,nz=size(d);
  Rn_x,Rn_y,Rn_z=New_Map(d,anglex,angley,anglez);
  Rn_x=pointer_function(Rn_x,nx);
  Rn_y=pointer_function(Rn_y,ny);
  Rn_z=pointer_function(Rn_z,nz);
  gc();
  d_rotate=dim_sum_scalar(d,Rn_x,Rn_y,Rn_z);
  return d_rotate
end
function dim_sum_scalar(d::Cube,Rn_x::Cube,Rn_y::Cube,Rn_z::Cube)
  nx,ny,nz=size(Rn_x);
  rn_x=reshape(Rn_x,1,length(Rn_x));
  rn_y=reshape(Rn_y,1,length(Rn_x));
  rn_z=reshape(Rn_z,1,length(Rn_x));
  BIG_INDEX=sub2ind(size(Rn_x),rn_x[1:end],rn_y[1:end],rn_z[1:end]);
  d_rotate  =reshape(d[BIG_INDEX],nx,ny,nz);
  cF64sum(d,nx,ny,nz) = reshape(sum(convert(Array{Float64,3},d),1),(nx,ny));
  d_rotate;
end
function rotate_3d_vector(vec_x::Cube,vec_y::Cube,vec_z::Cube,anglex::Number,angley::Number,anglez::Number)
  nx,ny,nz=size(vec_x);
  Rn_x,Rn_y,Rn_z=New_Map(vec_x,anglex,angley,anglez);
  Rn_x=pointer_function(Rn_x,nx);
  Rn_y=pointer_function(Rn_y,ny);
  Rn_z=pointer_function(Rn_z,nz);
  R_vx,R_vy,R_vz=Vector_New_Map(vec_x,vec_y,vec_z,
                                anglex,angley,anglez);
  gc();
  New_Vx,New_Vy,New_Vz=dim_sum_vector(R_vx,R_vy,R_vz,Rn_x,Rn_y,Rn_z);
  return New_Vx,New_Vy,New_Vz
end
function dim_sum_vector(iv::Cube,jv::Cube,kv::Cube,Rn_x::Cube,Rn_y::Cube,Rn_z::Cube)
  nx,ny,nz=size(Rn_x);
  rn_x=reshape(Rn_x,1,length(Rn_x));
  rn_y=reshape(Rn_y,1,length(Rn_x));
  rn_z=reshape(Rn_z,1,length(Rn_x));
  BIG_INDEX=sub2ind(size(Rn_x),rn_x[1:end],rn_y[1:end],rn_z[1:end]);
  cF64sum(d,nx,ny,nz) = reshape(sum(convert(Array{Float64,3},d),1),(nx,ny));
  iv_r=reshape(iv[BIG_INDEX],nx,ny,nz);
  jv_r=reshape(jv[BIG_INDEX],nx,ny,nz);
  kv_r=reshape(kv[BIG_INDEX],nx,ny,nz);
  return iv_r,jv_r,kv_r;
end
function rotate_3d_vector(Vec::Dict,anglex::Number,angley::Number,anglez::Number)
  nx,ny,nz=size(Vec["d"]);
  rotate_Cube=Dict("d"=>copy(Vec["d"]),
                "iv"=>zeros(nx,ny,nz),
                "jv"=>zeros(nx,ny,nz),
                "kv"=>zeros(nx,ny,nz),
                "ib"=>zeros(nx,ny,nz),
                "jb"=>zeros(nx,ny,nz),
                "kb"=>zeros(nx,ny,nz),)
  return_Cube=copy(rotate_Cube)
  Rn_x,Rn_y,Rn_z=New_Map(Vec["d"],anglex,angley,anglez);
  Rn_x=pointer_function(Rn_x,nx);
  Rn_y=pointer_function(Rn_y,ny);
  Rn_z=pointer_function(Rn_z,nz);
  rotate_Cube["iv"],rotate_Cube["jv"],rotate_Cube["kv"]=Vector_New_Map(Vec["iv"],Vec["jv"],Vec["kv"],
                                                                        anglex,angley,anglez);
  gc();
  rotate_Cube["ib"],rotate_Cube["jb"],rotate_Cube["kb"]=Vector_New_Map(Vec["ib"],Vec["jb"],Vec["kb"],
                                                                        anglex,angley,anglez);
  gc();
  for k in 1:nz, j in 1:ny ,i in 1:nx
    @inbounds return_Cube["d"][i,j,k] =rotate_Cube["d"][Rn_x[i,j,k],Rn_y[i,j,k],Rn_z[i,j,k]]
    @inbounds return_Cube["iv"][i,j,k]=rotate_Cube["iv"][Rn_x[i,j,k],Rn_y[i,j,k],Rn_z[i,j,k]]
    @inbounds return_Cube["jv"][i,j,k]=rotate_Cube["jv"][Rn_x[i,j,k],Rn_y[i,j,k],Rn_z[i,j,k]]
    @inbounds return_Cube["kv"][i,j,k]=rotate_Cube["kv"][Rn_x[i,j,k],Rn_y[i,j,k],Rn_z[i,j,k]] 
    @inbounds return_Cube["ib"][i,j,k]=rotate_Cube["ib"][Rn_x[i,j,k],Rn_y[i,j,k],Rn_z[i,j,k]]
    @inbounds return_Cube["jb"][i,j,k]=rotate_Cube["jb"][Rn_x[i,j,k],Rn_y[i,j,k],Rn_z[i,j,k]] 
    @inbounds return_Cube["kb"][i,j,k]=rotate_Cube["kb"][Rn_x[i,j,k],Rn_y[i,j,k],Rn_z[i,j,k]]
  end
  return return_Cube
end
==#


  # # GreensFunction.jl
 
function crosscorr(u::Mat,v::Mat)
  # KH : crosscorr function using the convolution method
  #      used for **NON PERIODIC** correlation studies
  #      see Lazarian et. al (2018) for more detail
  nx,ny=size(v);
  Nx=2*nx+1;
  Ny=2*ny+1;
  G_r=zeros(Nx,Ny);
  G_r[1:Nx-1,1:Ny-1]=[u u;u u];
  G_k=fft(G_r);
  rho_r=zeros(Nx,Ny);
  rho_r[1:nx,1:ny].=v;
  rho_k=fft(rho_r);
  phi_k=G_k.*conj(rho_k);
  rho_r=0;
  rho_k=0;
  G_k=0;
  phi_r=ifft(phi_k);
  phi_k=0;
  phi_rr=real(phi_r[nx:-1:1,ny:-1:1])
  phi_r=0;
  return fftshift(phi_rr);
end

function autocorr(u::Mat)
  return crosscorr(u,u);
end

  # # ellipse.jl

function angle_back(x,y,I,angle)
  # KH : Computed the rotation angle for the frame
  #      Originally written by Dora Ho (CUHK)
    nx,ny=size(I)
    x_=x-nx/2
    y_=y-ny/2
    X_dot= x_.*cos(angle)+y_.*sin(angle)
    Y_dot= -x_.*sin(angle)+y_.*cos(angle)
    return round(Int,X_dot+nx/2),round(Int,Y_dot+ny/2)
end

function dydx(x,Data,ii,jj,neighborhood_map)
  # KH : Compute the derivative using the angle_back function
  #      Originally written by Dora Ho (CUHK) 
    x1,y1=angle_back(x,Data["y1"][ii],neighborhood_map,Data["angle"][ii])
    x2,y2=angle_back(x,Data["y2"][ii],neighborhood_map,Data["angle"][ii])
    x3,y3=angle_back(x,Data["y1"][jj],neighborhood_map,Data["angle"][jj])
    x4,y4=angle_back(x,Data["y2"][jj],neighborhood_map,Data["angle"][jj])
    vector1=[x2-x1,y2-y1]
    vector2=[x4-x3,y4-y3]
    return dot(vector1,vector2)
end

function ellipse_axis(Cf,pixel_distance)
  # KH : Main function in determining the axises of the ellipse
  #      used in determining the major/minor axis ratio in CFA studies
  #      This program is usable for both periodic and non-periodic maps
  #      which has been used in Lazarian et. al (2018)
    nx,ny=size(Cf);
    mdd=Cf[div(nx,2),div(ny,2)+pixel_distance] #take the value of n pixel_distance from the centre
    #mdd=magic_number*(maximum(Cf)+minimum(Cf))/2  #white ring 
    neighborhood_map=zeros(size(Cf))
    neighborhood_map[findall(Cf.>mdd)].=1
    x,y=div(size(neighborhood_map)[1],2),div(size(neighborhood_map)[2],2)
    # KH : The number of angles permitted is hardcoded.
    #      TODO: to allow a user-defined angle ranges and binning
    Data=Dict("angle"=>zeros(181),
            "length"=>zeros(181),
             "y1"=>zeros(181),
             "y2"=>zeros(181));
    for (index,angle_) in enumerate((0:1:180)/180*pi)
        Data["angle"][index]=angle_
        Rotate_Map=rotate_2d(neighborhood_map,angle_);
        index2=findall(Rotate_Map[x,:].==1)
        Data["length"][index]=maximum(index2)-minimum(index2)
        Data["y1"][index]=maximum(index2)
        Data["y2"][index]=minimum(index2)
    end
  anglexx=Data["angle"][Data["length"].==maximum(Data["length"])[1]][1];
    return maximum(Data["length"])[1],minimum(Data["length"])[1],anglexx
end

function circular_stat(θ::Array)
  #Note : 14/3/2020 DH: remove the correction facrtor of 2 
    θ    = θ[.~isnan.(θ)];
    cosθ = cos.(θ);
    sinθ = sin.(θ);
    C_p  = mean(cosθ);
    S_p  = mean(sinθ);
    R_p  = sqrt(sum(cosθ)^2+sum(sinθ)^2)/length(cosθ);
    V    = 1-R_p;
    if C_p < 0
        return (atan(S_p/C_p)+pi), V #RM 2
    elseif S_p <0
        return (atan(S_p/C_p)+2*pi), V #RM 2
    else 
        return atan(S_p/C_p), V #RM 2
    end
end

function ScratterRotation(θ::Number,x::Array,y::Array)
    R(x) = [[cos(x) -sin(x)];[sin(x) cos(x)]]
    NRx,NRy = zeros(size(x)),zeros(size(y));
    for k = 1:size(x)[1]
        NRx[k],NRy[k] =R(θ)*[x[k],y[k]]
    end
    NRx,NRy
end

function EllispeFitting(x::Array,y::Array,rx::Number,ry::Number)
    Rx,Ry = x.-rx,y.-ry 
    N     = 180;
    θs    = collect(-pi/2:pi/N:pi/2);
    Δx    = zeros(size(θs));
    for (k,θ) in enumerate(θs)
        Nrx,Nry = ScratterRotation(θ,Rx,Ry)
        Δx[k]   = maximum(Nrx).-minimum(Nrx) #project the length into x-axis
    end
    Δx_Max,Δx_Min  = maximum(Δx),minimum(Δx);
    θ_k = findall(Δx.==Δx_Max)[1];
    Δx_Max,Δx_Min,θs[θ_k] #return the best estimisted angle,length
end

function ellipse_axis(Cf;level=15,Nsample=3,st_pt=2)
    ϕs   = zeros(Nsample)
    figplot = figure();
    A = contour(Cf,level=level);
    L   = zeros(2);
    rx,ry = size(Cf)./2;
    for i = 1:Nsample
        Ellipse = A.allsegs[end-i-st_pt][1];
        x,y = Ellipse[:,1],Ellipse[:,2];
        L[1],L[2],ϕs[i] = EllispeFitting(x,y,rx,ry);
    end
    close(figplot)
    θ = circular_stat(ϕs*2)[1]
    return L[1],L[2],θ
end

#= expirmental function, not accurate in detecting angle of the ellipse
function fitEllipse(x::Array,y::Array)
    D = hcat(x.*x, x.*y, y.*y, x, y, ones(size(x)));
    S = D'*D; #np.dot(D.T,D)
    C = zeros((6,6));
    C[1,3] = C[3,1] = 2;
    C[2,2] = -1;
    E, V =  eigvals(inv(S)'*C),eigvecs(inv(S)'*C)
    n    = argmax(abs.(E));
    a    = V[:,n]
    a
end

function ellipse_center(a::Array)
    b,c,d,f,g,a = a[2]/2, a[3], a[4]/2, a[5]/2, a[6], a[1]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return x0,y0
end

function ellipse_angle_of_rotation(a::Array)
    # This alg may need to have a correction of 2
    b,c,d,f,g,a = a[2]/2, a[3], a[4]/2, a[5]/2, a[6], a[1];
    if b == 0
        if a > c
            return 0
        else
            return pi/2
        end
    else
        if a > c
            return atan(2*b/(a-c))/2
        else
            return pi/2 + atan(2*b/(a-c))/2
        end
    end
end

function ellipse_axis_length(a::Array)
    b,c,d,f,g,a = a[2]/2, a[3], a[4]/2, a[5]/2, a[6], a[1]
    up    = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1 = (b*b-a*c)*( (c-a)*sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2 = (b*b-a*c)*( (a-c)*sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1  = sqrt(up/down1)
    res2  = sqrt(up/down2)
    return [res1, res2]
end

function ellipse_axis(Cf;level=30,Nsample=5,st_pt=2)
    figplot = figure();
    A = contour(Cf,level=level);
    phi = zeros(Nsample);
    L   = zeros(2);
    for i = 1:Nsample
        Ellipse = A.allsegs[end-i-st_pt][1];
        x,y = Ellipse[:,1],Ellipse[:,2];
        a   = fitEllipse(x,y);
        Cen = ellipse_center(a);
        phi[i] = ellipse_angle_of_rotation(a);
        L = ellipse_axis_length(a);
    end
    MeanPhi   = circular_stat(phi*2)[1]; #Correction Factor 2
    MinL,MaxL = minimum(L),maximum(L)
    close(figplot)
    return MaxL,MinL,MeanPhi
end
=#
function sban_cfa(C,dn)
  # KH : The main function to perform sub-block CFA
  #      Originally written by Dora Ho (CUHK)
    nx,ny=size(C)
    Ca=zeros(div(nx,dn),div(ny,dn));
  Cr=zeros(div(nx,dn),div(ny,dn));
  for j in 1:div(ny,dn),i in 1:div(nx,dn)
    is=(i-1)*dn+1;
    ie=i*dn;
    js=(j-1)*dn+1;
    je=j*dn;
    Cx=C[is:ie,js:je];
    Cf=autocorr(Cx);
    a,b,c=ellipse_axis(Cf,5);
    Ca[i,j]=c;
    Cr[i,j]=a./b;
  end
  return Ca,Cr
end


end # module LazCFA
