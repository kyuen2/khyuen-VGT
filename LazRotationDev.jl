module LazRotationDev
using PyCall, Images, HDF5, Statistics, StatsBase;
using LazCore, LazType


##############################################################################
#
# Copyright (c) 2018
# Ka Wai Ho, Ka Ho Yuen and Alex Lazarian
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
#                                    Code for                                 # 
#                                                                             #
#                              Sclar & Vector Rotation                        #
#                                                                             #
###############################################################################
#   
#
#       Version     : v1.0.2(10/6/2020)
#       Author      : KW HO , KH Yuen@ Lazarian Technology
#       Description : Rewrite Version for LazRotation, providing Sclar & Vector Rotation 
#                     Change :
#                      1. Improvement of code strcuture with more brevity
#                      2. Improvement of performance and reduce the memory allocations
#       Note        : Performance Test for both scaler and vector between two vision for 480^3 simulation
#                      Old :
#                         Scalar Rotation 23.242066 seconds (1.51 k allocations: 8.446 GiB, 4.75% gc time)
#                         Vector Rotation 41.843450 seconds (1.55 k allocations: 15.038 GiB, 4.17% gc time)
#                      New :
#                         Scalar Rotation 19.889152 seconds (1.48 k allocations: 3.502 GiB, 3.88% gc time)
#                         Vector Rotation 23.268467 seconds (1.49 k allocations: 3.914 GiB, 4.83% gc time)
#       Capability  : Work for julia version 1.2.0
#       Cuation     : Second developer version, bug may exist
#
###############################################################################


#export the Main Rotation function for scaler and vecotr

export rotate_3d_scalar,rotate_3d_vector

function sub2ind(d::Cube,vi::Vec,vj::Vec,vk::Vec)
  nx,ny,nz=size(d)
  nl=minimum([length(vi),length(vj),length(vk)])
  s2i = LinearIndices(size(d))
  s2ii=zeros(Int,nl)
  @inbounds @simd for i in 1:nl
      s2ii[i] = round(Int,s2i[vi[i],vj[i],vk[i]])
    end
  return s2ii
end

f16(A::Cube)   = convert(Array{Float16,3},A);
f16(A::Number) = convert(Float16,A);

function meshgrid(nx::Number,ny::Number,nz::Number)
  X = zeros(Float16,nx,ny,nz);
  Y = zeros(Float16,nx,ny,nz);
  Z = zeros(Float16,nx,ny,nz);
  for k = 1:nz
    for j = 1:ny
      @inbounds @simd for i = 1:nx
        X[i,j,k] = i - nx/2 - 0.5;
        Y[i,j,k] = j - ny/2 - 0.5;
        Z[i,j,k] = k - nz/2 - 0.5;
      end
    end
  end
  return X,Y,Z
end

function axis_rotation(jv::Cube,kv::Cube,ϴ::Number)
  cosϴ = cos(ϴ);
  sinϴ = sin(ϴ);
  @inbounds @simd  for k in eachindex(jv)
    jv_cac = copy(jv[k]);
    kv_cac = copy(kv[k]);
    jv[k] = jv_cac*cosϴ - kv_cac*sinϴ;
    kv[k] = kv_cac*cosϴ + jv_cac*sinϴ;
  end
end

function rodrigue_rotation(iv::Cube,jv::Cube,kv::Cube,ϴ::Number,rotation_axis_number::Int)
 # Source
 # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
 if (rotation_axis_number==1)
  #Rotating the X-axis
  #jv'=jv.*cos.(ϴ).-kv.*sin.(ϴ)
  #kv'=kv.*cos.(ϴ).+jv.*sin.(ϴ)
    axis_rotation(jv,kv,ϴ);
 elseif (rotation_axis_number==2)
  #Rotating the Y-axis
  #kv'=kv.*cos.(ϴ).-iv.*sin.(ϴ)
  #iv'=iv.*cos.(ϴ).+kv.*sin.(ϴ)
    axis_rotation(kv,iv,ϴ);
 elseif (rotation_axis_number==3)
  #Rotating the Z-axis
  #ivp=iv.*cos.(ϴ).-jv.*sin.(ϴ)
  #jvp=jv.*cos.(ϴ).+iv.*sin.(ϴ)
    axis_rotation(iv,jv,ϴ);
 else
  println("KH: WTF are you projecting?");
  return 0; 
 end
end

function pointer_function(Block::Cube,nx::Int)
  ΔX = nx/2;
  for k in eachindex(Block)
    if (Block[k]>ΔX)
      n=div(Block[k],nx);
      Block[k] -= (n+1)*nx;
    elseif -1*Block[k]>ΔX
          n=div(abs(Block[k]),nx)
          Block[k]+=(n+1)*nx
      end
      Block[k]=round(Int,Block[k]+ΔX+0.5)
    if ( Block[k]==0 )
      Block[k]+=nx
    end
    end
    # KH (Jun22,2019): No need to use Int64. Int16 is more than enough (indices never go over |2^15| ~ 32768)
    return convert(Array{Int16,3},Block) # in 2d ,i make the block big , so it is nx but here is nx/2
end

function CoordinationShift(RX::Cube,RY::Cube,RZ::Cube)
    nx,ny,nz = size(RX);
    RX = pointer_function(RX,nx);
    RY = pointer_function(RY,ny);
    RZ = pointer_function(RZ,nz);
    return RX,RY,RZ
end

function CoordinateMap(d::Cube,ϴ_x::Number,ϴ_y::Number,ϴ_z::Number)
  nx,ny,nz = size(d);
  X,Y,Z    = meshgrid(nx,ny,nz);
  rodrigue_rotation(X,Y,Z,ϴ_x,1);
  rodrigue_rotation(X,Y,Z,ϴ_y,2);
  rodrigue_rotation(X,Y,Z,ϴ_z,3);
  X,Y,Z    = CoordinationShift(X,Y,Z);
    return X,Y,Z
end

function VectorRotation(V_x::Cube,V_y::Cube,V_z::Cube,ϴ_x::Number,ϴ_y::Number,ϴ_z::Number);
    Vx,Vy,Vz = copy(V_x),copy(V_y),copy(V_z);
    rodrigue_rotation(Vx,Vy,Vz,ϴ_x,1);
    rodrigue_rotation(Vx,Vy,Vz,ϴ_y,2);
    rodrigue_rotation(Vx,Vy,Vz,ϴ_z,3);
    return Vx,Vy,Vz
end

function ScalarReCoordination(d::Cube,RX::Cube,RY::Cube,RZ::Cube)
  nx,ny,nz = size(d);
  RX = reshape(RX,1,length(RX));
  RY = reshape(RY,1,length(RY));
  RZ = reshape(RZ,1,length(RZ));
  _1D_INDEX_ = sub2ind(d,RX[1:end],RY[1:end],RZ[1:end]);
  Rd  =reshape(d[_1D_INDEX_],nx,ny,nz);
    return Rd
end

function VectorReCoordination(Vx::Cube,Vy::Cube,Vz::Cube,RX::Cube,RY::Cube,RZ::Cube)
  nx,ny,nz = size(Vx);
  RX = reshape(RX,1,length(RX));
  RY = reshape(RY,1,length(RY));
  RZ = reshape(RZ,1,length(RZ));
  _1D_INDEX_ = sub2ind(Vx,RX[1:end],RY[1:end],RZ[1:end]);
  RVx  = reshape(Vx[_1D_INDEX_],nx,ny,nz);
  RVy  = reshape(Vy[_1D_INDEX_],nx,ny,nz);
  RVz  = reshape(Vz[_1D_INDEX_],nx,ny,nz);
  return RVx,RVy,RVz
end

#1. Rotation for scalar
#Step 1. Create a coordinate Map for the center axis rotation
#Step 2. Relocate the scalar to the rotated coordinate map
function rotate_3d_scalar(d::Cube,ϴ_x::Number,ϴ_y::Number,ϴ_z::Number)
  RX,RY,RZ = CoordinateMap(d,ϴ_x,ϴ_y,ϴ_z);
  Rd       = ScalarReCoordination(d,RX,RY,RZ);
  return Rd
end

#2. Rotation for vector
#Step 1. Create a coordinate Map for the center axis rotation
#Step 2. Rotating the vector 
#Step 3. Relocate the vector to the rotated coordinate map
function rotate_3d_vector(Vx::Cube,Vy::Cube,Vz::Cube,ϴ_x::Number,ϴ_y::Number,ϴ_z::Number)
  RX ,RY ,RZ   = CoordinateMap(Vx,ϴ_x,ϴ_y,ϴ_z);
  RVx,RVy,RVz  = VectorRotation(Vx,Vy,Vz,-ϴ_x,-ϴ_y,-ϴ_z); 
  RVx,RVy,RVz  = VectorReCoordination(RVx,RVy,RVz,RX,RY,RZ);
  return RVx,RVy,RVz
end

end # module LazRotation
