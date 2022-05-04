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
Changelog:
    - Light speed compilation
"""
module LazCyvecd

using LazType

export cgrad,cdiv,pcurl,ccurl
export cgrad2d

function cgrad(v::Cube,dx)
    vdx = zeros(size(v));
    vdy = zeros(size(v));
    vdz = zeros(size(v));
    nx,ny,nz = size(v);
    for i=1:nx,j=1:ny,k=1:nz
        ip = mod(i,nx)+1;
        jp = mod(j,ny)+1;
        kp = mod(k,nz)+1;
        vdx[i,j,k] = (v[ip,j,k]-v[i,j,k])/dx;
        vdy[i,j,k] = (v[i,jp,k]-v[i,j,k])/dx;
        vdz[i,j,k] = (v[i,j,kp]-v[i,j,k])/dx;
    end
    return (vdx,vdy,vdz)
end

function cgrad2d(v::Mat,dx)
    vdx = zeros(size(v));
    vdy = zeros(size(v));
    nx,ny = size(v);
    for i=1:nx,j=1:ny
        ip = mod(i,nx)+1;
        jp = mod(j,ny)+1;
        vdx[i,j] = (v[ip,j]-v[i,j])/dx;
        vdy[i,j] = (v[i,jp]-v[i,j])/dx;
    end
    return (vdx,vdy)
end

function cdiv(vx::Cube,vy::Cube,vz::Cube,dx)
    vdiv = zeros(size(vx));
    nx,ny,nz = size(vx);
    for i=1:nx,j=1:ny,k=1:nz
        ip = mod(i,nx)+1;
        jp = mod(j,ny)+1;
        kp = mod(k,nz)+1;
        vdiv[i,j,k] = (vx[ip,j,k]-vx[i,j,k] + vy[i,jp,k]-vy[i,j,k] + vz[i,j,kp]-vz[i,j,k]) / dx;
    end
    return vdiv
end

function pcurl(ax::Cube,ay::Cube,az::Cube,bx::Cube,by::Cube,bz::Cube)
  # cx = zeros(size(ax));
  # cy = zeros(size(ax));
  # cz = zeros(size(ax));
  cx = ay.*bz - az.*by;
  cy = az.*bx - ax.*bz;
  cz = ax.*by - ay.*bx;
  return (cx,cy,cz)
end

function ccurl(vx::Cube,vy::Cube,vz::Cube,dx)
  vdx = zeros(size(vx));
  vdy = zeros(size(vx));
  vdz = zeros(size(vx));
  nx,ny,nz = size(vx);
  for i=1:nx,j=1:ny,k=1:nz
    ip = mod(i,nx)+1;
    jp = mod(j,ny)+1;
    kp = mod(k,nz)+1;
    vdx[i,j,k] = (vy[i,j,kp]-vy[i,j,k]-vz[i,jp,k]+vz[i,j,k])/dx;
    vdy[i,j,k] = (vz[ip,j,k]-vz[i,j,k]-vx[i,j,kp]+vx[i,j,k])/dx;
    vdz[i,j,k] = (vx[i,jp,k]-vx[i,j,k]-vy[ip,j,k]+vy[i,j,k])/dx;
  end
  return (vdx,vdy,vdz)
end

end
