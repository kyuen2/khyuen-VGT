module LazLIC
using PyCall,LazCore,LazType,Interpolations
using Base.Threads

# From Interpolations.CubicSplineInterpolation
sir=CubicSplineInterpolation

  ##############################################################################
  #
  # Copyright (c) 2016,2020
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
   KH: This is v1.0+ only
  ==#

export advance,licmap,licmap_final

function advance(vx,vy,x,y,fx,fy,w,h)
 tx=0;ty=0;
 if (vx>=0)
  tx=(1-fx)/vx;
 else
  tx=-fx/vx;
 end
 if(vy>=0)
  ty=(1-fy)/vy;
 else
  ty=-fy/vy;
 end
 if (tx<ty)
  if (vx>0)
   x+=1; 
   fx=0;
  else
   x-=1;
   fx=1;
  end
  fy+=tx*vy;
 else
  if (vy>=0)
   y+=1;
   fy=0;
  else
   y-=1;
   fy=1;
  end
  fx+=ty*vx;
 end
 if (x>w)
  x=w;
 end
 if (x<1)
  x=1;
 end
 if (y<1)
  y=1
 end
 if (y>h)
  y=h
 end
 return x,y,fx,fy
end



function licmap(u::Mat,v::Mat,texture::Mat,kernel::Vec)
 # u is sine
 # v is cosine
 ny,nx=size(u);
 klen=length(kernel);
 result=zeros(ny,nx);
 for i in 1:ny
  for j in 1:nx
  x=j;y=i;fx=0.5;fy=0.5
  k=div(klen,2);
  result[i,j]+=kernel[k]*texture[y,x]
  while (k<klen)
   x,y,fx,fy=advance(u[y,x],v[y,x],x,y,fx,fy,nx,ny);
   k+=1;
   result[i,j]+=kernel[k]*texture[y,x]
  end
  x=j;y=i;fx=0.5;fy=0.5
  k=div(klen,2);
  while (k>1)
   x,y,fx,fy=advance(-u[y,x],-v[y,x],x,y,fx,fy,nx,ny);
   k-=1;
   result[i,j]+=kernel[k]*texture[y,x]
  end
 end;end
 return result
end

function licmap_final(angle::Mat,texture::Mat,bit::Mat,klength)
 # KH: Your texture map will judge your interpolation level
 # angle : block-avereged angle
 # texture : the background map, suggesting using random noise map with significantly larger resolution
 ca=cos.(angle);
 sa=sin.(angle);
 ca[bit.==0].=0;
 sa[bit.==0].=0;
 mx,my=size(ca);
 nx,ny=size(texture);
 # The following lines are from experience    
 kernel=zeros(klength)
 kernel=sin.((Array(1:klength).-1)./(klength-1).*pi);
 #==
 KH: The problem lies in the python package. The Julia "licmap" function works very well
 Therefore we are going to fix the interpolation using Julia's native package using a wrapper trick
 ==#
 fc=sir((1:mx,1:my),ca);
 fs=sir((1:mx,1:my),sa);
 cca=fc(1:(mx-1)/(nx-1):mx,1:(my-1)/(ny-1):my);
 ssa=fs(1:(mx-1)/(nx-1):mx,1:(my-1)/(ny-1):my);
 result=licmap(ssa,cca,texture,kernel);
 return result
end

end #module 