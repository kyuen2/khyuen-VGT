module LazCh5
using HDF5,Statistics,LsqFit,PyCall,FFTW,StatsBase,Images
using LazCore,LazType,LazIO
using LazRHT # Filament moduile
using LazThermal,LazThermal_Kritsuk # Thermal broadening module
using LazDust # The core package for PvB
using LazGAC # Gradient aand Curvature
using Base.Threads # Parallelism

##############################################################################
#
# Copyright (c) 2020
# Ka Ho Yuen, Alex Lazarian and Dmitri Pogosyan
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


export spectral_filtering,spectrallines3D,spectrallines3D_k,spectrallines2D_k,spectrallines3D_kk
export sf2_bruteforce,threept_sf2_bruteforce,fourpt_sf2_bruteforce,threept_sf2_bruteforce_montecarlo,local_velocity_sf2_threept_montecarlo

function spectral_filtering(A::Mat,k_low::Number,k_high::Number)
 nx,ny=size(A);
 Af=fftshift(fft(A));
 for i in 1:nx, j in 1:ny
  idx=i-div(nx,2)-1;
  jdx=j-div(ny,2)-1;
  rr=round(Int,sqrt(idx^2+jdx^2));
  if ((rr<k_low) || (rr>k_high))
    Af[i,j]=0        
  end
 end
 Ap=abs.(ifft(ifftshift(Af)));
 return Ap
end

function spectrallines3D(A::Cube)
 nx,ny,nz=size(A);
 nl=round(Int,sqrt(nx^2+ny^2+nz^2));
 kn=zeros(nl);
 kl=zeros(nl);
 Ax2=fftshift((abs.(fft(A))).^2.0);
 for i in 1:nx, j in 1:ny, k in 1:nz
  idx=i-div(nx,2)-1;
  jdx=j-div(ny,2)-1;
  kdx=k-div(nz,2)-1;
  rr=round(Int,sqrt(idx^2+jdx^2+kdx^2));
  if ((rr<=nl) & (rr>0))
   kn[rr]+=Ax2[i,j,k];
   kl[rr]=rr;
  end
 end
 return kn,kl
end

line(x,p)=p[1].+x.*p[2];

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

function spectrallines3D_k(A::Cube)
 nx,ny,nz=size(A);
 nl=round(Int,sqrt(nx^2+ny^2+nz^2));
 kn=zeros(nl);
 kl=zeros(nl);
 AA=fftshift(A)
 for i in 1:nx, j in 1:ny, k in 1:nz
  idx=i-div(nx,2)-1;
  jdx=j-div(ny,2)-1;
  kdx=k-div(nz,2)-1;
  rr=round(Int,sqrt(idx^2+jdx^2+kdx^2));
  if ((rr<=nl) & (rr>0))
   kn[rr]+=AA[i,j,k];
   kl[rr]=rr;
  end
 end
 return kn,kl
end

function spectrallines2D_k(A::Mat)
 nx,ny=size(A);
 nl=round(Int,sqrt(nx^2+ny^2));
 kn=zeros(nl);
 kl=zeros(nl);
 AA=fftshift(A)
 for i in 1:nx, j in 1:ny
  idx=i-div(nx,2)-1;
  jdx=j-div(ny,2)-1;
  rr=round(Int,sqrt(idx^2+jdx^2));
  if ((rr<=nl) & (rr>0))
   kn[rr]+=AA[i,j];
   kl[rr]=rr;
  end
 end
 return kn,kl
end


function spectrallines3D_kk(A::Cube)
 nx,ny,nz=size(A);
 nl=round(Int,sqrt(nx^2+ny^2+nz^2));
 kn=zeros(nl);
 kl=zeros(nl);
 kll=zeros(nl);
 ky=zeros(nl);
 AA=fftshift(A)
 for i in 1:nx, j in 1:ny, k in 1:nz
  idx=i-div(nx,2)-1;
  jdx=j-div(ny,2)-1;
  kdx=k-div(nz,2)-1;
  r=sqrt(idx^2+jdx^2+kdx^2);
  rr=round(Int,r);
  if ((rr<=nl) & (rr>0))
   kn[rr]+=AA[i,j,k];
   kl[rr]+=r;
   kll[rr]=rr
   ky[rr]+=1;
  end
 end
 kl=kl./ky;
 kn[1:end-1] = (4*pi/3).*(kll[2:end].^3.0.-kll[1:end-1].^3.0)./ky[1:end-1].*kn[2:end]
 return kn,kl
end

function cf2(A::Mat)
    return real.(ifft(abs.(fft(A)).^2.0))./length(A);
end

function sf2_bruteforce(AA::Mat;nnx=100,nny=100)
    nx,ny=size(AA)
    AAA=zeros(typeof(AA[1]),nx,ny)
    NNN=zeros(Int,nx,ny)
@threads for i in 1:nx
          for j in 1:ny, ii in 1:nnx, jj in 1:nny
            ip=i+(ii-div(nx,2));
            jp=j+(jj-div(ny,2));
            im=i-(ii-div(nx,2));
            jm=j-(jj-div(ny,2));
            if ((ip<=nx) && (jp<=ny) && (ip>=1) && (jp>=1))# && (im<=nx) && (jm<=ny) && (im>=1) && (jm>=1))
                AAA[ii,jj]+=(AA[ip,jp]-AA[i,j])^2;
                NNN[ii,jj]+=1;
            end
        end
    end
    BBB=AAA./NNN
    BBB[NNN.==0].=0
    return BBB
end

function threept_sf2_bruteforce(AA::Mat;nnx=100,nny=100)
    nx,ny=size(AA)
    AAA=zeros(typeof(AA[1]),nx,ny)
    NNN=zeros(Int,nx,ny)
@threads for i in 1:nx
          for j in 1:ny, ii in 1:nnx, jj in 1:nny
            ip=i+(ii-div(nx,2));
            jp=j+(jj-div(ny,2));
            im=i-(ii-div(nx,2));
            jm=j-(jj-div(ny,2));
            if ((ip<=nx) && (jp<=ny) && (ip>=1) && (jp>=1) && (im<=nx) && (jm<=ny) && (im>=1) && (jm>=1))
                AAA[ii,jj]+=(AA[ip,jp]+AA[im,jm]-2*AA[i,j])^2/2;
                NNN[ii,jj]+=1;
            end
        end
    end
    BBB=AAA./NNN
    BBB[NNN.==0].=0
    return BBB
end

function fourpt_sf2_bruteforce(AA::Mat;nnx=100,nny=100)
    nx,ny=size(AA)
    AAA=zeros(typeof(AA[1]),nx,ny)
    NNN=zeros(Int,nx,ny)
@threads for i in 1:nx
          for j in 1:ny, ii in 1:nnx, jj in 1:nny
            ipp=i+2*(ii-div(nx,2));
            jpp=j+2*(jj-div(ny,2));
            ip=i+(ii-div(nx,2));
            jp=j+(jj-div(ny,2));
            im=i-(ii-div(nx,2));
            jm=j-(jj-div(ny,2));
            if ((ipp<=nx) && (jpp<=ny) && (ipp>=1) && (jpp>=1) &&(ip<=nx) && (jp<=ny) && (ip>=1) && (jp>=1) && (im<=nx) && (jm<=ny) && (im>=1) && (jm>=1))
                AAA[ii,jj]+=(AA[ipp,jpp]-3*AA[ip,jp]-AA[im,jm]+3*AA[i,j])^2/10;
                NNN[ii,jj]+=1;
            end
        end
    end
    BBB=AAA./NNN
    BBB[NNN.==0].=0
    return BBB
end

function threept_sf2_bruteforce_montecarlo(AA::Mat;nnx=100,nny=100,fac=0.001)
    nx,ny=size(AA)
    AAA=zeros(typeof(AA[1]),nx,ny)
    NNN=zeros(Int,nx,ny)
@threads for ii in 1:nnx
          for jj in 1:nny
           for idx in 1:round(Int,nx*ny*fac)
            i = rand(1:nx)
            j = rand(1:ny)
            ip=i+(ii-div(nx,2));
            jp=j+(jj-div(ny,2));
            im=i-(ii-div(nx,2));
            jm=j-(jj-div(ny,2));
            if ((ip<=nx) && (jp<=ny) && (ip>=1) && (jp>=1) && (im<=nx) && (jm<=ny) && (im>=1) && (jm>=1))
                AAA[ii,jj]+=(AA[ip,jp]+AA[im,jm]-2*AA[i,j])^2/2;
                NNN[ii,jj]+=1;
            end
           end
          end
         end
    BBB=AAA./NNN
    BBB[NNN.==0].=0
    return BBB
end

function threept_sf(A::Mat;periodic=true,nnx=100,nny=100)
    nx,ny=size(A);
    if (periodic)
        Acf=cf2(A);
        A2=mean(A.^2.0);
        Af=zeros(nx,ny);
        for i in 1:nx, j in 1:ny
            i2=mod(2*i-2,nx)+1;
            j2=mod(2*j-2,ny)+1;
            Af[i,j]=6.0.*A2.-8.0.*Acf[i,j].+2.0.*Acf[i2,j2]
        end
    else
        Af = threept_sf2_bruteforce(A;nnx=nnx,nny=nny);
    end
    return Af
end

function fourpt_sf(A::Mat;periodic=true,nnx=100,nny=100)
    nx,ny=size(A);
    if (periodic)
        Acf=cf2(A);
        A2=mean(A.^2.0);
        Af=zeros(nx,ny);
        for i in 1:nx, j in 1:ny
            i2=mod(2*(i-1),nx)+1;
            j2=mod(2*(j-1),ny)+1;
            i3=mod(3*(i-3),nx)+1;
            j3=mod(3*(j-3),ny)+1;
            Af[i,j]=20*A2-30*Acf[i,j]+12*Acf[i2,j2]-2*Acf[i3,j3]
        end
    else
        Af = fourpt_sf2_bruteforce(A,nnx=nnx,nny=nny)
    end
    return Af
end

function local_velocity_sf2_threept_montecarlo(iv::Cube,jv::Cube,kv::Cube,ib::Cube,jb::Cube,kb::Cube;nnx=100,nny=100,nnz=100,frac=0.001)
    nx,ny,nz=size(iv)
    AAA=zeros(typeof(iv[1]),nx,ny,nz)
    NNN=zeros(Int,nx,ny,nz)
@threads for ii in 1:nnx
          for jj in 1:nny, kk in 1:nnz
          for idx in 1:round(Int,nx*ny*nz*frac)
            i = rand(1:nx)
            j = rand(1:ny)
            k = rand(1:nz)
            ip=i+(ii-div(nx,2));
            jp=j+(jj-div(ny,2));
            kp=k+(kk-div(nz,2));
            im=i-(ii-div(nx,2));
            jm=j-(jj-div(ny,2));
            km=k-(kk-div(nz,2));
            bb=sqrt(ib[i,j,k]^2+jb[i,j,k]^2+kb[i,j,k]^2);
            if ((ip<=nx) && (jp<=ny) && (kp<=nz) && (ip>=1) && (jp>=1) && (kp>=1) && (im<=nx) && (jm<=ny) && (km<=nz) && (im>=1) && (jm>=1) && (km>=1))
                dvx=(iv[ip,jp,kp]+iv[im,jm,km]-2*iv[i,j,k]).*ib[i,j,k]./bb./2.0;
                dvy=(jv[ip,jp,kp]+jv[im,jm,km]-2*jv[i,j,k]).*jb[i,j,k]./bb./2.0;
                dvz=(kv[ip,jp,kp]+kv[im,jm,km]-2*kv[i,j,k]).*kb[i,j,k]./bb./2.0;
                AAA[ii,jj,kk]+=(dvx.+dvy.+dvz).^2;
                NNN[ii,jj,kk]+=1;
            end
        end
        end
    end
    BBB=AAA./NNN
    BBB[NNN.==0].=0
    return BBB
end


end # module LazCh5