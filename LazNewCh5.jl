module LazNewCh5
using HDF5,PyPlot,Statistics,LsqFit # Standard packages
using PyCall,FFTW,StatsBase,Images  # Standard packages
using FITSIO                        # Standard packages

# LazTech-VGT main codes
using LazCore,LazType,LazIO         # LazTech Basic Modules
using LazRHT,LazRHT_investigation   # Filament modules
using LazThermal,LazThermal_Kritsuk # Thermal broadening modules
using LazDust                       # The core package for MGT
using LazGAC                        # Gradient aand Curvature code
using LazCFA                        # CFA-related studies
using LazCh5                        # ch5-related studies
using LazMode                       # Mode decomposition 
using LazIN                         # Ion-neutral 
using LazRotation                   # Rotation module
using LazSyntheticCube              # Synthetic cube synthesis
using LazTurbStat                   # Dora's Anisotropy code
using LazAMW                        # Smoothing
using LazLIC                        # LIC algorithm
using LazTsallis                    # Tsallis Statistics

# LazTech-VGT-new main codes
using LazDDA                        # Dmitri Decomposition Algorithm
using LazFilament                   # Filamentary formation algorithm
using LazMultiGaussian              # Wing Channel ch6: Multigaussian fitting
using LazNewCore

# Accelerations
using Base.Threads                  # CPU Parallelism
#using ArrayFire                     # GPU Parallelism
using Profile                       # Profiling
# Interpolations
using Interpolations
# From Interpolations.CubicSplineInterpolation
sir=CubicSplineInterpolation

##############################################################################
#
# Copyright (c) 2021
# Ka Ho Yuen &  Alex Lazarian
# All Rights Reserved.
# 
# Jan 11 2022: Emergency Patch: the fftshift bug, causing <B_P> ≢ 0
# fixed via compensating the fftshift deficiency
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



export sr,si,imshowx,colorbarx,tight,noticks,rs,smooth,hread,hread_veco,hread_vec
export read_binary,read_binary64,read_binary64t,read_chepurnov
export cdot3,n3,outerproduct,identitymatrix,ccross3,ucross3,T_tensor
export PCA_decomposition,getMa,PCA_decomposition_cone_removed
export PCASF_decomposition,PCASF_decomposition_v,getmode_z,get_Bmode_z

# utility functions
function sr(a;digits=2)
    return string(round(a,digits=2))
end

function si(a)
    return string(round(Int,a))
end

#==
function imshowx(I;a=3,b=0,noleg=true)
    ima=imshow(I,vmin=mean(I)-b*std(I),vmax=mean(I)+a*std(I),cmap="RdGy_r")
    if noleg
        ax1.set_xticklabels([]);ax1.set_yticklabels([]);
    end
    return ima
end

function colorbarx(ima,PP;b=3,a=3,vlabel=L"$p_{v,HI}$")
    cbar=colorbar(ima,orientation="horizontal",fraction=0.046, pad=0.00,ticks=mean(PP).+std(PP).*Array(-b:a))
    cbar.ax.set_xticklabels([(c!=0 ? (c>0 ? L"$\langle $"*vlabel*L"$\rangle +$"*si(c)*L"$std($"*vlabel*L"$)$"
                                : L"$\langle $"*vlabel*L"$\rangle -$"*si(abs(c))*L"$std($"*vlabel*L"$)$"   
                                ) : L"$\langle$ "*vlabel*L"$ \rangle$") for c in -b:a],
                              rotation=270)
    return cbar
end 

tight()=subplots_adjust(wspace=0,hspace=0);

function noticks()
    ax1.set_xticklabels([]);ax1.set_yticklabels([]);
end
==#
function rs(a;dims=3)
    return reshape(sum(a,dims=dims),size(a)[1:2])
end

function smooth(a;b=2)
    return imfilter(a,Kernel.gaussian(b))
end

function hread(f,db)
    return convert(Array{Float32,3},read(f,db))
end

function read_chepurnov(db;Nsize=128,pos=5)
    # db is the x-filename
    ib=Array{Float64}(undef,Nsize,Nsize,Nsize);
    jb=Array{Float64}(undef,Nsize,Nsize,Nsize);
    kb=Array{Float64}(undef,Nsize,Nsize,Nsize);
    dbl=db[1:end-pos-1]*"?"*db[end-pos+1:end]
    bdat=sort(glob.glob(dbl))
    read!(bdat[1],ib);
    read!(bdat[2],jb);
    read!(bdat[3],kb);
    return ib,jb,kb
end

function read_binary(db,Nsize,Nnum)
    d=Array{Float32}(undef,Nsize,Nsize,Nsize,Nnum);
    read!(db,d);
    return d
end

function read_binary64(db,Nsize)
    d=Array{Float64}(undef,Nsize,Nsize,Nsize);
    read!(db,d);
    return d
end

function read_binary64t(db,Nsize,Nnum)
    d=Array{Float64}(undef,Nsize,Nsize,Nsize,Nnum);
    read!(db,d);
    return d
end

function hread(f,n)
    return convert(Array{Float32,3},read(f,n))
end

function hread_veco(f,n)
    ny="j"*n[2:end];
    nz="k"*n[2:end];
    return convert(Array{Float32,3},read(f,n)),convert(Array{Float32,3},read(f,ny)),convert(Array{Float32,3},read(f,nz))
end

function hread_vec(f,n)
    ny="by"*n[3:end];
    nz="bz"*n[3:end];
    return convert(Array{Float32,3},read(f,n)),convert(Array{Float32,3},read(f,ny)),convert(Array{Float32,3},read(f,nz))
end


# Leakage

## Essential functions

function cdot3(k,λ)
    return sum(k.*λ)
end

function m3(k)
    return cdot3(k,k)
end

function n3(k)
    return k./sqrt(cdot3(k,k))
end

function outerproduct(u,v)
    return u*transpose(v)
end

function identitymatrix(k)
    A=zeros(length(k),length(k))
    for i in 1:length(k) 
        A[i,i]=1;
    end;
    return A
end

function ccross3(k,λ)
    o=zeros(size(k))
    for i in 1:length(o)
        ip=mod1(i+1,length(o))
        ipp=mod1(i+2,length(o))
        o[i]=k[ip]*λ[ipp]-λ[ip]*k[ipp]
    end
    return o 
end

# Here S and C are the E and F in LP12.
function T_tensor(k::Vec,λ::Vec;mode='p')
    # input 
    ## k: The k_vector, will normalize below
    ## lambda: The lambda vctor, will normalize below
    khat=n3(k)
    λhat=n3(λ)
    kdotλ=cdot3(khat,λhat)
    Id=identitymatrix(khat)
    T_ij=zeros(3,3)
    if (mode=='p')
        # k x k
        T_ij=outerproduct(khat,khat)
    elseif (mode=='s')
        # I - k x k
        T_ij=Id.-outerproduct(khat,khat)   
    elseif (mode=='c')
        # ....
        T_ij=(kdotλ.^2.0.*outerproduct(khat,khat).+outerproduct(λhat,λhat).-kdotλ.*(outerproduct(khat,λhat).+outerproduct(λhat,khat)))./(1.0.-kdotλ.^2.0);
    elseif (mode=='a')
        # I - k x k - ....
        T_ij=Id.-outerproduct(khat,khat).-(kdotλ.^2.0.*outerproduct(khat,khat).+outerproduct(λhat,λhat).-kdotλ.*(outerproduct(khat,λhat).+outerproduct(λhat,khat)))./(1.0.-kdotλ.^2.0);
    elseif (mode=='n')
        kxλ=n3(ccross3(khat,λhat))
        T_ij= outerproduct(kxλ,kxλ)
    end
    # remove the NaN case
    if (abs.(kdotλ)>=1)
        T_ij=zeros(3,3)
    end
    return T_ij
end

function getMa(ib,jb,kb)
    return sqrt(var(ib)+var(jb)+var(kb))/sqrt(mean(ib)^2+mean(jb)^2+mean(kb)^2)
end

function ucross3(x::Vec,y::Vec)
    # Normalization
    xamp=sqrt(sum(x.^2.0));
    yamp=sqrt(sum(y.^2.0));
    x = x./xamp;
    y = y./yamp;

    # cross product
    xxy=zeros(length(x));

    for i in 1:length(x)
         # logic for cross product
         # for i-component
         # x x y (i) = x(i+1) * y(i+2) - x(i+2)*y(i+1)
         # mod1(x,y)= mod(x-1,y)+1
         ip1=mod1(i+1,length(x))
         ip2=mod1(i+2,length(x))
         xxy[i] = x[ip1]*y[ip2]-x[ip2]*y[ip1]
    end
    # just to make sure it's normalized
    xxyamp=sqrt(sum(xxy.^2.0))
    xxy = xxy ./xxyamp
    return xxy
end
#==
function PCA_decomposition(ib,jb,kb;mode="C")
    ib_f=fftshift(fft(ib))
    jb_f=fftshift(fft(jb))
    kb_f=fftshift(fft(kb))
    ibf_o = zeros(typeof(1.0.+im),size(ib_f));
    jbf_o = zeros(typeof(1.0.+im),size(jb_f));
    kbf_o = zeros(typeof(1.0.+im),size(kb_f));
    nx,ny,nz=size(ib);
    
    mib=mean(ib);
    mjb=mean(jb);
    mkb=mean(kb);
    Bhat = [mib,mjb,mkb]
    for i in 1:nx, j in 1:ny, k in 1:nz
        # get the kk vector
        kx = i - div(nx,2);
        ky = j - div(ny,2);
        kz = k - div(nz,2);
        kk = [kx,ky,kz]

        if (sum(kk.^2.0)>0)
            if (mode=="A")
                zeta_o = ucross3(kk,Bhat)
            elseif (mode=="C")
                zeta_A = ucross3(kk,Bhat)
                zeta_o = ucross3(kk,zeta_A)
            elseif (mode=="P")
                #zeta_A = ucross3(kk,Bhat)
                #zeta_C = ucross3(kk,zeta_A)
                #zeta_o = ucross3(zeta_A,zeta_C)
                zeta_o = kk./sqrt.(sum(kk.^2.0))
            end
            # assign a vector at (i,j,k)
            bf = [ib_f[i,j,k],jb_f[i,j,k],kb_f[i,j,k]]
            # do a dot product
            bfdotC = sum(bf.*zeta_o)
            ibf_o[i,j,k] = bfdotC.*zeta_o[1]
            jbf_o[i,j,k] = bfdotC.*zeta_o[2]
            kbf_o[i,j,k] = bfdotC.*zeta_o[3]
        end
    end
    ib_o = real(ifft(ifftshift(ibf_o)));
    jb_o = real(ifft(ifftshift(jbf_o)));
    kb_o = real(ifft(ifftshift(kbf_o))); 
    return ib_o,jb_o,kb_o
end
==#
function PCA_decomposition(ib,jb,kb;mode="C",epsilon=1/2/size(ib)[1]^2)
    ib_f=fftshift(fft(ib))
    jb_f=fftshift(fft(jb))
    kb_f=fftshift(fft(kb))
    ibf_o = zeros(typeof(1.0.+im),size(ib_f));
    jbf_o = zeros(typeof(1.0.+im),size(jb_f));
    kbf_o = zeros(typeof(1.0.+im),size(kb_f));
    nx,ny,nz=size(ib);

    mib=mean(ib);
    mjb=mean(jb);
    mkb=mean(kb);
    Bhat = [mib,mjb,mkb]
    for i in 1:nx, j in 1:ny, k in 1:nz
        # get the kk vector
        kx = i - div(nx,2)-1;
        ky = j - div(ny,2)-1;
        kz = k - div(nz,2)-1;
        kk = [kx,ky,kz]
        

        if (sum(kk.^2.0)>0)
            kdotB=sum(kk.*Bhat)./sqrt.(sum(kk.^2.0))./sqrt.(sum(Bhat.^2.0))
            if (1.0.-abs.(kdotB) >epsilon)
            
                if (mode=="A")
                    zeta_o = ucross3(kk,Bhat)
                elseif (mode=="C")
                    zeta_A = ucross3(kk,Bhat)
                    zeta_o = ucross3(kk,zeta_A)
                elseif (mode=="P")
                    zeta_o = kk./sqrt.(sum(kk.^2.0))
                end
                # assign a vector at (i,j,k)
                bf = [ib_f[i,j,k],jb_f[i,j,k],kb_f[i,j,k]]
                # do a dot product
                bfdotC = sum(bf.*zeta_o)
                ibf_o[i,j,k] = bfdotC.*zeta_o[1]
                jbf_o[i,j,k] = bfdotC.*zeta_o[2]
                kbf_o[i,j,k] = bfdotC.*zeta_o[3]
            end
        end
    end
    ib_o = real(ifft(ifftshift(ibf_o)));
    jb_o = real(ifft(ifftshift(jbf_o)));
    kb_o = real(ifft(ifftshift(kbf_o))); 
    return ib_o,jb_o,kb_o
end

# from LazMode.jl

function s(vx::Cube,vy::Cube,vz::Cube)
    return vx.*vx.+vy.*vy.+vz.*vz;
end

function s(vx::Number,vy::Number,vz::Number)
    return vx.*vx.+vy.*vy.+vz.*vz;
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

function double_cross(a1,a2,a3,b1,b2,b3,c1,c2,c3)
    # a x (b x c)

    return cross_product(a1,a2,a3,cross_product(b1,b2,b3,c1,c2,c3)...)
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



function PCASF_decomposition(ib,jb,kb;mode="C",epsilon=1/2/size(ib)[1]^2,β=1)
    ib_f=fftshift(fft(ib))
    jb_f=fftshift(fft(jb))
    kb_f=fftshift(fft(kb))
    ibf_o = zeros(typeof(1.0.+im),size(ib_f));
    jbf_o = zeros(typeof(1.0.+im),size(jb_f));
    kbf_o = zeros(typeof(1.0.+im),size(kb_f));
    nx,ny,nz=size(ib);

    mib=mean(ib);
    mjb=mean(jb);
    mkb=mean(kb);
    Bhat = [mib,mjb,mkb]
    for i in 1:nx, j in 1:ny, k in 1:nz
        # get the kk vector
        kx = i - div(nx,2)-1;
        ky = j - div(ny,2)-1;
        kz = k - div(nz,2)-1;
        kk = [kx,ky,kz]
        α=β/2
        if (sum(kk.^2.0)>0)
            kdotB=sum(kk.*Bhat)./sqrt.(sum(kk.^2.0))./sqrt.(sum(Bhat.^2.0))            
            D=(1+α)^2-4*α*kdotB*kdotB

            # fast mode 
            prefactor1 = 1-sqrt(D)+α;
            prefactor2 = 1+sqrt(D)-α;

            # slow mode
            prefactor3 = 1-sqrt(D)-α;
            prefactor4 = 1+sqrt(D)+α;
            
            if (1.0.-abs.(kdotB) >epsilon)
            
                if (mode=="A")
                    zeta_o = ucross3(kk,Bhat)
                elseif (mode=="C")
                    zeta_A = ucross3(kk,Bhat)
                    zeta_o = ucross3(kk,zeta_A)
                elseif (mode=="P")
                    zeta_o = n3(kk)
                elseif (mode=="S")
                    zeta_ll = n3(Bhat)
                    zeta_pp = n3(kk.-cdot3(kk,zeta_ll).*zeta_ll)
                    zeta_x  = -prefactor2.*zeta_ll.+prefactor1.*zeta_pp
                    if (m3(zeta_x)>epsilon)
                        zeta_o  = n3(zeta_x)  
                    else
                        zeta_o = [0,0,0]                  
                    end
                elseif (mode=="F")
                    zeta_ll = n3(Bhat)
                    zeta_pp = n3(kk.-cdot3(kk,zeta_ll).*zeta_ll)
                    zeta_x  = -prefactor3.*zeta_ll.+prefactor4.*zeta_pp
                    if (m3(zeta_x)>epsilon)
                        zeta_o  = n3(zeta_x)  
                    else
                        zeta_o = [0,0,0]                  
                    end
                end
                # assign a vector at (i,j,k)
                bf = [ib_f[i,j,k],jb_f[i,j,k],kb_f[i,j,k]]
                # do a dot product
                bfdotC = sum(bf.*zeta_o)
                ibf_o[i,j,k] = bfdotC.*zeta_o[1]
                jbf_o[i,j,k] = bfdotC.*zeta_o[2]
                kbf_o[i,j,k] = bfdotC.*zeta_o[3]
            end
        end
    end
    ib_o = real(ifft(ifftshift(ibf_o)));
    jb_o = real(ifft(ifftshift(jbf_o)));
    kb_o = real(ifft(ifftshift(kbf_o))); 
    return ib_o,jb_o,kb_o
end

function PCASF_decomposition_v(iv,jv,kv,ib,jb,kb;mode="C",epsilon=1/2/size(ib)[1]^2,β=1)
    iv_f=fftshift(fft(iv))
    jv_f=fftshift(fft(jv))
    kv_f=fftshift(fft(kv))
    ivf_o = zeros(typeof(1.0.+im),size(iv_f));
    jvf_o = zeros(typeof(1.0.+im),size(jv_f));
    kvf_o = zeros(typeof(1.0.+im),size(kv_f));
    nx,ny,nz=size(ib);

    mib=mean(ib);
    mjb=mean(jb);
    mkb=mean(kb);
    Bhat = [mib,mjb,mkb]
    for i in 1:nx, j in 1:ny, k in 1:nz
        # get the kk vector
        kx = i - div(nx,2)-1;
        ky = j - div(ny,2)-1;
        kz = k - div(nz,2)-1;
        kk = [kx,ky,kz]
        α=β/2
        if (sum(kk.^2.0)>0)
            kdotB=sum(kk.*Bhat)./sqrt.(sum(kk.^2.0))./sqrt.(sum(Bhat.^2.0))            
            D=(1+α)^2-4*α*kdotB*kdotB

            # fast mode 
            prefactor1 = 1-sqrt(D)+α;
            prefactor2 = 1+sqrt(D)-α;

            # slow mode
            prefactor3 = 1-sqrt(D)-α;
            prefactor4 = 1+sqrt(D)+α;
            
            if (1.0.-abs.(kdotB) >epsilon)
            
                if (mode=="A")
                    zeta_o = ucross3(kk,Bhat)
                elseif (mode=="C")
                    zeta_A = ucross3(kk,Bhat)
                    zeta_o = ucross3(kk,zeta_A)
                elseif (mode=="P")
                    zeta_o = n3(kk)
                elseif (mode=="S")
                    zeta_ll = n3(Bhat)
                    zeta_pp = n3(kk.-cdot3(kk,zeta_ll).*zeta_ll)
                    zeta_x  = -prefactor2.*zeta_ll.+prefactor1.*zeta_pp
                    if (m3(zeta_x)>epsilon)
                        zeta_o  = n3(zeta_x)  
                    else
                        zeta_o = [0,0,0]                  
                    end
                elseif (mode=="F")
                    zeta_ll = n3(Bhat)
                    zeta_pp = n3(kk.-cdot3(kk,zeta_ll).*zeta_ll)
                    zeta_x  = -prefactor3.*zeta_ll.+prefactor4.*zeta_pp
                    if (m3(zeta_x)>epsilon)
                        zeta_o  = n3(zeta_x)  
                    else
                        zeta_o = [0,0,0]                  
                    end                  
                end
                # assign a vector at (i,j,k)
                vf = [iv_f[i,j,k],jv_f[i,j,k],kv_f[i,j,k]]
                # do a dot product
                vfdotC = sum(vf.*zeta_o)
                ivf_o[i,j,k] = vfdotC.*zeta_o[1]
                jvf_o[i,j,k] = vfdotC.*zeta_o[2]
                kvf_o[i,j,k] = vfdotC.*zeta_o[3]
            end
        end
    end
    iv_o = real(ifft(ifftshift(ivf_o)));
    jv_o = real(ifft(ifftshift(jvf_o)));
    kv_o = real(ifft(ifftshift(kvf_o))); 
    return iv_o,jv_o,kv_o
end

function getmode_z(d::Cube,vx::Cube,vy::Cube,vz::Cube,bx::Cube,by::Cube,bz::Cube;cs=0.19195751,mode=0,ma_bool=false)
    if (ma_bool)
        Ma=sqrt(mean(@. d*(vx^2+vy^2+vz^2)/(bx^2+by^2+bz^2)))
    else
        Ma=sqrt(var(bx)+var(by)+var(bz))/(mean(bz))
    end
    Ms=sqrt(var(vx)+var(vy)+var(vz))/cs
    β=2*Ma^2/Ms^2;
    if (mode==0)
        return PCASF_decomposition_v(vx,vy,vz,bx,by,bz,mode="A",β=β)
    elseif (mode==1)
        return PCASF_decomposition_v(vx,vy,vz,bx,by,bz,mode="S",β=β)
    elseif (mode==2)
        return PCASF_decomposition_v(vx,vy,vz,bx,by,bz,mode="F",β=β)
    else
        return 0
    end
end


function get_Bmode_z(d::Cube,vx::Cube,vy::Cube,vz::Cube,bx::Cube,by::Cube,bz::Cube;cs=0.19195751,mode=0,ma_bool=false)
    if (ma_bool)
        Ma=sqrt(mean(@. d*(vx^2+vy^2+vz^2)/(bx^2+by^2+bz^2)))
    else
        Ma=sqrt(var(bx)+var(by)+var(bz))/(mean(bz))
    end
    Ms=sqrt(var(vx)+var(vy)+var(vz))/cs
    β=2*Ma^2/Ms^2;
    if (mode==0)
        return PCASF_decomposition(bx,by,bz,mode="A",β=β)
    elseif (mode==1)
        return PCASF_decomposition(bx,by,bz,mode="S",β=β)
    elseif (mode==2)
        return PCASF_decomposition(bx,by,bz,mode="F",β=β)
    else
        return 0
    end
end


# end from LazMode.jl
#==
==#

end #module LazNewCh5