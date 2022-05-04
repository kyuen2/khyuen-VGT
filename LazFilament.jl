module LazFilament
using HDF5,PyPlot,Statistics,LsqFit,PyCall,FFTW,StatsBase,Images
using LazCore,LazType,LazThermal,LazIO,LazThermal_Kritsuk
using LazRHT_investigation,LazCFA

	##############################################################################
	#
	# Copyright (c) 2020
	# Ka Ho Yuen, Alex Lazarian and Dmitri Pogosyan
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
	 KH: This is v1.3+ only
	==#

	export put_a_GS95_filament_withQU,put_a_GS95_filament_3d,put_a_gaussian_filament_3d,put_a_GS95_filament_withQU_3d

	function boxcar2d(nx::Number,ny::Number,lx::Number,ly::Number)
	    a=zeros(nx,ny);
	    for i in 1:nx, j in 1:ny
	        idx=i-div(nx,2);
	        jdx=j-div(ny,2);
	        if ((abs(idx)<div(lx,2)) & (abs(jdx)<div(ly,2)))
	            a[i,j]=1
	        end
	    end
	    return a
	end

	function boxcorr(A::Mat,lx::Number,ly::Number,orientation::Number)
	    nx,ny=size(A);
	    ap=boxcar2d(nx,ny,lx,ly);
	    ap=rotate_2d(ap,orientation);
	    Ax=zeros(size(A));
	    Ax=crosscorr(A,Array(ap))
	    apl=length(findall(ap.>0))/length(ap)
	    return fftshift(Ax./apl)
	end

	#==
	function iboxcorr(Ax::Mat,lx::Number,ly::Number,orientation::Number)
	    nx,ny=size(Ax);
	    ap=boxcar2d(nx,ny,lx,ly);
	    ap=rotate_2d(ap,orientation);
	    A=zeros(size(Ax));
	    for i in 1:nx, j in 1:ny
	        A+=Ax[i,j].*circshift(ap,[i,j])
	    end
	    return A
	end

	==#

	function hough(A::Mat;magnification_factor=10)
	    A_hough=zeros(size(A))
	    nx,ny=size(A);
	    BB=zeros(size(A));
	    BB[:,div(ny,2)].=1;
	    for i in 1:nx, j in 1:ny
	        x=i-div(nx,2)
	        y=j-div(ny,2)
	        if ((A[i,j]>0) & ((x!=0) | (y!=0)))
	            A_hough+=A[i,j].*circshift(rotate_2d(BB,atan(-x)),[0,y])
	        end
	    end
	    return A_hough
	end

	function ihough(A_hough::Mat;thres=10)
	    A_real=zeros(size(A_hough))
	    nx,ny=size(A_hough)
	    for i in 1:nx, j in 1:ny
	        c=i-div(nx,2)
	        m=j-div(ny,2)
	        if (A_hough[i,j]>thres)
	            leng=convert(Int,round(A_hough[i,j],digits=0));
	            BBB=zeros(size(A_hough));
	            xs=div(nx-leng,2)-1
	            xe=div(nx+leng,2)
	            if (xs<1)
	                xs=1
	            end
	            if (xe>nx)
	                xe=nx
	            end
	            BBB[xs:xe,div(ny,2)].=1
	            A_real+=circshift(rotate_2d(Array(BBB'),atan(m)),[0,c]);
	        end
	    end
	    return A_real
	end

		function odd(a::Number)
		return mod(a,2)
	end


	#==
	KH: 3D version, needed for VCA paper.
	==#

	function put_a_filament_3d(A::Mat,x::Number,y::Number,z::Number,len::Number,wid::Number,angle::Number;period=true)
			Ax=zeros(size(A))
		# KH: Call the CFA rotation function LazCFA.rotate2D to perform rotation
		# KH: And then use circshift to perform displacement (if periodic)
		Nx,Ny=size(A);
		Nxmid=div(Nx,2);
		Nymid=div(Ny,2);
		Ax[Nxmid-div(len,2):Nxmid+div(len,2)-odd(len),Nxmid-div(wid,2):Nxmid+div(wid,2)-odd(wid)].=1;
		Ax=rotate_2d(Ax,angle*pi/180);
		if period
			# Current filament position at (Nxmid,NyMid)
			# Want it to be at (x,y)
			# shift (x-Nxmid,y-Nymid)
			Ax=circshift(Ax,[x-Nxmid,y-Nymid]);
		else
			Axx=zeros(3*Nx,3*Ny);
			Axx[Nx+1:2*Nx,Ny+1:2*Ny]=Ax;
			Axx=circshift(Axx,[x-Nxmid,y-Nymid]);
			Ax=Axx[Nx+1:2*Nx,Ny+1:2*Ny];
		end
		return Ax
	end

	function put_a_gaussian_filament(A::Mat,x::Number,y::Number,len::Number,wid::Number,angle::Number;period=true,full_width=true)
			Ax=zeros(size(A))
		# KH: Call the CFA rotation function LazCFA.rotate2D to perform rotation
		# KH: And then use circshift to perform displacement (if periodic)
		Nx,Ny=size(A);
		Nxmid=div(Nx,2);
		Nymid=div(Ny,2);
		for i in 1:Nx, j in 1:Ny
			ii = i - Nxmid;
			jj = j - Nymid;
			if (full_width)
				FWHM=4*sqrt(2)
			else
				FWHM=2
			end
			Ax[i,j]=1.0./sqrt(2.0.*pi).*exp(-ii^2/2.0/(len/FWHM)^2).*exp(-jj^2/2.0/(wid/FWHM)^2)
		end
		Ax=rotate_2d(Ax,angle*pi/180);
		if period
			# Current filament position at (Nxmid,NyMid)
			# Want it to be at (x,y)
			# shift (x-Nxmid,y-Nymid)
			Ax=circshift(Ax,[x-Nxmid,y-Nymid]);
		else
			Axx=zeros(3*Nx,3*Ny);
			Axx[Nx+1:2*Nx,Ny+1:2*Ny]=Ax;
			Axx=circshift(Axx,[x-Nxmid,y-Nymid]);
			Ax=Axx[Nx+1:2*Nx,Ny+1:2*Ny];
		end
		return Ax
	end

	function GS95_lh(h::Number,L_inj::Number,Ma::Number)
	    return L_inj^(1/3)*Ma^(-4/3)*h^(2/3)
	end

	function GS95_rho(h::Number,L_inj::Number,v_inj::Number,rho_0::Number,Ms::Number,Ma::Number,a::Number,b::Number;c=1)
	    d=(rho_0*v_inj)/(c*L_inj^(1/3))*h^(1/3)*Ms^a*Ma^b;
	    return d
	end

	function GS95_v(h::Number,v_inj::Number,L_inj::Number,Ma::Number)
	    return v_inj*Ma^(1/3)*(h/L_inj)^(1/3)
	end

	function N(h_dec::Number,L_inj::Number,v_inj::Number,rho_0::Number,Ms::Number,Ma::Number,a::Number,b::Number;c=1,L=2*L_inj)
	    bracket=2*L_inj/h_dec-2/Ma^2*(L_inj/L)^(3/2)
	    if (bracket>0)
	        return round(Int,1/(Ms^a*Ma^(b+8/3))*c/v_inj*bracket)
	    else
	        return 0
	    end
	end

	#==
	KH: 3D version, needed for VCA paper.
	==#

	function put_a_filament_3d(A::Cube,x::Number,y::Number,z::Number,len::Number,wid::Number,angle::Number;period=true)
	    Ax=zeros(size(A))
	    # KH: Call the CFA rotation function LazCFA.rotate2D to perform rotation
	    # KH: And then use circshift to perform displacement (if periodic)
	    Nx,Ny,Nz=size(A);
	    Nxmid=div(Nx,2);
	    Nymid=div(Ny,2);
	    Nzmid=div(Nz,2);
	    Ax[Nxmid-div(len,2):Nxmid+div(len,2)-odd(len),Nymid-div(wid,2):Nymid+div(wid,2)-odd(wid),Nzmid-div(wid,2):Nzmid+div(wid,2)-odd(wid)].=1;
	    
	    # KH: Slice by slice rotation, doable since mean field is along x-dir
	    for k in 1:Nz
	        Ax[:,:,k]=rotate_2d(Ax[:,:,k],angle*pi/180);
	    end
	    
	    if period
	        # Current filament position at (Nxmid,NyMid)
	        # Want it to be at (x,y)
	        # shift (x-Nxmid,y-Nymid)
	        Ax=circshift(Ax,[x-Nxmid,y-Nymid,z-Nzmid]);
	    else
	        Axx=zeros(3*Nx,3*Ny,3*Nz);
	        Axx[Nx+1:2*Nx,Ny+1:2*Ny,Nz+1:2*Nz]=Ax;
	        Axx=circshift(Axx,[x-Nxmid,y-Nymid,z-Nzmid]);
	        Ax=Axx[Nx+1:2*Nx,Ny+1:2*Ny,Nz+1:2*Nz];
	    end
	    return Ax
	end

	function put_a_GS95_filament_withQU(
	        A::Mat,
	        L_inj::Number,
	        v_inj::Number,
	        rho_0::Number,
	        Ms::Number,
	        Ma::Number,
	        a::Number,
	        b::Number;
	        h_dec=1,h_max=div(size(A)[1],10),c=1,L=2*L_inj)
	    # mean field default point to z_hat
	        
	    anglemax=Ma/2.0.*(180/pi);
	    
	    N_filaments=N(h_dec,L_inj,v_inj,rho_0,Ms,Ma,a,b;c=1,L=2*L_inj)
	    
	    
	    # Density storage
	    Ax=zeros(size(A))
	    
	    # velocity storage
	    Bx=zeros(size(A))
	    
	            
	    # QU storage
	    Qx=zeros(size(A));
	    Ux=zeros(size(A));
	    
	    nx,ny=size(Ax);
	    
	    # Angle, position and width generation
	    angles=(rand(N_filaments).-0.5).*anglemax;
	    x=mod1.(rand(Int,N_filaments),nx);
	    y=mod1.(rand(Int,N_filaments),ny);
	    wid=mod.(rand(Int,N_filaments),h_max-h_dec).+h_dec
	    
	    for i in 1:N_filaments
	        len=round(Int,GS95_lh(wid[i],L_inj,Ma))
	        d=GS95_rho(wid[i],L_inj,v_inj,rho_0,Ms,Ma,a,b;c=1);
	        v=GS95_v(wid[i],v_inj,L_inj,Ma)
	        U=put_a_gaussian_filament(Ax,x[i],y[i],len,wid[i],angles[i],period=true);
	        
	        # Density is positive skewed, and has to be weighted by the width (projection width)
	        Ax.+=d.*U.*wid[i]
	        # Give sign to the filament speed!
	        Bx.+=(mod(rand(Int),2)*2-1).*v.*U.*wid[i]    
	        
	        Qx.+=d.*U.*cos.(2.0.*angles[i])*wid[i]
	        Ux.+=d.*U.*sin.(2.0.*angles[i])*wid[i]
	    end
	    return Ax,Bx,Qx,Ux
	end

	function put_a_gaussian_filament_3d(A::Cube,x::Number,y::Number,z::Number,len::Number,wid::Number,angle::Number;period=true,full_width=true)
	    Ax=zeros(size(A))
	    # KH: Call the CFA rotation function LazCFA.rotate2D to perform rotation
	    # KH: And then use circshift to perform displacement (if periodic)
	    Nx,Ny,Nz=size(A);
	    Nxmid=div(Nx,2);
	    Nymid=div(Ny,2);
	    Nzmid=div(Nz,2);
	    for i in 1:Nx, j in 1:Ny, k in 1:Nz
	        ii = i - Nxmid;
	        jj = j - Nymid;
	        kk = k - Nzmid;
	        if (full_width)
	            FWHM=4*sqrt(2)
	        else
	            FWHM=2
	        end
	        Ax[i,j,k]=1.0./sqrt(2.0.*pi).*exp(-ii^2/2.0/(len/FWHM)^2).*exp(-jj^2/2.0/(wid/FWHM)^2).*exp(-kk^2/2.0/(wid/FWHM)^2)
	    end
	    # KH: Slice by slice rotation, doable since mean field is along x-dir
	    for k in 1:Nz
	        Ax[:,:,k]=rotate_2d(Ax[:,:,k],angle*pi/180);
	    end
	            
	    if period
	        # Current filament position at (Nxmid,NyMid)
	        # Want it to be at (x,y)
	        # shift (x-Nxmid,y-Nymid)
	        Ax=circshift(Ax,[x-Nxmid,y-Nymid,z-Nzmid]);
	    else
	        Axx=zeros(3*Nx,3*Ny,3*Nz);
	        Axx[Nx+1:2*Nx,Ny+1:2*Ny,Nz+1:2*Nz]=Ax;
	        Axx=circshift(Axx,[x-Nxmid,y-Nymid,z-Nzmid]);
	        Ax=Axx[Nx+1:2*Nx,Ny+1:2*Ny,Nz+1:2*Nz];
	    end
	    return Ax
	end

	function put_a_GS95_filament_3d(
        A::Cube,
        L_inj::Number,
        v_inj::Number,
        rho_0::Number,
        Ms::Number,
        Ma::Number,
        a::Number,
        b::Number;
        h_dec=1,h_max=div(size(A)[1],10),c=1,L=2*L_inj,ud_num=false)
    # mean field default point to z_hat
        
	    anglemax=Ma/2.0.*(180/pi);
	    
	    if(ud_num)
	        N_filaments=1
	    else
	        N_filaments=N(h_dec,L_inj,v_inj,rho_0,Ms,Ma,a,b;c=1,L=2*L_inj)
	    end
	    
	    
	    # Density storage
	    Ax=zeros(size(A))
	    
	    # velocity storage
	    Bx=zeros(size(A))
	        
	    nx,ny,nz=size(Ax);
	    
	    # Angle, position and width generation
	    angles=(rand(N_filaments).-0.5).*anglemax;
	    x=mod1.(rand(Int,N_filaments),nx);
	    y=mod1.(rand(Int,N_filaments),ny);
	    z=mod1.(rand(Int,N_filaments),ny);
	    wid=mod.(rand(Int,N_filaments),h_max-h_dec).+h_dec
	    
	    for i in 1:N_filaments
	        len=round(Int,GS95_lh(wid[i],L_inj,Ma))
	        d=GS95_rho(wid[i],L_inj,v_inj,rho_0,Ms,Ma,a,b;c=1);
	        v=GS95_v(wid[i],v_inj,L_inj,Ma)
	        U=put_a_gaussian_filament_3d(Ax,x[i],y[i],z[i],len,wid[i],angles[i],period=true);
	        
	        # Density is positive skewed, and has to be weighted by the width (projection width)
	        Ax.+=d.*U
	        # Give sign to the filament speed!
	        Bx.+=(mod(rand(Int),2)*2-1).*v.*U
	    end
	    return Ax,Bx
	end
	    
	function put_a_GS95_filament_withQU_3d(
	        A::Cube,
	        L_inj::Number,
	        v_inj::Number,
	        rho_0::Number,
	        Ms::Number,
	        Ma::Number,
	        a::Number,
	        b::Number;
	        h_dec=1,h_max=div(size(A)[1],10),c=1,L=2*L_inj,ud_num=false)
	    # mean field default point to z_hat
	        
	    anglemax=Ma/2.0.*(180/pi);
	    
	    if(ud_num)
	        N_filaments=1
	    else
	        N_filaments=N(h_dec,L_inj,v_inj,rho_0,Ms,Ma,a,b;c=1,L=2*L_inj)
	    end   
	    
	    # Density storage
	    Ax=zeros(size(A))
	    
	    # velocity storage
	    Bx=zeros(size(A))
	    
	            
	    # QU storage
	    Qx=zeros(size(A));
	    Ux=zeros(size(A));
	    
	    nx,ny,nz=size(Ax);
	    
	    # Angle, position and width generation
	    angles=(rand(N_filaments).-0.5).*anglemax;
	    x=mod1.(rand(Int,N_filaments),nx);
	    y=mod1.(rand(Int,N_filaments),ny);
	    z=mod1.(rand(Int,N_filaments),ny);
	    wid=mod.(rand(Int,N_filaments),h_max-h_dec).+h_dec
	    
	    for i in 1:N_filaments
	        len=round(Int,GS95_lh(wid[i],L_inj,Ma))
	        d=GS95_rho(wid[i],L_inj,v_inj,rho_0,Ms,Ma,a,b;c=1);
	        v=GS95_v(wid[i],v_inj,L_inj,Ma)
	        U=put_a_gaussian_filament_3d(Ax,x[i],y[i],z[i],len,wid[i],angles[i],period=true);
	        
	        # Density is positive skewed, and has to be weighted by the width (projection width)
	        Ax.+=d.*U
	        # Give sign to the filament speed!
	        Bx.+=(mod(rand(Int),2)*2-1).*v.*U
	        
	        Qx.+=d.*U.*cos.(2.0.*angles[i])
	        Ux.+=d.*U.*sin.(2.0.*angles[i])
	    end
	    return Ax,Bx,Qx,Ux
	end

	#==
	Usage
	A=zeros(nx,nx,nx);
	Ax,Bx,Qx,Ux=put_a_GS95_filament_withQU_3d(A,
	    div(nx,2), # L_inj
	    1,   # v_inj
	    1,   # rho_0
	    ms,   # Ms
	    ma,   # Ma
	    1,   # a
	    0,   # b
	    h_max=40,ud_num=true);
	==#

end