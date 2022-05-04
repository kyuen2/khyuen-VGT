module LazRHT_investigation
using PyCall,FITSIO,LazType,Statistics,FFTW,LazPyWrapper,LinearAlgebra,LazRHT,LazCFA
	##############################################################################
	#
	# Copyright (c) 2019
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
	 rht.py must be in your local folder!
	==#
	export put_a_filament,random_filament_generator
	export put_an_arc,random_arc_generator,random_filament_orientation_generator
	export put_a_gaussian_filament,random_gaussian_filament_orientation_generator

	function odd(a::Number)
		return mod(a,2)
	end

	function put_a_filament(A::Mat,x::Number,y::Number,len::Number,wid::Number,angle::Number;period=true)
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

	function put_an_arc(A::Mat,x::Number,y::Number,r::Number,wid::Number,angles::Number,anglee::Number;period=true)
		Nx,Ny=size(A);
		Ax=zeros(Nx,Ny);
		Nxmid=div(Nx,2);
		Nymid=div(Ny,2);
		if (anglee<angles)
			anglee+=pi*2;
		end
		for i in 1:Nx, j in 1:Ny
			idx=i-Nxmid;
			jdx=j-Nymid;
			rr=sqrt(idx^2+jdx^2);
			theta=atan(jdx,idx);
			if (theta<0)
				theta+=2*pi;
			end
			if ((rr>r) && (rr<r+wid) && (theta>angles) && (theta <anglee))
				Ax[i,j]+=1;
			end
		end
		if period
			# Current arc position at (Nxmid,NyMid)
			# Want it to be at (x,y)
			# shift (x-Nxmid,y-Nymid)
			Ax=circshift(Ax,[x-Nxmid,y-Nymid]);
		else
			Axx=zeros(3*Nx,3*Ny);
			Axx[Nx+1:2*Nx,Ny+1:2*Ny]=Ax;
			Axx=circshift(Axx,[x-Nxmid,y-Nymid]);
			Ax=Axx[Nx+1:2*Nx,Ny+1:2*Ny];
		end
	end

	function random_filament_generator(N::Number,n_filament::Number;
		min_filament_length=10,
		max_filament_length=div(N,2),
		min_filament_width=3,
		max_filament_width=div(N,10),
		period=true)
		
		A=zeros(N,N);
		filament_angle=rand(n_filament).*180.0;
		filament_x_position=rand(1:N,n_filament);
		filament_y_position=rand(1:N,n_filament);
		filament_length=rand(min_filament_length:max_filament_length,n_filament);
		filament_width=rand(min_filament_width:max_filament_width,n_filament);
		for ii in 1:n_filament
			Ax=put_a_filament(A,filament_x_position[ii],filament_y_position[ii],filament_length[ii],filament_width[ii],filament_angle[ii],period=period);
			A+=Ax
		end
	return A
	end

	function random_arc_generator(N::Number,n_arc::Number;
		min_arc_radius=10,
		max_arc_radius=div(N,4),
		min_arc_width=3,
		max_arc_width=div(N,20),
		period=true)

	A=zeros(N,N);
	arc_starting_angle=rand(n_arc).*360.0./pi.-180.0./pi;
	arc_ending_angle=rand(n_arc).*360.0./pi.-180.0./pi;
	arc_x_position=rand(1:N,n_arc);
	arc_y_position=rand(1:N,n_arc);	
	arc_radius=rand(n_arc).*(max_arc_radius.-min_arc_radius).+min_arc_radius;
	arc_width=rand(n_arc).*(max_arc_width.-min_arc_width).+min_arc_width;
	for ii in 1:n_arc
		Ax=put_an_arc(A,arc_x_position[ii],arc_y_position[ii],arc_radius[ii],arc_width[ii],arc_starting_angle[ii],arc_ending_angle[ii],period=period);
		A+=Ax
	end
	return A
	end

	function random_filament_orientation_generator(N::Number,n_filament::Number,minangle::Number,maxangle::Number;
    min_filament_length=10,
    max_filament_length=div(N,2),
    min_filament_width=3,
    max_filament_width=div(N,10),
    period=true)

    A=zeros(N,N);
    filament_angle=rand(n_filament).*(maxangle.-minangle).+minangle;
    filament_x_position=rand(1:N,n_filament);
    filament_y_position=rand(1:N,n_filament);
    filament_length=rand(min_filament_length:max_filament_length,n_filament);
    filament_width=rand(min_filament_width:max_filament_width,n_filament);
    for ii in 1:n_filament
        Ax=put_a_filament(A,filament_x_position[ii],filament_y_position[ii],filament_length[ii],filament_width[ii],filament_angle[ii],period=period);
        A+=Ax
    end
	return A
	end

	function random_gaussian_filament_orientation_generator(N::Number,n_filament::Number,minangle::Number,maxangle::Number;
    min_filament_length=10,
    max_filament_length=div(N,2),
    min_filament_width=3,
    max_filament_width=div(N,10),
    period=true,angle_gaussian=false)

    A=zeros(N,N);
    if (angle_gaussian)
    	# gaussian along mean angle, dispersion of (max-min)/2
    	filament_angle=randn(n_filament).*(maxangle.-minangle).+(maxangle.+minangle)./2.0
    else
    	filament_angle=rand(n_filament).*(maxangle.-minangle).+minangle;
    end
    filament_x_position=rand(1:N,n_filament);
    filament_y_position=rand(1:N,n_filament);
    filament_length=rand(min_filament_length:max_filament_length,n_filament);
    filament_width=rand(min_filament_width:max_filament_width,n_filament);
    for ii in 1:n_filament
        Ax=put_a_gaussian_filament(A,filament_x_position[ii],filament_y_position[ii],filament_length[ii],filament_width[ii],filament_angle[ii],period=period);
        A+=Ax
    end
	return A
	end

	function crosscorr(A::Mat,B::Mat)
	    nx,ny=size(A)
	    C=fft(A).*conj(fft(B))
	    Cf=real.(ifft(C))/nx/ny
	    return Cf
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

end #module LazRHT_investigation