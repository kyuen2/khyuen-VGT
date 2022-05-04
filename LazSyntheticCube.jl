module LazSyntheticCube
# Public machages
using PyCall,LsqFit,FFTW,Statistics
# LazTech Modules
using LazCore,LazType,LazPCA
#@pyimport numpy as np
#@pyimport scipy.interpolate as si
#sir=si.RectBivariateSpline;


##############################################################################
#
# Copyright (c) 2019
# Ka Ho Yuen, Ka Wai Ho and Alex Lazarian
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

#==
 KH: This is v1.0+ only
 Create synthetic cubes for numerical testing purposes
==#

export inject_spectra,inject_grav_spectra
export inject_aniso_spectra,inject_aniso_spectra_3d

# KH: For some magical reason they like rand(Float64, size...) more than rand(size...)

function inject_spectra(A::Cube,kkk::Number)
	#=
		KH Yuen @ LazTech Mar 05, 2018
		For the new paper with Ka Wai Ho. Test the agrument of SPGs by Lazarian \& Yuen (2018b)
		Variable def
		A :: Arbitrary cubes, carrying a spectrum scaling of k_0
		kkk :: Index for the additional turbulence spectrum scaling. By default the injection is E(x) ~ x^-k
	=#
	Af = fftshift(fft(A));
	nx,ny,nz=size(A);
	Ac = 0;
	for k in 1:nz, j in 1:ny, i in 1:nx
		ii = i-div(nx,2)-1;
		jj = j-div(ny,2)-1;
		kk = k-div(nz,2)-1;
		rr = sqrt(ii^2+jj^2+kk^2);
		if (rr!=0)
			Af[i,j,k]=Af[i,j,k]*rr^(-kkk);
			Ac += rr^(-kkk)
		else
			Af[i,j,k]=mean(A);
		end
	end
	Af[div(nx,2),div(ny,2),div(nz,2)]=mean(A);
	return real.(ifft(ifftshift(Af./Ac)));
end


function inject_grav_spectra(A::Cube,kkk::Number,k2::Number)
	#=
		KH Yuen @ LazTech Mar 05, 2018
		Let's say if we only band the spectrum after some wavenumber k2?
		Variable def
		A :: Arbitrary cubes, carrying a spectrum scaling of k_0
		kkk :: Index for the additional turbulence spectrum scaling. By default the injection is E(x) ~ x^-k
		k2 :: Bendng threshold
	=#
	Af = fftshift(fft(A));
	nx,ny,nz=size(A);
	Ac = 0;
	for k in 1:nz, j in 1:ny, i in 1:nx
		ii = i-div(nx,2)-1;
		jj = j-div(ny,2)-1;
		kk = k-div(nz,2)-1;
		rr = sqrt(ii^2+jj^2+kk^2);
		if (rr!=0)
			if (rr>k2)
				Af[i,j,k]=Af[i,j,k]*rr^(-kkk);
				Ac += rr^(-kkk)
			else
				Ac +=1
			end
		else
			Af[i,j,k]=mean(A);
		end
	end
	Af[div(nx,2),div(ny,2),div(nz,2)]=mean(A);
	return real.(ifft(ifftshift(Af./Ac)));
end


function inject_aniso_spectra(nsize::Number,p::Number,pp::Number,bending_pixel::Number);
	n=-pp
	A=zeros(nsize,nsize);
	for j in 1:nsize, i in 1:nsize
		ii = i-div(nsize,2)-1;
		jj = j-div(nsize,2)-1;
		rr = sqrt(ii^2+(jj^2)^p);
		if (rr>bending_pixel) 
			A[i,j]=0; 
		elseif (rr>0)
			A[i,j]=abs(rr)^n
		else
			A[i,j]=1;
		end
	end
	AA=real.(ifft(sqrt.(fft(A)).*exp.(im.*rand(Float64,size(A)).*2.0.*pi)));
	return AA	
end 

function inject_aniso_spectra_3d(nsize::Number,p::Number,pp::Number,bending_pixel::Number);
	n=-pp
    A=zeros(nsize,nsize,nsize);
	for k=1:nsize,j in 1:nsize, i in 1:nsize
		ii = i-div(nsize,2)-1;
		jj = j-div(nsize,2)-1;
		kk = k-div(nsize,2)-1;
		rr = sqrt(ii^2+jj^2+(kk^2)^p);
		if (rr>bending_pixel) 
            # Dissipation range
            # A[i,j,k]=abs(rr)^-4; 
            A[i,j,k]=0; 
		elseif (rr>0)
			A[i,j,k]=abs(rr)^n
		else
			A[i,j,k]=1;
		end
	end
    phase=rand(Float64,size(A));
    AA=real.(ifft(sqrt.(ifftshift(A)).*exp.(im.*phase.*2.0.*pi)));
    BB=imag.(ifft(sqrt.(ifftshift(A)).*exp.(im.*phase.*2.0.*pi)));
	return AA,BB
end 

function inject_dens_aniso_spectra_3d(nsize::Number,p::Number,pp::Number,bending_pixel::Number);
	n=-pp
    A0=exp.(randn(nsize,nsize,nsize));
    A=fftshift(fft(A0))
	for k=1:nsize,j in 1:nsize, i in 1:nsize
		ii = i-div(nsize,2)-1;
		jj = j-div(nsize,2)-1;
		kk = k-div(nsize,2)-1;
		rr = sqrt(ii^2+jj^2+(kk^2)^p);
		if (rr>bending_pixel) 
            # Dissipation range
            # A[i,j,k]=abs(rr)^-4; 
            A[i,j,k]=A[i,j,k]*0; 
		elseif (rr>0)
			A[i,j,k]=A[i,j,k]*abs(rr)^n
		else
			A[i,j,k]=A[i,j,k];
		end
	end
    phase=rand(Float64,size(A));
    AA=abs.(ifft(sqrt.(ifftshift(A)).*exp.(im.*phase.*2.0.*pi)));
	return AA
end 

end #module LazSyntheticCube
