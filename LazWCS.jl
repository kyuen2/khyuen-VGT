module LazWCS
using HDF5,PyPlot,Statistics,LsqFit,PyCall,FFTW,StatsBase,Images
using LazCore,LazType,LazThermal,LazIO,LazThermal_Kritsuk
using LazRHT_investigation,LazCFA
using GalToEqr
using FITSIO


##############################################################################
#
# Copyright (c) 2020
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
 KH: This is v1.3+ only
==#

# The world coordinate for RA-DEC maps (regardless of RA-tan or RA-sin) appears
# to shift ~ 0.1 from the real cooordinates.
# the reason is because of the cos(δ) factor. When δ small the 1/cos(δ) factor 
# becomes big, so a shift happens.

# A simple correction has to be made based on this shift
# https://www.aanda.org/articles/aa/full/2002/45/aah3860/node7.html

# From GalToEqr
#   tan2(alpha-alpha_G) = (cos(b)sin(l_NCP-l))/(sin(b)cos(delta_G)-cos(b)sin(delta_G)cos(l_NCP-l))
#	sin(delta)          = sin(b)*sin(delta_G)+cos(b)cos(delta_G)cos(l_NCP-l)

# Caution: Using WCS from python is MUCH better!


const degtorad = pi/180;
const radtodeg = 1/degtorad;
const alpha_G = 192.85948.*degtorad;
const delta_G = 27.12825.*degtorad;
const l_NCP   = 122.93192.*degtorad;

export real_coordiantes_tangent

	function real_coordiantes_tangent(f)
		# f: FITS file header
		d=read(f[1])
		nx,ny,nv=size(d)
		CRVAL1=read_header(f[1])["CRVAL1"];
		CDELT1=read_header(f[1])["CDELT1"];
		CRPIX1=read_header(f[1])["CRPIX1"];
		CRVAL2=read_header(f[1])["CRVAL2"];
		CDELT2=read_header(f[1])["CDELT2"];
		CRPIX2=read_header(f[1])["CRPIX2"];
		CRVAL3=read_header(f[1])["CRVAL3"];
		CDELT3=read_header(f[1])["CDELT3"];
		CRPIX3=read_header(f[1])["CRPIX3"];

		# julia convention: flip x-y axis.
		x=degtorad.*CDELT1.*(Array(1:ny).-CRPIX1);
		y=degtorad.*CDELT2.*(Array(1:nx).-CRPIX2);
		v=(CRVAL3.+CDELT3.*(Array(1:nv).-CRPIX3))./1000;
		RA=zeros(nx,ny)
		DEC=zeros(nx,ny)
		for i in 1:nx, j in 1:ny
			Φ=atan.(-y[i],x[j]) # galactic 180 range
			θ=atan.(1.0./sqrt.(x[j].^2.0.+y[i].^2.0)) # TAN projection

			#RA[i,j] = CRVAL1.+radtodeg.*atan.(cos(θ).*sin.(Φ),sin.(θ).*cos(CRVAL1.*degtorad).+cos.(θ).*sin(CRVAL1.*degtorad).*cos.(Φ));
			DEC[i,j] = radtodeg.*asin.(sin.(θ).*sin(CRVAL2.*degtorad).-cos.(Φ).*cos(θ).*cos.(CRVAL2.*degtorad))
			RA[i,j]  = CRVAL1.+radtodeg.*asin.(cos.(θ).*sin.(Φ)./cos.(DEC[i,j].*degtorad))
		end

		return RA,DEC,v
	end

end # module LazWCS