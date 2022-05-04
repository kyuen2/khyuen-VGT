module GalToEqr
using LazCore,LazType
	##############################################################################
	#
	# Copyright (c) 2019
	# Ka Ho Yuen, Yue Hu and Alex Lazarian
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
	 KH: This is v1.0+
	==#

#==
  KH: WCS coordinate transformation in Julia
  First edit: Apr 02 2019
  Second edit: Mar 15 2020
==#


const degtorad = pi/180;
const alpha_G = 192.85948.*degtorad;
const delta_G = 27.12825.*degtorad;
const l_NCP   = 122.93192.*degtorad;

export convert_vec_Gal_from_Eq2000,convert_Gal_from_Eq2000,convert_vec_Gal_from_Eq2000_with_angle
export convert_Eq2000_from_Gal
export generate_RADEC_matrix

function lspace(a,b,c)
   # KH : Construct a 1d linspace
   width = (b-a)/c;
   x=zeros(round(Int,c));
   for i in 1:round(Int,c)
       x[i]=a.+width*(i-1)
   end
   return x
end

function generate_RADEC_matrix(RAmin::Number,RAmax::Number,DECmin::Number,DECmax::Number,nx::Number,ny::Number)
	#==
	Generator for the RA,DEC matrix
	==#
	a=zeros(nx+1,ny);
	b=zeros(nx+1,ny);
	RA=lspace(RAmin,RAmax,nx+1)
	DEC=lspace(DECmin,DECmax,ny)
	for i in 1:nx+1
		b[i,:].=DEC;
	end
	for j in 1:ny
		a[:,j].=RA;
	end
	return a,b
end

function convert_Eq2000_from_Gal(l::Number,b::Number)
	#==
	Input:
	` l  ` : Input l matrix, in degree, one needs an extra dimesion along l coordinate
	` b ` :  Input b matrix, in degree, one needs an extra dimesion along b coordinate

	formula: We compute back the l,b coordinate matrix
	(p.s. has to use tan2)
	tan2(alpha-alpha_G) = (cos(b)sin(l_NCP-l))/(sin(b)cos(delta_G)-cos(b)sin(delta_G)cos(l_NCP-l))
	sin(delta)          = sin(b)*sin(delta_G)+cos(b)cos(delta_G)cos(l_NCP-l)
	 
	==#
	ll = l.*degtorad;
	bb = b.*degtorad;
	RA = alpha_G.+atan.(cos(bb).*sin.(l_NCP.-ll),sin.(bb).*cos(delta_G).-cos.(bb).*sin(delta_G).*cos.(l_NCP.-ll));
	DEC = asin.(sin.(bb).*sin(delta_G).+cos.(bb).*cos(delta_G).*cos.(l_NCP.-ll))
	return RA./degtorad,DEC./degtorad
end



function convert_Gal_from_Eq2000(RA::Number,DEC::Number)
	#==
	Input:
	` RA  ` : Input RA matrix, in degree, one needs an extra dimesion along RA coordinate
	` DEC ` : Input DEC matrix, in degree, one needs an extra dimesion along RA coordinate

	formula: We compute back the l,b coordinate matrix
	(p.s. has to use tan2)
	tan2(l_NCP-l) = (cos(delta_G)sin(alpha-alpha_G))/(sin(delta)cos(delta_G)-cos(delta)sin(delta_G)cos(alpha-alpha_G))
	sin(b)        = sin(delta)*sin(delta_G)+cos(delta)cos(delta_G)cos(alpha-alpha_G)
	 
	==#
	alpha = RA.*degtorad;
	delta = DEC.*degtorad;
	l = l_NCP.-atan.(cos(delta_G).*sin.(alpha.-alpha_G),sin.(delta).*cos(delta_G).-cos.(delta).*sin(delta_G).*cos.(alpha.-alpha_G));
	b = asin.(sin.(delta).*sin(delta_G).+cos.(delta).*cos(delta_G).*cos.(alpha.-alpha_G))
	return l./degtorad,b./degtorad
end

function convert_Gal_from_Eq2000(RA::Mat,DEC::Mat)
	#==
	Input:
	` RA  ` : Input RA matrix, in degree, one needs an extra dimesion along RA coordinate
	` DEC ` : Input DEC matrix, in degree, one needs an extra dimesion along RA coordinate

	formula: We compute back the l,b coordinate matrix
	(p.s. has to use tan2)
	tan2(l_NCP-l) = (cos(delta_G)sin(alpha-alpha_G))/(sin(delta)cos(delta_G)-cos(delta)sin(delta_G)cos(alpha-alpha_G))
	sin(b)        = sin(delta)*sin(delta_G)+cos(delta)cos(delta_G)cos(alpha-alpha_G)
	 
	==#
	alpha = RA.*degtorad;
	delta = DEC.*degtorad;
	l = l_NCP.-atan.(cos(delta_G).*sin.(alpha.-alpha_G),sin.(delta).*cos(delta_G).-cos.(delta).*sin(delta_G).*cos.(alpha.-alpha_G));
	b = asin.(sin.(delta).*sin(delta_G).+cos.(delta).*cos(delta_G).*cos.(alpha.-alpha_G))
	return l./degtorad,b./degtorad
end

function convert_vec_Gal_from_Eq2000(RAmin::Number,RAmax::Number,DECmin::Number,Decmax::Number,nx::Number,ny::Number;VSize=0.0000001)
	# compute the differences between (l,b) along the RA (x) axis
	# Suggest to put VSize=0.0001`
	RA,DEC=generate_RADEC_matrix(RAmin,RAmax,DECmin,Decmax,nx,ny)
	l,b=convert_Gal_from_Eq2000(RA,DEC);
	ln,bn=convert_Gal_from_Eq2000(RA.+VSize,DEC)
	l_RA_diff = ln.-l;
	b_RA_diff = bn.-b;
	ang_RA_diff = atan.(b_RA_diff./l_RA_diff);
	# From Yue:
	# In julia convention
	# phi = .5* atan2(U,Q) - 90
	# Real offset = -(90 - ang_RA_diff)
	# Real angle = phi - (90 - ang_RA_diff)
	return ang_RA_diff./degtorad
end

function convert_vec_Gal_from_Eq2000_with_angle(RAmin::Number,RAmax::Number,DECmin::Number,Decmax::Number,pol::Mat;VSize=0.0000001)
	# compute the differences between (l,b) along the RA (x) axis
	# Suggest to put VSize=0.0001`
	nx,ny=size(pol)
	RA,DEC=generate_RADEC_matrix(RAmin,RAmax,DECmin,Decmax,nx,ny)
	l,b=convert_Gal_from_Eq2000(RA,DEC);
	# Put VSize to negative if one wants to use negative RA definition.
	ln,bn=convert_Gal_from_Eq2000(RA.+VSize.*cos.(pol),DEC.+VSize.*sin.(pol))
	l_RA_diff = l.-ln;
	b_RA_diff = b.-bn; 
	ang_RA_diff = atan.(b_RA_diff./l_RA_diff);
	return ang_RA_diff
end

function convert_vec_Gal_from_Eq2000_s(RA::Number,DEC::Number,pol::Number;VSize=0.0000001)
	# compute the differences between (l,b) along the RA (x) axis
	# Suggest to put VSize=0.0001`
	l,b=convert_Gal_from_Eq2000(RA,DEC);
	ln,bn=convert_Gal_from_Eq2000(RA.-VSize.*cos.(pol),DEC.+VSize.*sin.(pol))
	l_RA_diff = ln.-l;
	b_RA_diff = bn.-b;
	ang_RA_diff = atan.(b_RA_diff./l_RA_diff);
	return ang_RA_diff
end


end# module GalToEqr