module LazIN
using PyCall,FITSIO,FFTW,Statistics
using LazType,LazCore

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
	 KH: This is v1.0+
	==#

#==
  KH: Function specified for Ion-Neutral Gradients
==#
export ddg,ddg_decomposition
function ddg(A::Mat,distx,disty;dx=1,periodic=true)
	#==
	ddg : distance depeendent gradient
	  A        : Input map
	  distance : distance pixel
	2nd order central difference operator
	==#
	if periodic
		Arx=circshift(A,[ distx,0]);
		Alx=circshift(A,[-distx,0]);
		Ary=circshift(A,[0, disty]);
		Aly=circshift(A,[0,-disty]);
		Ax=(0.5./dx/distx).*(Arx.+Alx.-2.0.*A);
		Ay=(0.5./dx/disty).*(Ary.+Aly.-2.0.*A);
		return Ax,Ay
	else
		error("Non-periodic maps are not supported yet")
	end
end

function ddg_decomposition(A::Mat,phi::Mat,distx,disty;dx=1,periodic=true)
	#==
	ddg : distance depeendent gradient
	  A        : Input map
	  phi	   : magnetic field angle
	  distance : distance pixel
	2nd order central difference operator
	==#
	Ax,Ay=ddg(A,distx,disty,dx=dx,periodic=periodic);
	Bx=cos.(phi);
	By=sin.(phi);
	# Notice B is a unit vector
	A_ll=Ax.*Bx.+Ay.*By
	A_ll_x = A_ll.*Bx;
	A_ll_y = A_ll.*By;
	A_pp_x = Ax.-A_ll_x;
	A_pp_y = Ay.-A_ll_y;
	return A_ll_x,A_ll_y,A_pp_x,A_pp_y;
end





end #module LazIN