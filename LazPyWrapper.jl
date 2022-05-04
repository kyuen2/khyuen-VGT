module LazPyWrapper
using PyCall,LazType
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
	export meshgrid,meshgrid_j

	function __init__()
	py"""
	import numpy as np

	def pymeshgrid(X,Y):
	    return np.meshgrid(X,Y)
	"""
	end

	meshgrid(X,Y)=py"pymesgrid"(X,Y)

	function lspace(a,b,c)
           # KH : Construct a 1d linspace
           width = (b-a)/c;
           x=zeros(round(Int,c));
           for i in 1:round(Int,c)
               x[i]=a+width*(i-1)
           end
           return x
    end

	function meshgrid_j(nxstart,nxend,nxbin,nystart,nyend,nybin)
		nxc=lspace(nxstart,nxend,nxbin);
		nyc=lspace(nystart,nyend,nybin);
		X=zeros(length(nxc),length(nyc));
		Y=zeros(length(nxc),length(nyc));
		for j in 1:length(nyc)
			X[i,:]=nxc;
		end
		for i in 1:length(nxc)
			Y[:,j]=nyc;
		end
		return X,Y
	end

end # module LazPyWrapper