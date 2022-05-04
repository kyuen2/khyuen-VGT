##############################################################################
#
# Copyright (c) 2018 
# Ka Ho Yuen, Ka Wai Ho, Yue Hu, Junda Chen and Alex Lazarian
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


module LazType

# Function Bar Alias
export Vec 	# Abstract 1D Array
export Mat 	# Abstract 2D Array
export Cube	# Abstract 3D Array

const  Vec{T}  = Array{T,1};
const  Mat{T}  = Array{T,2};
const Cube{T}  = Array{T,3};
const Vecx{T}  = LinRange{T};



end
