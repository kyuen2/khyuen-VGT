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

"""
	module LazIO

Handle IO and basic cube operations.

Author: Ka Ho Yuen, Yue Hu, Dora Ho, Junda Chen

Changelog:
	- Mike Initiate module LazIO

Todo:
	- Performance: Add benchmark method to test I/O
	- Accuracy: Benmark accuracy to legacy methods
	- Bug: gc() not callable

"""
module LazIO

using LazType

using HDF5


export hdfreader,savf
export getIQU, getdIQU, getIQU_dust
export getIC
export getByz, getwByz



function benchmark()
	error("LazIO.jl : benchmark currently not available")
end

function savf(s::AbstractString,v::Any)
	s=Symbol(s)
	@eval (($s) = ($v))
end

"""
	hdfreader(path::String)

Read HDF5 Cube in specified `path`.
`hdf5path` must be a valid path in UNIX or Windows form

Example:
```julia-repl

julia> d,iv,jv,kv,ib,jb,kb = hdfreader("data.h5")

```

"""
function hdfreader(path::String)

	f = h5open(path)

	d  = read(f,"gas_density");

	iv = read(f,"i_velocity");
	jv = read(f,"j_velocity");
	kv = read(f,"k_velocity");

	ib = read(f,"i_mag_field");
	jb = read(f,"j_mag_field");
	kb = read(f,"k_mag_field");

	return d,iv,jv,kv,ib,jb,kb

end

function sum_project_x( A ::Cube)
	return sum(A, dims=1)[1, :, :]
end

"""
	getIQU(jb::Cube,kb::Cube)
	getdIQU(d::Cube, jb::Cube, kb::Cube)
	getIQU_dust(d::Cube, ib::Cube, jb::Cube, kb::Cube)

Get intensity(`I`) and stokes parameters(`Q`, `U`) in differnet scenario.

	getIC(d::Cube, iv::Cuve)

Get intensity and centroid map.

	getByz(jb::Cube, kb::Cube)
	getwByz(d::Cube, jb::Cube, kb::Cube)

Get projected magnetic field

Example:

```julia-repl

julia> d,iv,jv,kv,ib,jb,kb = hdfreader("data.h5")

julia> I,Q,U = getIQU(jb, kb)

julia> I,Q,U = getdIQU(d, jb, kb)

julia> I,Q,U = getIQU_dust(d, jb, kb)


julia> I,C   = getIC(d, iv)


julia> By, Bz = getByz(jb, kb)

julia> By, Bz = getwByz(d, jb, kb)

```

"""
function getIQU(jb::Cube,kb::Cube)
	bb = jb.^2 + kb.^2;
	cb = kb.^2 - jb.^2;
	sb = -2 * jb .* kb;

	I = sum_project_x( bb )
	Q = sum_project_x( cb )
	U = sum_project_x( sb )

	return I,Q,U
end

function getdIQU(d::Cube, jb::Cube, kb::Cube)

	bb = d.* (jb.^2 + kb.^2);
	cb = d.* (kb.^2 - jb.^2);
	sb = d.* (-2 * jb .* kb);

	I = sum_project_x( bb )
	Q = sum_project_x( cb )
	U = sum_project_x( sb )

	return I, Q, U
end


function getIQU_dust(d::Cube,ib::Cube,jb::Cube,kb::Cube)

	# WARN: Memory > 8G should work fine here
	bb = jb.^2+kb.^2;
	b  = ib.^2+bb;
	cb = (kb.^2-jb.^2)./bb;
	sb = 2*jb.*kb./bb;

	# gc(); # garbage collection is disabled in > v1.0.0 in case you wonder

	I = sum_project_x( d .* bb )
	Q = sum_project_x( d .* cb )
	U = sum_project_x( d .* sb )
	return I,Q,U
end

function getIC(d::Cube,iv::Cube)
	I = sum_project_x( d )
	C = sum_project_x( d .* iv ./ I )
	return I, C
end


function getByz(jb::Cube,kb::Cube)
	By = sum_project_x( d .* jb )
	Bz = sum_project_x( d .* kb )
 	return By, Bz
end


function getwByz(d::Cube,jb::Cube,kb::Cube)
	 I = sum_project_x( d )
	By = sum_project_x( d .* jb ./ I )
	Bz = sum_project_x( d .* kb ./ I )
 	return By, Bz
end
end
