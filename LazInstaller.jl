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

module LazInstaller

export install,pyinstall
export lazDependencies,lazpyDependencies

lazDependencies = [
	"PyCall"
	"Conda"
	"LsqFit"
	"StatsBase"
	"FITSIO"
	"HDF5"
	"FFTW"
	"Interpolations"
	"ImageSegmentation"
	"Images"		# We dont need the image solver right now
	"Statistics"
	"Arpack"
	"PyPlot"
	"LinearAlgebra"
	"WriteVTK"
	"FastGaussQuadrature"
];

lazpyDependencies = [
	"scipy"
	"numpy"
	"matplotlib"
	"astropy"
];

#==
	KH: For some systems the installation of FITSIO will have problems.
	Suggest to perform `sudo apt install libcfitsio5 libcfitsio-bin libcfitsio-dev` before manually typing `] add FITSIO`
==#

using  Pkg

function install()

	# 1. Check if module is installed
	packages = Pkg.installed()

	# 2. Install module on behave
	for depmodule = lazDependencies
		if get(packages, depmodule, v"0.0") == v"0.0"
			println("Pkg.add($depmodule)...")
			Pkg.add(depmodule)
			println("Pkg.add($depmodule) finished")
		end
	end
	# println("All dependencies are installed properly.")

	# 3. Suggest change of module load path
	# println("")
end

function pyinstall()
	for db = lazpyDependencies
		Conda.add(db)
	end
end

end
