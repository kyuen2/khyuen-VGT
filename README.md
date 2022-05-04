# khyuen-VGT
The **Velocity Gradient Technique (VGT)** code, as of the private repository `LazTech-VGT`, merged with `LazTech-VGT-new`. 

## Introduction

Last Updated: May 4 2022 by Ka Ho Yuen (kyuen2.astro@gmail.com) 

This is the main code for the Velocity Gradient Technique (Yuen & Lazarian 2017a,b, Lazarian et.al 2017, Lazarian & Yuen 2018ab, Yuen et.al 2018a, Lazarian et.a 2018a, Hu et.al 2018a)

The code embedded here records the **development of the codes over the past few years (since 2017), which some of the codes might not be up-to-date**. 
For each file they have their own license file. You are free to edit them as long as proper acknowledgment is given.

## Prerequisite

Either
1. [Julia 1.6.2](https://julialang.org/downloads/) (or Julia > 1.0) is the only application you need to install now, or 
2. Use a [Docker](https://hub.docker.com/repository/docker/kyuen2/gsa-hi)

## The structure of the module

Unless explicitly mentioned, the author of the module is by default Ka Ho Yuen (KH). 

The *minimal* version of VGT are listed as follows:

### Quick Start Instruction, Dependency Installer and User Config
1. `startup.sh` or `startup.bat`: Quick start command for windows and linux user.
2. `LazInstaller.jl`: Install Julia packages for the project. 

### Header definitions 
1. `LazIO.jl` : deal with simple I/O 
2. `LazType.jl`  : define universal type alias in the project

### Main modules
1. `LazCyvecd.jl`: The module supporting the three modules above in terms of vector operations.  
2. `LazCore.jl`: The base module corresponding to Yuen & Lazarian 2017a,b, and Lazarian & Yuen 2018a.
3. `LazPCA.jl`: The new technique introduced from Lazarian et.al 2018a and Hu et.al 2018a, written with Yue Hu.
4. `LazCFA.jl`: The core module for computing the anisotropy, used in Yuen et. al 2018a. Written by Ka Wai Ho and KH.
5. `LazMode.jl` : The mode code. But there are some issues now. Check `LazNewCh5.jl`.

There are extra modules established due to the development of the VGT

### Thermal modules (Newly updated Mar 25 2019)
1. `LazThermal.jl`: The core of the thermal broadening modules for our recently submitted reply
2. `LazRHT.jl` : A simple wrapper for the RHT (rht.py attached)
3. `LazSyntheticCube.jl` : A module allowing the synthesis of numerical cubes with a correct power law and anisotropy.
4. `LazThermal_Stochastic.jl`: The stochastic thermal broadening tool

### Planck-related modules
1. `GalToEqr.jl`: Provide essential Planck support
2. `LazAMW.jl`: The moving window algorithm and its variant

### LazTech-VGT-new
1. `LazNewCh5.jl`, `LazMuiltiPole.jl` : See [here](https://github.com/kyuen2/MHD_Mode)
2. `LazVDA.jl` : See [here](https://github.com/kyuen2/LazDDA) 

### Other modules
1. `LazGAC.jl`: The gradient amplitude and curvature related modules
2. `LazVTK.jl`: Supports VTK IO
3. `LazTsallis.jl`: Tsallis parameters.
4. `LazVCA.jl` : See Yuen et.al 2019
5. `ath2h5.jl` : [Ka Wai Ho](https://github.com/doraemonho)'s athena -> HDF5 module.
6. The 3D cube rotation and geometric transformation modules ( https://github.com/doraemonho/LazRotationDev)
7. Filament-related modules (In `LazGAC.jl` now)

## License to use the code

A statement should be included in the **Acknowledgment** section of any peer-reviewed journal papers: 

> This research is performed using the code `khyuen-VGT` developed by Ka Ho Yuen et.al under the support of NSF AST 1212096

Proper citations should be made according to the papers we quoted above. 

Modifications and further developement of the code is free under the GPLv3:

```
GPLv3: https://www.gnu.org/licenses/
##############################################################################
#
# Copyright (c) 2016-2022
# Ka Ho Yuen, Ka Wai Ho, Yue Hu, Junda Chen and Alex Lazarian
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

```
