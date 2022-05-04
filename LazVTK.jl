module LazVTK
export VTKWrite
using WriteVTK,LazType

    ##############################################################################
    #
    # Copyright (c) 2019
    # Ka Ho Yuen, Ka Wai Hoand Alex Lazarian
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
     VTK dumping wrapper 
    ==#


function VTKWrite(A::Cube,name::String)
    nx,ny,nz=size(A)
    vtkfile = vtk_grid(name, nx, ny, nz)
    vtk_point_data(vtkfile, A,name)
    outfiles = vtk_save(vtkfile)
end

function VTKWrite(A::Dict,name::String)
    Constant=false
    for k in keys(A)
        if Constant==false
            typeof(A[k]) <: Cube || error("The thing inside should be 3d array!")
                nx,ny,nz=size(A[k])
                vtkfile = vtk_grid(name, nx, ny, nz)
                Constant=true
        end
        vtk_point_data(vtkfile, A,k)
    end
    outfiles = vtk_save(vtkfile)

end

end