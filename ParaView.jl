using WriteVTK,HDF5,LazCore,LazIO,Statistics
f = h5open("hdfaa.013");
d = read(f,"gas_density");
IMap = sum(d,dims=1)[1,:,:];
Nx,Ny,Nz = size(d);
d = 0;
GC.gc();
jb = read(f,"j_mag_field");
kb = read(f,"k_mag_field");
I,Q,U = getIQU(jb,kb);
phi = .5*atan2(U,Q);
vtkfile = vtk_grid("h9_1200", Ny, Nz)    # 2-D
vtk_point_data(vtkfile, log.(IMap), "Intensity_Map");
vtk_point_data(vtkfile, cos.(phi), "Bx");
vtk_point_data(vtkfile, sin.(phi), "By");
outfiles = vtk_save(vtkfile)

function degrade(M,n)
    Nx,Ny = size(M);
    nx = Int(div(Nx,n));
    New_M = zeros((nx,nx));
    Up(x) =Int(round(x,RoundUp));
    Down(x) = Int(round(x,RoundUp));
    for j in 1:nx, i in 1:nx
        si,ei = Down((i-1)*n+1),Up(i*n);
        sj,ej = Down((j-1)*n+1),Up(j*n);
        New_M[i,j] = mean(M[si:ei,sj:ej]);
    end
    New_M
end

dePhi = degrade(phi,10);
nx,ny = size(dePhi);
vtkfile = vtk_grid("h9_1200B", nx, ny)    # 2-D                                                                        
vtk_point_data(vtkfile, cos.(dePhi), "Bx");
vtk_point_data(vtkfile, sin.(dePhi), "By");                                                                             
outfiles = vtk_save(vtkfile)
