module LazAMW
using PyCall,FITSIO,LazType,LazCore,Statistics,FFTW
	##############################################################################
	#
	# Copyright (c) 2019
	# Ka Ho Yuen, Ka Wai Ho, Yue Hu and Alex Lazarian
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
	export moving_window_gfft,Helmholtz_Decomposition_2D

	#==
	KH: Helmholtz decomposition
	F = -grad phi + curl A + const	
	del F  = -del grad phi = -L phi
	curl F = curl curl A =  + L A
    LA = -4pi G rho
	B_{divfree} = curl A
	A = 1/(4pi) conv(1/r,curl A);
	==#
	function MatCurl(Map::Mat;dx=1.0)
	nx,ny = size(Map);
	Cz = zeros(nx-2,ny-2);
	Bx = cos.(Map);
	By = sin.(Map);
	dBxdy = (Bx[2:end-1,3:end].- Bx[2:end-1,1:end-2])./2.0./dx;
	dBydx = (By[3:end,2:end-1].- By[1:end-2,2:end-1])./2.0./dx;
	Cz = dBxdy.-dBydx;
	return Cz
	end

	function MatDiv(Map::Mat;dx=1.0)
	nx,ny = size(Map);
	Bx = cos.(Map);
	By = sin.(Map);
	dBxdx = (Bx[2:end-1,3:end].- Bx[2:end-1,1:end-2])./2.0./dx;
	dBydy = (By[3:end,2:end-1].- By[1:end-2,2:end-1])./2.0./dx;
	return dBxdx.+dBydy
	end

	function MatGrad(Map::Mat;dx=1.0)
	nx,ny = size(Map);
	Bx = cos.(Map);
	By = sin.(Map);
	dBxdx = (Bx[2:end-1,3:end].- Bx[2:end-1,1:end-2])./2.0./dx;
	dBydy = (By[3:end,2:end-1].- By[1:end-2,2:end-1])./2.0./dx;
	return dBxdx,dBydy
	end

	function MatCurlGrad(Map::Mat;dx=1.0)
	nx,ny = size(Map);
	Bx = cos.(Map);
	By = sin.(Map);
	dBxdx = (Bx[2:end-1,3:end].- Bx[2:end-1,1:end-2])./2.0./dx;
	dBydy = (By[3:end,2:end-1].- By[1:end-2,2:end-1])./2.0./dx;
	return dBydy,-dBxdx
	end

	function DivFreeCorrection(VG::Mat;dx=1.0)
	    A  = convgravity2d(MatCurl(VG,dx),dx.^2)./4.0./pi;
	    return A
	end

	function CurlFreeCorrection(VG::Mat;dx=1.0)
		Adivv = convgravity2d(MatDiv(VG,dx),dx.^2)./4.0./pi
		Adivx,Adivy=MatGrad(Adivv,dx);
		return Adivx,Adivy
	end

	function Helmholtz_Decomposition_2D(A::Mat;dx=1)
		#==
		A is the input angle
		A_curl = div_free part
		A_grad = curl_free_part
		A_const = constant part
		==#		
		Ax=cos.(A)
		Ay=sin.(A)
		Ax_curl,Ay_curl = MatCurlGrad(DivFreeCorrection(A,dx),dx);
		Ax_grad,Ay_grad = CurlFreeCorrection(A,dx)
		Ax_const=Ax[3:end-2,3:end-2].-Ax_curl.-Ax_grad;
		Ay_const=Ay[3:end-2,3:end-2].-Ay_curl.-Ay_grad;
		A_curl = atan.(Ay_curl,Ax_curl);
		A_grad = atan.(Ay_grad,Ax_grad);
		A_const = atan.(Ay_const,Ax_const);

		Aamp_curl = sqrt.(Ax_curl.^2.0.+Ay_curl.^2.0);
		Aamp_grad = sqrt.(Ax_grad.^2.0.+Ay_grad.^2.0);
		Aamp_const = sqrt.(Ax_const.^2.0.+Ay_const.^2.0);
		return A_grad,A_curl,A_const,Aamp_grad,Aamp_curl,Aamp_const
	end
	#===

	Moving functions

	===#

	function moving_window_gfft(cna::Mat,dm)
	    # Idea:
	    # The cosine mean and sin mean can be done by convolution theorem
	    # and then being processed by a simple arctan
	    nx,ny=size(cna);
	    ccna=cos.(cna);
	    scna=sin.(cna);
	    ccnag=imfilter(ccna,Kernel.gaussian(dm));
	    scnag=imfilter(scna,Kernel.gaussian(dm));
	    aa=atan.(scnag./ccnag)
	    return aa
	end

    function moving_window_wgfft(cna::Mat,dm,amp::Mat)
	 # Idea:
	 # The cosine mean and sin mean can be done by convolution theorem
	 # and then being processed by a simple arctan
	 nx,ny=size(cna);
	 ccna=cos(cna);
	 scna=sin(cna);
	 ccnag=Images.imfilter_gaussian(amp.*ccna,[dm,dm]);
	 scnag=Images.imfilter_gaussian(amp.*scna,[dm,dm]);
	 aa=atan2(scnag,ccnag)
	 return aa
	end

	function moving_window_tfft(cna::Mat,dm)
	 # Idea:
	 # The cosine mean and sin mean can be done by convolution theorem
	 # and then being processed by a simple arctan
	 nx,ny=size(cna);
	 ccna=cos(cna);
	 scna=sin(cna);
	 ccnag=strictfilter2d(ccna,1,dm);
	 scnag=strictfilter2d(scna,1,dm);
	 aa=atan2(scnag,ccnag)
	 return aa
	end

	#==
	KH: From GTA_v10.jl
	==#

	function union_pixels(d::Cube,threshold,kmin,kmax)
	 dd=similar(d);
	 dd[d.<threshold]=0;dd[d.>threshold]=1;
	 nx,ny,nv=size(d);
	 dx=ones(nx,ny);
	 for k in kmin:kmax, j in 1:ny, i in 1:nx
	  dx[i,j]=dx[i,j]*dd[i,j,k]
	 end
	 return dx
	end

	function convgravity2d(v::Mat,dA)
	  nx,ny=size(v);
	  dx=dA^(1/2);
	  Nx=2*nx;
	  Ny=2*ny;
	  G_r = zeros(typeof(v[1]),(Nx,Ny));
	  # Here one can invert any kinds of function you want.
	  for i in 1:Nx
	   for j in 1:Ny
	    xx = (i-nx-1)*dx;
	    yy = (j-ny-1)*dx;
	    G_r[i,j] = (xx^2+yy^2)^(-.5);
	   end
	  end
	  G_r[.~isfinite.(G_r)] .= 1/dx;
	  G_r[1,:].=0;
	  G_r[:,1].=0;
	  G_rt=ifftshift(G_r);
	  G_r=0;
	  G_k=fft(G_rt);
	  G_rt=0;
	  rho_r=zeros(Nx,Ny);
	  rho_r[1:nx,1:ny].=v;
	  rho_k=fft(rho_r);
	  phi_k=G_k.*rho_k.*dA;
	  rho_r=0;
	  rho_k=0;
	  G_k=0;
	  phi_r=ifft(phi_k);
	  phi_k=0;
	  phi_rr=real.(phi_r[1:nx,1:ny]);
	  phi_r=0;
	  return phi_rr;
	end

	function angle_averaging(ca::Mat)
	 nx,ny=size(ca);
	 return atan2(sum(sin(ca[~isnan(ca)])),sum(cos(ca[~isnan(ca)])));
	end

	function angle_averaging(ca::Vec)
	 #nx,ny=size(ca);
	 return atan2(sum(sin(ca[~isnan(ca)])),sum(cos(ca[~isnan(ca)])));
	end

	function angle_dispersion(ca::Mat)
	 nx,ny=size(ca);
	 return sqrt(-log(mean(sin(ca[~isnan(ca)]))^2+mean(cos(ca[~isnan(ca)]))^2));
	end

	function angle_regulation(ca::Mat)
	 ca[ca.<-pi/2]+=pi;
	 ca[ca.>pi/2]-=pi;
	 return ca
	end

	function boxcar(v::Mat,dA,pixel)
	   nx,ny=size(v);
	   dx=dA^(1/2);
	   L=dx*nx;
	   Nx=2*nx;
	   Ny=2*ny;
	   G_r = SharedArray(typeof(v[1]),(Nx,Ny));
	   # Here one can invert any kinds of function you want.
	   sigma = pixel*dx;
	   for j in 1:Ny,i in 1:Nx
		xx = (i-nx-1)*dx;
		yy = (j-ny-1)*dx;
		if ((abs(xx)<sigma) && (abs(yy)<sigma))
		 G_r[i,j] = 1/(2*sigma)^2;
		end
	   end
	   #println(string(sum(G_r)))
	   G_r[1,:]=0;
	   G_r[:,1]=0;
	   G_rt=ifftshift(fetch(G_r));
	   G_r=0;
	   G_k=fft(G_rt);
	   G_rt=0;
	   rho_r=zeros(Nx,Ny);
	   # This is for periodic system,
	   #  [ v v ]
	   #  [ v v ]
	   rho_r[1:nx,1:ny]=v;
	   rho_r[1:nx,ny+1:Ny]=v;
	   rho_r[nx+1:Nx,1:ny]=v;
	   rho_r[nx+1:Nx,ny+1:Ny]=v;
	   rho_k=fft(rho_r);
	   phi_k=G_k.*rho_k.*dA;
	   rho_r=0;
	   rho_k=0;
	   G_k=0;
	   phi_r=ifft(phi_k);
	   phi_k=0;
	   phi_rr=real(phi_r[1:nx,1:ny]);
	   phi_r=0;
	   return phi_rr;
	end
	 


	function strictfilter2d(v::Mat,dA,pixel)
	   nx,ny=size(v);
	   dx=dA^(1/2);
	   L=dx*nx;
	   Nx=2*nx;
	   Ny=2*ny;
	   G_r = SharedArray(typeof(v[1]),(Nx,Ny));
	   # Here one can invert any kinds of function you want.
	   sigma = pixel*dx;
	   for j in 1:Ny,i in 1:Nx
		xx = (i-nx-1)*dx;
		yy = (j-ny-1)*dx;
		if ((abs(xx)<sigma) && (abs(yy)<sigma))
		 G_r[i,j] = 1/((sigma-1)^2+sigma^2+2*sigma*(sigma-1));
		end
	   end
	   G_r[1,:]=0;
	   G_r[:,1]=0;
	   G_rt=ifftshift(fetch(G_r));
	   G_r=0;
	   G_k=fft(G_rt);
	   G_rt=0;
	   rho_r=zeros(Nx,Ny);
	   # This is for periodic system,
	   #  [ v v ]
	   #  [ v v ]
	   rho_r[1:nx,1:ny]=v;
	   rho_r[1:nx,ny+1:Ny]=v;
	   rho_r[nx+1:Nx,1:ny]=v;
	   rho_r[nx+1:Nx,ny+1:Ny]=v;
	   rho_k=fft(rho_r);
	   phi_k=G_k.*rho_k.*dA;
	   rho_r=0;
	   rho_k=0;
	   G_k=0;
	   phi_r=ifft(phi_k);
	   phi_k=0;
	   phi_rr=real(phi_r[1:nx,1:ny]);
	   phi_r=0;
	   return phi_rr;
	end
	 

	function moving_window(cx::Mat,cy::Mat,dn,dm)
	 # dn: Block size
	 # dm : Kernel size
	 nx,ny=size(cx);
	 cna,cns=sban2d_f(cx,cy,dn);
	 # Assuming periodic boundary conditions
	 # Correcting vectors = convoluting the angles with 
	 cnax=similar(cna);
	 dmh=div(dm,2);
	 for j in 1:ny, i in 1:nx
	  cnax[i,j]=angle_averaging(circshift(cna,(i-dmh,j-dmh))[1:dm,1:dm]);
	 end
	 return cnax
	end

	#In principle, moving window does not need to embedded with block averaging
	function moving_window_raw(cna::Mat,dn,dm)
	 dmh=div(dm,2);
	 cnax=similar(cna)
	 nx,ny=size(cnax)
	 for j in 1:ny, i in 1:nx
	  cnax[i,j]=angle_averaging(circshift(cna,(i-dmh,j-dmh))[1:dm,1:dm]);
	 end
	 return cnax
	end


#==
	function moving_window_gfft(cna::Mat,dm)
	 # Idea:
	 # The cosine mean and sin mean can be done by convolution theorem
	 # and then being processed by a simple arctan
	 nx,ny=size(cna);
	 ccna=cos(cna);
	 scna=sin(cna);
	 ccnag=Images.imfilter_gaussian(ccna,[dm,dm]);
	 scnag=Images.imfilter_gaussian(scna,[dm,dm]);
	 aa=atan2(scnag,ccnag)
	 return aa
	end
==#

	function angle_constraining_real(ca::Mat,dn)
	 # Angle constraining using turbulent conditions
	 # i.e. max differences of neighboring angle ~ BS^(1/3)
	 # Assuming the map is periodic, and sba-ed
	 nx,ny=size(ca);
	 cax,cay=cgrad2d(ca,1);			#Abusing the gradient operator
	 # Only cares on orientation
	 # All angle < 
	 cax=angle_averaging(cax);
	 cay=angle_averaging(cay);
	 return 0;
	end
	 
	function angle_constraining(ca::Mat,dn1,multiple)
	 # So called averaging over different scales
	 dn2=dn1*multiple;
	 cx=cos(ca);
	 cy=sin(ca);
	 cna1,cns1=sban2d(cx,cy,dn1);
	 cna2,cns2=sban2d(cx,cy,dn2);
	 nx,ny=size(cx);
	 #angle normalization
	 #take atan after tan
	 cna1n=atan(tan(cna1));
	 cna2n=atan(tan(cna2));
	 nx,ny=size(ca);
	 cna=similar(cna1);
	 for i in 1:nx, j in 1:ny 
	  posx_b = div(i-1,dn2)+1;
	  posy_b = div(j-1,dn2)+1;
	  posx_s = div(i-1,dn1)+1;
	  posy_s = div(j-1,dn1)+1;
	  cna[posx_s,posy_s] = 0.5*(cna1n[posx_s,posy_s]+cna2n[posx_b,posy_b]);
	 end
	 return cna
	end

end #module LazAMW