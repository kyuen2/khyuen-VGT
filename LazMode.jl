module LazMode
using StatsBase,HDF5, LsqFit, PyCall, FITSIO, Images, PyPlot, FFTW, Statistics
using LazCore, LazType, LazThermal, LazNewCore
using Base.Threads
np=pyimport("numpy");

#=
 KH Yuen @ Lazarian Technology
 Original code CL03decompo_v10.jl
 	1st Modified : Jan 06 2018
 	2nd Modified : Dec 19 2018
 Migrated to LazMode
 	3rd Modified : Dec 22 2019
 Previous Note:
	 (1) Numerical setting:
		(a) my cube has sonic speed of 0.19195751
		(b) my cube's mean field is always z
		(c) B field is scaled up by sqrt(2pi), i.e. real B = b in numerical cube/2pi
	 (2) CL02 has the correct cos^2 factor
	 and both CL03 and CL02 are actually using "mean field" for decomposition
	 (3) The method is not applicable to Ma>1
	 (4) The method is very memory intensive
	 (5) In the new version of Julia v1.0, multiple lines have to be changed.
 Current Note:
     (1) Only one mode is out for each calculation
     (2) there is no further assumption on direction of B and cs. they are now input parameters
     (3) The CL02 version is wrong since the fast and slow modes are mistakenly swapped.
     (4) Recently added alf mode B algorithm.
=#

export getmode,getdensitymode,get_Bmode,getmode_mt,get_Bmode_mt

function s(vx::Cube,vy::Cube,vz::Cube)
	return vx.*vx.+vy.*vy.+vz.*vz;
end

function s(vx::Number,vy::Number,vz::Number)
	return vx.*vx.+vy.*vy.+vz.*vz;
end

function dot_product(a1,a2,a3,b1,b2,b3)
	return a1*b1+a2*b2+a3*b3
end

function dot_product(a::Vec,b::Vec)
	return sum(a.*b);
end

function cross_product(a1,a2,a3,b1,b2,b3)
	#==
	| x̂  ŷ  ẑ|
	|a1 a2 a3|
	|b1 b2 b3|
	==#
	return [a2*b3-a3*b2,a3*b1-a1*b3,a1*b2-a2*b1]
end

function double_cross(a1,a2,a3,b1,b2,b3,c1,c2,c3)
	# a x (b x c)

	return cross_product(a1,a2,a3,cross_product(b1,b2,b3,c1,c2,c3)...)
end

function t(a1,a2,a3,b1,b2,b3)
	x=dot_product(a1,a2,a3,b1,b2,b3)/sqrt(s(a1,a2,a3))/sqrt(s(b1,b2,b3));
	if (x>1) 
		x=1 
	end;
	if (x<-1)
		x=-1
	end;
	return acos(x)
end

function getmode(d::Cube,vx::Cube,vy::Cube,vz::Cube,bx::Cube,by::Cube,bz::Cube;cs=1,mode=0)
	# mode
	#  0: Alfven
	#  1: Slow
	#  2: Fast
	#v2=s(vx,vy,vz);
	b2=s(bx,by,bz);
	#ma=sqrt(mean(d.*v2./b2));
	#ms=sqrt(var(vx)+var(vy)+var(vz))/cs;
	alpha=mean(d.*cs.*cs)./mean(b2)

	v2=0;b2=0;GC.gc()

	# get mean direction in cube
	mbx=mean(bx);
	mby=mean(by);
	mbz=mean(bz);

	vxf=fftshift(fft(vx));
	vyf=fftshift(fft(vy));
	vzf=fftshift(fft(vz));

	vxq=zeros(typeof(vxf[1]),size(vx))
	vyq=zeros(typeof(vxf[1]),size(vx))
	vzq=zeros(typeof(vxf[1]),size(vx))

	nx,ny,nz=size(vx);

	for i in 1:nx, j in 1:ny, k in 1:nz
		idx=i-div(nx,2)-1;
		jdx=j-div(ny,2)-1;
		kdx=k-div(nz,2)-1;

		kk=sqrt(s(idx,jdx,kdx))
		kx=0;
		ky=0;
		kz=0;
		if (kk>0)
			theta=t(mbx,mby,mbz,idx,jdx,kdx)

			D=(1+alpha)^2-4*alpha*cos(theta)*cos(theta)

			# fast mode 
			prefactor1 = 1-sqrt(D)+alpha;
			prefactor2 = 1+sqrt(D)-alpha;

			# slow mode
			prefactor3 = 1-sqrt(D)-alpha;
			prefactor4 = 1+sqrt(D)+alpha;

			k_ll = kk*cos(theta)
			k_pp = kk*sin(theta)

			kunit_ll=[mbx,mby,mbz]./sqrt(s(mbx,mby,mbz));
			kvec=[idx,jdx,kdx];
			kvec_pp = kvec.-dot_product(kvec[1],kvec[2],kvec[3],kunit_ll[1],kunit_ll[2],kunit_ll[3]).*kunit_ll;
			kunit_pp=kvec_pp./sqrt(s(kvec_pp[1],kvec_pp[2],kvec_pp[3]))
			phi_unit = cross_product(kunit_pp[1],kunit_pp[2],kunit_pp[3],kunit_ll[1],kunit_ll[2],kunit_ll[3])

			# Find the unit vectors

			if (mode==0)
				# Alfven mode
				#= CL02:
				z_a=k_ll x k_pp
				=#
				if (1.0.-abs.(dot_product(kvec[1],kvec[2],kvec[3],kunit_ll[1],kunit_ll[2],kunit_ll[3]))>1/2/size(bx)[1]^2)
					ka=phi_unit./sqrt(s(phi_unit[1],phi_unit[2],phi_unit[3]))
					kx=ka[1]
					ky=ka[2]
					kz=ka[3]
				else
					ka=0
				end
			elseif (mode==1)
				# Slow mode
				#= CL03 (Corrected by LY18a):
				z_s \propto (-1-sqrt(D)+alpha) k_ll +(1-sqrt(D)+alpha)k_pp
				=#
				#ks=k_ll.*kunit_ll.+((prefactor3/prefactor4)*cot(theta)^2).*k_pp.*kunit_pp;
				ks = (-prefactor2).*k_ll.*kunit_ll.+prefactor1.*k_pp.*kunit_pp;
				ks=ks./sqrt(s(ks[1],ks[2],ks[3]))
				kx=ks[1]
				ky=ks[2]
				kz=ks[3]				
			elseif (mode==2)
				# Fast mode
				#= CL03 (Corrected by LY18a):
				z_f \propto  (-1+sqrt(D)+alpha)k_ll + (1+sqrt(D)+alpha) k_pp
				=#
				#kf=((prefactor1/prefactor2)*tan(theta)^2).*k_ll.*kunit_ll.+k_pp.*kunit_pp;
				kf=(-prefactor3).*k_ll.*kunit_ll.+prefactor4.*k_pp.*kunit_pp;
				kf=kf./sqrt(s(kf[1],kf[2],kf[3]))
				kx=kf[1]
				ky=kf[2]
				kz=kf[3]	
			end
			vxq[i,j,k]=(vxf[i,j,k]*kx+vyf[i,j,k]*ky+vzf[i,j,k]*kz)*kx
			vyq[i,j,k]=(vxf[i,j,k]*kx+vyf[i,j,k]*ky+vzf[i,j,k]*kz)*ky
			vzq[i,j,k]=(vxf[i,j,k]*kx+vyf[i,j,k]*ky+vzf[i,j,k]*kz)*kz
		else
			vxq[i,j,k]=0
			vyq[i,j,k]=0
			vzq[i,j,k]=0
		end
	end
	vxq[isnan.(vxq)].=0;
	vyq[isnan.(vyq)].=0;
	vzq[isnan.(vzq)].=0;
	vxq=real(ifft(ifftshift(vxq)))
	vyq=real(ifft(ifftshift(vyq)))
	vzq=real(ifft(ifftshift(vzq)))

	return vxq,vyq,vzq
end


function getmode_mt(d::Cube,vx::Cube,vy::Cube,vz::Cube,bx::Cube,by::Cube,bz::Cube;cs=1,mode=0)
	# mode
	#  0: Alfven
	#  1: Slow
	#  2: Fast
	#v2=s(vx,vy,vz);
	b2=s(bx,by,bz);
	#ma=sqrt(mean(d.*v2./b2));
	#ms=sqrt(var(vx)+var(vy)+var(vz))/cs;
	alpha=mean(d.*cs.*cs)./mean(b2)

	v2=0;b2=0;GC.gc()

	# get mean direction in cube
	mbx=mean(bx);
	mby=mean(by);
	mbz=mean(bz);

	vxf=fftshift(fft(vx));
	vyf=fftshift(fft(vy));
	vzf=fftshift(fft(vz));

	vxq=zeros(typeof(vxf[1]),size(vx))
	vyq=zeros(typeof(vxf[1]),size(vx))
	vzq=zeros(typeof(vxf[1]),size(vx))

	nx,ny,nz=size(vx);

	@threads for ii in 1:nx*ny*nz
		i,j,k=idx2ijk(ii,nx)
		idx=i-div(nx,2)-1;
		jdx=j-div(ny,2)-1;
		kdx=k-div(nz,2)-1;

		kk=sqrt(s(idx,jdx,kdx))
		kx=0;
		ky=0;
		kz=0;
		if (kk>0)
			theta=t(mbx,mby,mbz,idx,jdx,kdx)

			D=(1+alpha)^2-4*alpha*cos(theta)*cos(theta)

			# fast mode 
			prefactor1 = 1-sqrt(D)+alpha;
			prefactor2 = 1+sqrt(D)-alpha;

			# slow mode
			prefactor3 = 1-sqrt(D)-alpha;
			prefactor4 = 1+sqrt(D)+alpha;

			k_ll = kk*cos(theta)
			k_pp = kk*sin(theta)

			kunit_ll=[mbx,mby,mbz]./sqrt(s(mbx,mby,mbz));
			kvec=[idx,jdx,kdx];
			kvec_pp = kvec.-dot_product(kvec[1],kvec[2],kvec[3],kunit_ll[1],kunit_ll[2],kunit_ll[3]).*kunit_ll;
			kunit_pp=kvec_pp./sqrt(s(kvec_pp[1],kvec_pp[2],kvec_pp[3]))
			phi_unit = cross_product(kunit_pp[1],kunit_pp[2],kunit_pp[3],kunit_ll[1],kunit_ll[2],kunit_ll[3])

			# Find the unit vectors

			if (mode==0)
				# Alfven mode
				#= CL02:
				z_a=k_ll x k_pp
				=#
				if (1.0.-abs.(dot_product(kvec[1],kvec[2],kvec[3],kunit_ll[1],kunit_ll[2],kunit_ll[3]))>1/2/size(bx)[1]^2)
					ka=phi_unit./sqrt(s(phi_unit[1],phi_unit[2],phi_unit[3]))
					kx=ka[1]
					ky=ka[2]
					kz=ka[3]
				else
					ka=0
				end
			elseif (mode==1)
				# Slow mode
				#= CL03 (Corrected by LY18a):
				z_s \propto (-1-sqrt(D)+alpha) k_ll +(1-sqrt(D)+alpha)k_pp
				=#
				#ks=k_ll.*kunit_ll.+((prefactor3/prefactor4)*cot(theta)^2).*k_pp.*kunit_pp;
				ks = (-prefactor2).*k_ll.*kunit_ll.+prefactor1.*k_pp.*kunit_pp;
				ks=ks./sqrt(s(ks[1],ks[2],ks[3]))
				kx=ks[1]
				ky=ks[2]
				kz=ks[3]				
			elseif (mode==2)
				# Fast mode
				#= CL03 (Corrected by LY18a):
				z_f \propto  (-1+sqrt(D)+alpha)k_ll + (1+sqrt(D)+alpha) k_pp
				=#
				#kf=((prefactor1/prefactor2)*tan(theta)^2).*k_ll.*kunit_ll.+k_pp.*kunit_pp;
				kf=(-prefactor3).*k_ll.*kunit_ll.+prefactor4.*k_pp.*kunit_pp;
				kf=kf./sqrt(s(kf[1],kf[2],kf[3]))
				kx=kf[1]
				ky=kf[2]
				kz=kf[3]	
			end
			vxq[i,j,k]=(vxf[i,j,k]*kx+vyf[i,j,k]*ky+vzf[i,j,k]*kz)*kx
			vyq[i,j,k]=(vxf[i,j,k]*kx+vyf[i,j,k]*ky+vzf[i,j,k]*kz)*ky
			vzq[i,j,k]=(vxf[i,j,k]*kx+vyf[i,j,k]*ky+vzf[i,j,k]*kz)*kz
		else
			vxq[i,j,k]=0
			vyq[i,j,k]=0
			vzq[i,j,k]=0
		end
    end
	vxq[isnan.(vxq)].=0;
	vyq[isnan.(vyq)].=0;
	vzq[isnan.(vzq)].=0;
	vxq=real(ifft(ifftshift(vxq)))
	vyq=real(ifft(ifftshift(vyq)))
	vzq=real(ifft(ifftshift(vzq)))

	return vxq,vyq,vzq
end

function getdensitymode(d::Cube,vx::Cube,vy::Cube,vz::Cube,bx::Cube,by::Cube,bz::Cube;cs=1,mode=1)
	# mode
	#  1: Slow
	#  2: Fast
	#v2=s(vx,vy,vz);
	b2=s(bx,by,bz);
	#ma=sqrt(mean(d.*v2./b2));
	#ms=sqrt(var(vx)+var(vy)+var(vz))/cs;
	alpha=mean(d.*cs.*cs)./mean(b2)
	va=sqrt(mean(b2./d));

	v2=0;b2=0;GC.gc()

	# get mean direction in cube
	mbx=mean(bx);
	mby=mean(by);
	mbz=mean(bz);
	bb=sqrt(s(mbx,mby,mbz));

	vxf=fftshift(fft(vx));
	vyf=fftshift(fft(vy));
	vzf=fftshift(fft(vz));

	vxq=zeros(typeof(vxf[1]),size(vx))
	vyq=zeros(typeof(vxf[1]),size(vx))
	vzq=zeros(typeof(vxf[1]),size(vx))
	dq=zeros(typeof(vxf[1]),size(d));

	nx,ny,nz=size(vx);

	for i in 1:nx, j in 1:ny, k in 1:nz
		idx=i-div(nx,2)-1;
		jdx=j-div(ny,2)-1;
		kdx=k-div(nz,2)-1;
		kvec=[idx,jdx,kdx];
		kk=sqrt(s(idx,jdx,kdx))
		kx=0;
		ky=0;
		kz=0;
		cc=va;
		if (kk>0)
			theta=t(mbx,mby,mbz,idx,jdx,kdx)

			D=(1+alpha)^2-4*alpha*cos(theta)*cos(theta)

			# fast mode 
			prefactor1 = 1-sqrt(D)+alpha;
			prefactor2 = 1+sqrt(D)-alpha;

			# slow mode
			prefactor3 = 1-sqrt(D)-alpha;
			prefactor4 = 1+sqrt(D)+alpha;

			k_ll = kk*cos(theta)
			k_pp = kk*sin(theta)

			kunit_ll=[mbx,mby,mbz]./sqrt(s(mbx,mby,mbz));
			kvec_pp = kvec.-dot_product(kvec[1],kvec[2],kvec[3],kunit_ll[1],kunit_ll[2],kunit_ll[3]).*kunit_ll;
			kunit_pp=kvec_pp./sqrt(s(kvec_pp[1],kvec_pp[2],kvec_pp[3]))
			phi_unit = cross_product(kunit_pp[1],kunit_pp[2],kunit_pp[3],kunit_ll[1],kunit_ll[2],kunit_ll[3])

			# Find the unit vectors

			if (mode==0)
				# Alfven mode
				#= CL02:
				z_a=k_ll x k_pp
				=#
				if (1.0.-abs.(dot_product(kvec[1],kvec[2],kvec[3],kunit_ll[1],kunit_ll[2],kunit_ll[3]))>1/2/size(bx)[1]^2)
					ka=phi_unit./sqrt(s(phi_unit[1],phi_unit[2],phi_unit[3]))
					kx=ka[1]
					ky=ka[2]
					kz=ka[3]
				else
					ka=0
				end
			elseif (mode==1)
				# Slow mode
				#= CL03 (Corrected by LY18a):
				z_s \propto (-1-sqrt(D)+alpha) k_ll +(1-sqrt(D)+alpha)k_pp
				=#
				#ks=k_ll.*kunit_ll.+((prefactor3/prefactor4)*cot(theta)^2).*k_pp.*kunit_pp;
				ks = (-prefactor2).*k_ll.*kunit_ll.+prefactor1.*k_pp.*kunit_pp;
				ks=ks./sqrt(s(ks[1],ks[2],ks[3]))
				kx=ks[1]
				ky=ks[2]
				kz=ks[3]
				cc=sqrt(0.5*va*va*(1+alpha-sqrt(D)))
			elseif (mode==2)
				# Fast mode
				#= CL03 (Corrected by LY18a):
				z_f \propto  (-1+sqrt(D)+alpha)k_ll + (1+sqrt(D)+alpha) k_pp
				=#
				#kf=((prefactor1/prefactor2)*tan(theta)^2).*k_ll.*kunit_ll.+k_pp.*kunit_pp;
				kf=(-prefactor3).*k_ll.*kunit_ll.+prefactor4.*k_pp.*kunit_pp;
				kf=kf./sqrt(s(kf[1],kf[2],kf[3]))
				kx=kf[1]
				ky=kf[2]
				kz=kf[3]
				cc=sqrt(0.5*va*va*(1+alpha+sqrt(D)))	
			end
			vxq[i,j,k]=(vxf[i,j,k]*kx+vyf[i,j,k]*ky+vzf[i,j,k]*kz)*kx
			vyq[i,j,k]=(vxf[i,j,k]*kx+vyf[i,j,k]*ky+vzf[i,j,k]*kz)*ky
			vzq[i,j,k]=(vxf[i,j,k]*kx+vyf[i,j,k]*ky+vzf[i,j,k]*kz)*kz
			dq[i,j,k]=mean(d)/cc/kk*dot_product(kvec[1],kvec[2],kvec[3],vxq[i,j,k],vyq[i,j,k],vzq[i,j,k]);
		else
			vxq[i,j,k]=0
			vyq[i,j,k]=0
			vzq[i,j,k]=0
			dq[i,j,k]=0;
		end
		
	end
	vxq[isnan.(vxq)].=0;
	vyq[isnan.(vyq)].=0;
	vzq[isnan.(vzq)].=0;
	dq[isnan.(dq)].=0;
	vxq=real(ifft(ifftshift(vxq)))
	vyq=real(ifft(ifftshift(vyq)))
	vzq=real(ifft(ifftshift(vzq)))
	dq=real(ifft(ifftshift(dq)))

	return dq,vxq,vyq,vzq
end


function get_Bmode(d::Cube,vx::Cube,vy::Cube,vz::Cube,bx::Cube,by::Cube,bz::Cube;cs=1,mode=0)

	# CL03 right before eqA35
	# b_k = 1/c * hat{k} cross (B cross v_k)
	# Alfven mode speed = v_A

	# mode
	#  0: Alfven
	v2=s(vx,vy,vz);
	b2=s(bx,by,bz);
	#ma=sqrt(mean(d.*v2./b2));
	#ms=sqrt(var(vx)+var(vy)+var(vz))/cs;
	alpha=mean(d.*cs.*cs)./mean(b2);
	va=sqrt(mean(b2./d));
	v2=0;b2=0;GC.gc()

	# get mean direction in cube
	mbx=mean(bx);
	mby=mean(by);
	mbz=mean(bz);

	vxf=fftshift(fft(vx));
	vyf=fftshift(fft(vy));
	vzf=fftshift(fft(vz));

	bxq=zeros(typeof(vxf[1]),size(vx))
	byq=zeros(typeof(vxf[1]),size(vx))
	bzq=zeros(typeof(vxf[1]),size(vx))

	nx,ny,nz=size(vx);

	for i in 1:nx, j in 1:ny, k in 1:nz
		idx=i-div(nx,2)-1;
		jdx=j-div(ny,2)-1;
		kdx=k-div(nz,2)-1;

		kk=sqrt(s(idx,jdx,kdx))
		kx=0;
		ky=0;
		kz=0;
		if (kk>0)
			theta=t(mbx,mby,mbz,idx,jdx,kdx)

			D=(1+alpha)^2-4*alpha*cos(theta)*cos(theta)
			cfast=sqrt(0.5*va^2*((1+alpha)+sqrt((1+alpha)^2-4*alpha*cos(theta)*cos(theta))))
			cslow=sqrt(0.5*va^2*((1+alpha)-sqrt((1+alpha)^2-4*alpha*cos(theta)*cos(theta))))

			# fast mode 
			prefactor1 = 1-sqrt(D)+alpha;
			prefactor2 = 1+sqrt(D)-alpha;

			# slow mode
			prefactor3 = 1-sqrt(D)-alpha;
			prefactor4 = 1+sqrt(D)+alpha;

			k_ll = kk*cos(theta)
			k_pp = kk*sin(theta)

			kunit_ll=[mbx,mby,mbz]./sqrt(s(mbx,mby,mbz));
			kvec=[idx,jdx,kdx];
			kvec_pp = kvec.-dot_product(kvec[1],kvec[2],kvec[3],kunit_ll[1],kunit_ll[2],kunit_ll[3]).*kunit_ll;
			kunit_pp=kvec_pp./sqrt(s(kvec_pp[1],kvec_pp[2],kvec_pp[3]))
			phi_unit = cross_product(kunit_pp[1],kunit_pp[2],kunit_pp[3],kunit_ll[1],kunit_ll[2],kunit_ll[3])

 
			if (mode==0)
				# Alfven mode
				#= CL02:
				z_a=k_ll x k_pp
				=#
				if (1.0.-abs.(dot_product(kvec[1],kvec[2],kvec[3],kunit_ll[1],kunit_ll[2],kunit_ll[3]))>1/2/size(bx)[1]^2)
					ka=phi_unit./sqrt(s(phi_unit[1],phi_unit[2],phi_unit[3]))
					kx=ka[1]
					ky=ka[2]
					kz=ka[3]
				else
					ka=0
				end
			elseif (mode==1)
				# Slow mode
				#= CL03 (Corrected by LY18a):
				z_s \propto (-1-sqrt(D)+alpha) k_ll +(1-sqrt(D)+alpha)k_pp
				=#
				#ks=k_ll.*kunit_ll.+((prefactor3/prefactor4)*cot(theta)^2).*k_pp.*kunit_pp;
				ks = (-prefactor2).*k_ll.*kunit_ll.+prefactor1.*k_pp.*kunit_pp;
				ks=ks./sqrt(s(ks[1],ks[2],ks[3]))
				kx=ks[1]
				ky=ks[2]
				kz=ks[3]				
			elseif (mode==2)
				# Fast mode
				#= CL03 (Corrected by LY18a):
				z_f \propto  (-1+sqrt(D)+alpha)k_ll + (1+sqrt(D)+alpha) k_pp
				=#
				#kf=((prefactor1/prefactor2)*tan(theta)^2).*k_ll.*kunit_ll.+k_pp.*kunit_pp;
				kf=(-prefactor3).*k_ll.*kunit_ll.+prefactor4.*k_pp.*kunit_pp;
				kf=kf./sqrt(s(kf[1],kf[2],kf[3]))
				kx=kf[1]
				ky=kf[2]
				kz=kf[3]	
			end

			vxq=(vxf[i,j,k]*kx+vyf[i,j,k]*ky+vzf[i,j,k]*kz)*kx
			vyq=(vxf[i,j,k]*kx+vyf[i,j,k]*ky+vzf[i,j,k]*kz)*ky
			vzq=(vxf[i,j,k]*kx+vyf[i,j,k]*ky+vzf[i,j,k]*kz)*kz

			if (mode==0)
            	bxq[i,j,k],byq[i,j,k],bzq[i,j,k]=double_cross(idx/kk,jdx/kk,kdx/kk,mbx,mby,mbz,vxq,vyq,vzq)./va;
            elseif (mode==1)
            	bxq[i,j,k],byq[i,j,k],bzq[i,j,k]=double_cross(idx/kk,jdx/kk,kdx/kk,mbx,mby,mbz,vxq,vyq,vzq)./cslow;
            elseif (mode==2)
            	bxq[i,j,k],byq[i,j,k],bzq[i,j,k]=double_cross(idx/kk,jdx/kk,kdx/kk,mbx,mby,mbz,vxq,vyq,vzq)./cfast;
            end
		else
			bxq[i,j,k]=0
			byq[i,j,k]=0
			bzq[i,j,k]=0			
		end
	end
	bxq[isnan.(bxq)].=0;
	byq[isnan.(byq)].=0;
	bzq[isnan.(bzq)].=0;
	bxq[.~isfinite.(bxq)].=0;
	byq[.~isfinite.(byq)].=0;
	bzq[.~isfinite.(bzq)].=0;
	bxq=real(ifft(ifftshift(bxq)))
	byq=real(ifft(ifftshift(byq)))
	bzq=real(ifft(ifftshift(bzq)))

	return bxq,byq,bzq
end

function get_Bmode_mt(d::Cube,vx::Cube,vy::Cube,vz::Cube,bx::Cube,by::Cube,bz::Cube;cs=1,mode=0)

	# CL03 right before eqA35
	# b_k = 1/c * hat{k} cross (B cross v_k)
	# Alfven mode speed = v_A

	# mode
	#  0: Alfven
	v2=s(vx,vy,vz);
	b2=s(bx,by,bz);
	#ma=sqrt(mean(d.*v2./b2));
	#ms=sqrt(var(vx)+var(vy)+var(vz))/cs;
	alpha=mean(d.*cs.*cs)./mean(b2);
	va=sqrt(mean(b2./d));
	v2=0;b2=0;GC.gc()

	# get mean direction in cube
	mbx=mean(bx);
	mby=mean(by);
	mbz=mean(bz);

	vxf=fftshift(fft(vx));
	vyf=fftshift(fft(vy));
	vzf=fftshift(fft(vz));

	bxq=zeros(typeof(vxf[1]),size(vx))
	byq=zeros(typeof(vxf[1]),size(vx))
	bzq=zeros(typeof(vxf[1]),size(vx))

	nx,ny,nz=size(vx);

	@threads for ii in 1:nx*ny*nz
		i,j,k=idx2ijk(ii,nx)

		idx=i-div(nx,2)-1;
		jdx=j-div(ny,2)-1;
		kdx=k-div(nz,2)-1;

		kk=sqrt(s(idx,jdx,kdx))
		kx=0;
		ky=0;
		kz=0;
		if (kk>0)
			theta=t(mbx,mby,mbz,idx,jdx,kdx)

			D=(1+alpha)^2-4*alpha*cos(theta)*cos(theta)
			cfast=sqrt(0.5*va^2*((1+alpha)+sqrt((1+alpha)^2-4*alpha*cos(theta)*cos(theta))))
			cslow=sqrt(0.5*va^2*((1+alpha)-sqrt((1+alpha)^2-4*alpha*cos(theta)*cos(theta))))

			# fast mode 
			prefactor1 = 1-sqrt(D)+alpha;
			prefactor2 = 1+sqrt(D)-alpha;

			# slow mode
			prefactor3 = 1-sqrt(D)-alpha;
			prefactor4 = 1+sqrt(D)+alpha;

			k_ll = kk*cos(theta)
			k_pp = kk*sin(theta)

			kunit_ll=[mbx,mby,mbz]./sqrt(s(mbx,mby,mbz));
			kvec=[idx,jdx,kdx];
			kvec_pp = kvec.-dot_product(kvec[1],kvec[2],kvec[3],kunit_ll[1],kunit_ll[2],kunit_ll[3]).*kunit_ll;
			kunit_pp=kvec_pp./sqrt(s(kvec_pp[1],kvec_pp[2],kvec_pp[3]))
			phi_unit = cross_product(kunit_pp[1],kunit_pp[2],kunit_pp[3],kunit_ll[1],kunit_ll[2],kunit_ll[3])

 			
			if (mode==0)
				# Alfven mode
				#= CL02:
				z_a=k_ll x k_pp
				=#
				if (1.0.-abs.(dot_product(kvec[1],kvec[2],kvec[3],kunit_ll[1],kunit_ll[2],kunit_ll[3]))>1/2/size(bx)[1]^2)
					ka=phi_unit./sqrt(s(phi_unit[1],phi_unit[2],phi_unit[3]))
					kx=ka[1]
					ky=ka[2]
					kz=ka[3]
				else
					ka=0
				end
			elseif (mode==1)
				# Slow mode
				#= CL03 (Corrected by LY18a):
				z_s \propto (-1-sqrt(D)+alpha) k_ll +(1-sqrt(D)+alpha)k_pp
				=#
				#ks=k_ll.*kunit_ll.+((prefactor3/prefactor4)*cot(theta)^2).*k_pp.*kunit_pp;
				ks = (-prefactor2).*k_ll.*kunit_ll.+prefactor1.*k_pp.*kunit_pp;
				ks=ks./sqrt(s(ks[1],ks[2],ks[3]))
				kx=ks[1]
				ky=ks[2]
				kz=ks[3]				
			elseif (mode==2)
				# Fast mode
				#= CL03 (Corrected by LY18a):
				z_f \propto  (-1+sqrt(D)+alpha)k_ll + (1+sqrt(D)+alpha) k_pp
				=#
				#kf=((prefactor1/prefactor2)*tan(theta)^2).*k_ll.*kunit_ll.+k_pp.*kunit_pp;
				kf=(-prefactor3).*k_ll.*kunit_ll.+prefactor4.*k_pp.*kunit_pp;
				kf=kf./sqrt(s(kf[1],kf[2],kf[3]))
				kx=kf[1]
				ky=kf[2]
				kz=kf[3]	
			end

			vxq=(vxf[i,j,k]*kx+vyf[i,j,k]*ky+vzf[i,j,k]*kz)*kx
			vyq=(vxf[i,j,k]*kx+vyf[i,j,k]*ky+vzf[i,j,k]*kz)*ky
			vzq=(vxf[i,j,k]*kx+vyf[i,j,k]*ky+vzf[i,j,k]*kz)*kz

			if (mode==0)
            	bxq[i,j,k],byq[i,j,k],bzq[i,j,k]=double_cross(idx/kk,jdx/kk,kdx/kk,mbx,mby,mbz,vxq,vyq,vzq)./va;
            elseif (mode==1)
            	bxq[i,j,k],byq[i,j,k],bzq[i,j,k]=double_cross(idx/kk,jdx/kk,kdx/kk,mbx,mby,mbz,vxq,vyq,vzq)./cslow;
            elseif (mode==2)
            	bxq[i,j,k],byq[i,j,k],bzq[i,j,k]=double_cross(idx/kk,jdx/kk,kdx/kk,mbx,mby,mbz,vxq,vyq,vzq)./cfast;
            end
		else
			bxq[i,j,k]=0
			byq[i,j,k]=0
			bzq[i,j,k]=0			
		end
	end
	bxq[isnan.(bxq)].=0;
	byq[isnan.(byq)].=0;
	bzq[isnan.(bzq)].=0;
	bxq[.~isfinite.(bxq)].=0;
	byq[.~isfinite.(byq)].=0;
	bzq[.~isfinite.(bzq)].=0;
	bxq=real(ifft(ifftshift(bxq)))
	byq=real(ifft(ifftshift(byq)))
	bzq=real(ifft(ifftshift(bzq)))

	return bxq,byq,bzq
end

end # mode LazMode
