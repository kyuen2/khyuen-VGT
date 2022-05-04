module LazGAC
# Standard packages
using PyCall,FITSIO,Statistics,FFTW,Printf,Statistics,StatsBase
# Special packages
using ImageSegmentation,LinearAlgebra,Interpolations
# Laz-packages
using LazCore,LazPyWrapper,LinearAlgebra,LazType,LazRHT

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
	 KH: This is v1.0+ only
	 Gradient Amplitude and Curvature  (GAC)
	 Updated Jan 27 2020 to reflect the changes of the curvature paper.
	==#

	export superbit,CoM,MoI,snlabel
	export periodic_label,structure_curvature,label_lwo,RK4_curvature_torsion,RK2_curvature
	export dd,directional_derivative,cross_product,dot_product,amp,covariant_derivative,grad
	# Use RHT-output to determine the width, length of the filaments and also orientations


	#==
	 KH:
	 Surprisingly using the initialization function one can bypass 
	 the limitations from PyCall
	==#
	function __init__()
	py"""
	import scipy.ndimage.measurements as sn
	import numpy as np

	def slabel(A):
	    return sn.label(A,np.ones((3,3)))
	"""
	end

	snlabel(A)=py"slabel"(A)

	function superbit(A::Mat,b::Number)
		AA=zeros(size(A));
		AA[A.==b].=1;
		return AA;
	end

	function CoM(A::Mat;periodic=false)
		#==
		  Input:
		  	A : Mass matrix
		  Output
		    mx,my: Center of mass
		==#
		nx,ny=size(A);
		Al,Ac=snlabel(A);
		mx=0;
		my=0;
		if (Ac>1 && periodic)
			AA=fftshift(A);
			for i in 1:nx, j in 1:ny
				mx+=AA[i,j].*(i-div(nx,2));
				my+=AA[i,j].*(j-div(ny,2));
			end			
		else
			for i in 1:nx, j in 1:ny
				mx+=A[i,j].*i;
				my+=A[i,j].*j;
			end
		end
		mx=mx./sum(A);
		my=my./sum(A);
		return mx,my
	end

	function MoI(A::Mat)
		#==
		  Input:
		  	A : Mass matrix,
		  Output
		    R: Moment of Inertia Marix, a width measure
		==#
		mx,my=CoM(A);
		nx,ny=size(A);
		# Put it in center to avoid the periodic issue
		AA=circshift(A,[round(Int,mx),round(Int,my)])
		mmx=mx-round(Int,mx);
		mmy=my-round(Int,my);
		Rxx=0;
		Ryy=0;
		Rxy=0;
		for i in 1:nx, j in 1:ny
			Rxx+=A[i,j].*(i-mmx).^2;
			Ryy+=A[i,j].*(j-mmy).^2;
			Rxy+=A[i,j].*(i-mmx).*(j-mmy);
		end
		Rxx=Rxx./A;
		Ryy=Ryy./A;
		Rxy=Rxy./A;
		R=[[Rxx,Rxy] [Rxy,Ryy]];
		return R
	end

	function periodic_label(A::Mat)
		error("LazGAC.jl : Waiting for the tree type data for implementation")
		#==
			Input  
				A : Binary matrix
			Output
				Al : Label matrix 
		==#
		#==
		Al,Ac=snlabel(bitmap(A,0.5));
		nx,ny=size(A);
		# Check the boundariespc
		#==
		# Abuse the label alglorithm on 
		All,Acc=snlabel(fftshift(bitmap(A,0.5)))
		==#
		# Check the boundary in All
		All=ifftshift(Al);
		# Compare A and All
		Aout=zeros(size(A));
		# Create a copy of A
		Abit=bitmap(A,0.5)
		Aoutc=0;
		for i in 1:nx

		end
		#==
		# Dont want to do tree search
		# Abuse the geometrical properties of a periodic map, just search it for 2 times
		for i in 1:nx, j in 1:ny
			if (Abit[i,j]>0)
				Aoutc+=1
				Aout[(Al.==Al[i,j]) .| (All.==All[i,j])].=Aoutc
				Abit[(Al.==Al[i,j]) .| (All.==All[i,j])].=-10
				#Aout[All.==All[i,j]].=Aoutc
				#Abit[All.==All[i,j]].=0
			end
		end
		Aout2=zeros(size(A));
		Abit=bitmap(A,0.5)
		Aoutc=0;
		for i in nx:-1:1, j in ny:-1:1
			if (Abit[i,j]>0)
				Aoutc+=1
				Aout2[(Aout.==Aout[i,j]).|(Al.==Al[i,j]) .| (All.==All[i,j])].=Aoutc
				Abit[(Aout.==Aout[i,j]).|(Al.==Al[i,j]) .| (All.==All[i,j])].=-10
				#Aout[All.==All[i,j]].=Aoutc
				#Abit[All.==All[i,j]].=0
			end
		end	
		==#	
		return Aout,Aoutc
		==#
	end

	function label_lwo(A::Mat;WLEN=65,SMR=15,FRAC=0.7,periodic=false,want_curvature=true)
		Ax,hi,hj,rt=RHTs_wrapper(A,WLEN,SMR,FRAC);
		Axx=bitmap(A,0);
		if periodic
			Al,Ac=periodic_label(A);
		else
			Al,Ac=snlabel(A);
		end
		filament_x_cm=zeros(0);
		filament_y_cm=zeros(0);
		filament_width=zeros(0);
		filament_length=zeros(0);
		filament_orientation=zeros(0);
		filament_curvature=zeros(0);

		for i in 1:Ac
			#==
			Run through all filaments, weighted by the original matrix
			==#
			Abit=superbit(Axx,i);
			x,y=CoM(Abit.*A);
			R  =MoI(Abit.*A);
			l,h=eigvals(R);
			Rvec=eigvecs(R);
			if (l>h)
				fvec=Rvec[1,:]
			else
				fvec=Rvec[2,:]
				ll=l;
				l=h;
				h=ll; 
			end
			# Half-polar
			theta=atan(fvec[2]./fvec[1])

			push!(filament_x_cm,x)
			push!(filament_y_cm,y)
			push!(filament_width,h)
			push!(filament_length,l)
			push!(filament_orientation,theta)
			if (want_curvature)
				if (periodic)
					period=1
				else
					period=0
				end
				curvatue=structure_curvature(Abit,period);
				push!(filament_curvature,mean(curvature[Abit.>0]))
		    end
		end
		if (want_curvature)
			return filament_x_cm,filament_y_cm,filament_width,filament_length,filament_orientation,filament_curvature
		else
			return filament_x_cm,filament_y_cm,filament_width,filament_length,filament_orientation
		end
	end

	#==
	Curvature function
	From KHYuen's curvature.jl

	==#

	function RK2_curvature(u::Mat,vx::Mat,vy::Mat,nstep::Number,dt0::Number,periodic::Number)
    # KH Yuen @ Oct 28 2018, Unseo, S. Korea
    # vx,vy: Vector field
    # nstep: The number of steps required 
    # dt: user-defined time step
    # periodic:: whether the inputted velocity field is periodic.

    ###
    #
    # Caution: using the lagrangian frame would produce something that we dont understand right now
    #          use with care
    #
    ###
    
    
	    nx,ny=size(vx);
	    # Make sure nstep is odd
	    mod(nstep,2)==0 ? nstep+=1 : nstep+=0;
	    # Make sure nstep has at least 9 steps
	    (nstep>=9) ? nstep+=0 : nstep=9;
	    
	    centerstep=div(nstep,2)+1;
	    x=zeros(nx,ny,nstep);
	    y=zeros(nx,ny,nstep);
	    curvature=zeros(nx,ny);
	    #  t=zeros(nx,ny,nstep);
	    
	    
	    # Caution: KH is using Interpolations v0.3.6
	    # Newer version of Interpolations has a syntax change!
	    if (periodic>0)
	        vxx = interpolate(vx,BSpline(Quadratic(Periodic())), OnCell())
	        vyy = interpolate(vy,BSpline(Quadratic(Periodic())), OnCell())   
	    else
	         # Natural boundary condition is used
	        vxx = interpolate(vx,BSpline(Quadratic(Line())), OnCell());
	        vyy = interpolate(vy,BSpline(Quadratic(Line())), OnCell());
	    end
	    
	    
	    # a limiter to avoid skipping the cells
	    # from python.matplotlib.pyplot.streamplot
	    maxerror = 0.003
	    
	    for i in 1:nx, j in 1:ny
	    	if (u[i,j]>0)
		        x[i,j,centerstep]=i;
		        y[i,j,centerstep]=j;
		        dt=min(dt0,1/nx,1/ny,0.1);
		        for nn in 1:div(nstep,2)
		            # Forward integration  
		            ii=x[i,j,nn-1+centerstep]+dt/2*vxx[x[i,j,nn-1+centerstep],y[i,j,nn-1+centerstep]];
		            jj=y[i,j,nn-1+centerstep]+dt/2*vyy[x[i,j,nn-1+centerstep],y[i,j,nn-1+centerstep]];
		            x[i,j,nn+centerstep]=x[i,j,nn-1+centerstep]+dt/2*vxx[ii,jj];
		            y[i,j,nn+centerstep]=y[i,j,nn-1+centerstep]+dt/2*vyy[ii,jj];
		            # Backward integration  
		            ii=x[i,j,centerstep-nn+1]-dt/2*vxx[x[i,j,centerstep-nn+1],y[i,j,centerstep-nn+1]];
		            jj=y[i,j,centerstep-nn+1]-dt/2*vyy[x[i,j,centerstep-nn+1],y[i,j,centerstep-nn+1]]; 
		            x[i,j,centerstep-nn]=x[i,j,centerstep-nn+1]-dt/2*vxx[ii,jj];
		            y[i,j,centerstep-nn]=y[i,j,centerstep-nn+1]-dt/2*vyy[ii,jj];
		        end       
		        
		        Tx=zeros(nstep-4);
		        Ty=zeros(nstep-4);
		        centerstepx=div(nstep-4,2)+1;
		        for tt in centerstep-2:centerstep+2
		            Tx[tt-(centerstep-2)+1] = 1/(12*dt)*(-x[i,j,tt+2]+8*x[i,j,tt+1]-8*x[i,j,tt-1]+x[i,j,tt-2]);
		            Ty[tt-(centerstep-2)+1] = 1/(12*dt)*(-y[i,j,tt+2]+8*y[i,j,tt+1]-8*y[i,j,tt-1]+y[i,j,tt-2]);
		            TT=sqrt.(Tx[tt-(centerstep-2)+1].^2.0.+Ty[tt-(centerstep-2)+1].^2);
		            Tx[tt-(centerstep-2)+1]=Tx[tt-(centerstep-2)+1]/TT;
		            Ty[tt-(centerstep-2)+1]=Ty[tt-(centerstep-2)+1]/TT;
		        end 
		        Txx = 1.0./(12.0.*dt).*(-Tx[centerstepx+2].+8.0.*Tx[centerstepx+1].-8.0.*Tx[centerstepx-1].+Tx[centerstepx-2]);
		        Tyy = 1.0./(12.0.*dt).*(-Ty[centerstepx+2].+8.0.*Ty[centerstepx+1].-8.0.*Ty[centerstepx-1].+Ty[centerstepx-2]);
		        curvature[i,j]=sqrt.(Txx.^2+Tyy.^2);
		    else
		    	curvature[i,j]=NaN
		    end
	    end
	    return x,y,curvature
	end

	# We are using Julia.Interpolations

	function RK4_curvature_torsion(vx::Mat,vy::Mat,vz::Mat,nstep::Number,dt0::Number;periodic=1,L=1,RK4=true)
	    # Based on 
	        # KH Yuen @ Oct 28 2018, Unseo, S. Korea
	        # Updated by KH Yuen @ Jan 18 2020, Madison, WI
	    
	    # vx,vy,vz: Vector field
	    # nstep: The number of steps required 
	    # dt: user-defined time step
	    # periodic:: whether the inputted velocity field is periodic.
	    
	    
	    nx,ny,nz=size(vx);
	    # Make sure nstep is odd
	    mod(nstep,2)==0 ? nstep+=1 : nstep+=0;
	    # Make sure nstep has at least 9 steps
	    (nstep>=13) ? nstep+=0 : nstep=13;
	    
	    centerstep=div(nstep,2)+1;
	    x=zeros(nx,ny,nz,nstep);
	    y=zeros(nx,ny,nz,nstep);
	    curvature=zeros(nx,ny,nz);
	    torsion=zeros(nx,ny,nz);
	    
	    
	    # Caution: KH is using Interpolations v0.3.6
	    # Newer version of Interpolations has a syntax change!
	    if (periodic>0)
	        vxx = interpolate(vx,BSpline(Quadratic(Periodic())), OnCell())
	        vyy = interpolate(vy,BSpline(Quadratic(Periodic())), OnCell())   
	        vzz = interpolate(vz,BSpline(Quadratic(Periodic())), OnCell())   
	       else
	         # Natural boundary condition is used
	        vxx = interpolate(vx,BSpline(Quadratic(Line())), OnCell());
	        vyy = interpolate(vy,BSpline(Quadratic(Line())), OnCell());
	        vzz = interpolate(vz,BSpline(Quadratic(Line())), OnCell());
	    end
	    
	    
	    # a limiter to avoid skipping the cells
	    # from python.matplotlib.pyplot.streamplot
	    maxerror = 0.003
	    
	    for i in 1:nx, j in 1:ny
	        x[i,j,k,centerstep]=i;
	        y[i,j,k,centerstep]=j;
	        z[i,j,k,centerstep]=k;
	        dt=min(dt0,L/nx,L/ny,L/nz,0.1);
	        for nn in 1:div(nstep,2)
	            if (RK4)
	                # Forward integration
	                x0=x[i,j,k,nn-1+centerstep]
	                y0=y[i,j,k,nn-1+centerstep]
	                z0=z[i,j,k,nn-1+centerstep]
	                dx1=dt*vxx[x0,y0,z0];
	                dy1=dt*vyy[x0,y0,z0];
	                dz1=dt*vzz[x0,y0,z0];
	                dx2=dt*vxx[x0+dx1/2,y0+dy1/2,z0+dz1/2];
	                dy2=dt*vyy[x0+dx1/2,y0+dy1/2,z0+dz1/2];
	                dz2=dt*vzz[x0+dx1/2,y0+dy1/2,z0+dz1/2];
	                dx3=dt*vxx[x0+dx2/2,y0+dy2/2,z0+dz2/2];
	                dy3=dt*vyy[x0+dx2/2,y0+dy2/2,z0+dz2/2];
	                dz3=dt*vzz[x0+dx2/2,y0+dy2/2,z0+dz2/2];
	                dx4=dt*vxx[x0+dx3,y0+dy3,z0+dz3];
	                dy4=dt*vyy[x0+dx3,y0+dy3,z0+dz3];
	                dz4=dt*vzz[x0+dx3,y0+dy3,z0+dz3];
	                x[i,j,k,nn+centerstep]=x0+dt/6*(dx1+2*dx2+2*dx3+dx4)
	                y[i,j,k,nn+centerstep]=y0+dt/6*(dy1+2*dy2+2*dy3+dy4)
	                z[i,j,k,nn+centerstep]=z0+dt/6*(dz1+2*dz2+2*dz3+dz4)
	                # Backward integration
	                x0=x[i,j,k,centerstep-nn+1]
	                y0=y[i,j,k,centerstep-nn+1]
	                z0=z[i,j,k,centerstep-nn+1]
	                dx1=dt*vxx[x0,y0,z0];
	                dy1=dt*vyy[x0,y0,z0];
	                dz1=dt*vzz[x0,y0,z0];
	                dx2=dt*vxx[x0+dx1/2,y0+dy1/2,z0+dz1/2];
	                dy2=dt*vyy[x0+dx1/2,y0+dy1/2,z0+dz1/2];
	                dz2=dt*vzz[x0+dx1/2,y0+dy1/2,z0+dz1/2];
	                dx3=dt*vxx[x0+dx2/2,y0+dy2/2,z0+dz2/2];
	                dy3=dt*vyy[x0+dx2/2,y0+dy2/2,z0+dz2/2];
	                dz3=dt*vzz[x0+dx2/2,y0+dy2/2,z0+dz2/2];
	                dx4=dt*vxx[x0+dx3,y0+dy3,z0+dz3];
	                dy4=dt*vyy[x0+dx3,y0+dy3,z0+dz3];
	                dz4=dt*vzz[x0+dx3,y0+dy3,z0+dz3];
	                x[i,j,k,centerstep-nn]=x0+dt/6*(dx1+2*dx2+2*dx3+dx4)
	                y[i,j,k,centerstep-nn]=y0+dt/6*(dy1+2*dy2+2*dy3+dy4)
	                z[i,j,k,centerstep-nn]=z0+dt/6*(dz1+2*dz2+2*dz3+dz4)
	            else
	                # Forward integration  
	                ii=x[i,j,k,nn-1+centerstep]+dt/2*vxx[x[i,j,k,nn-1+centerstep],y[i,j,k,nn-1+centerstep],z[i,j,kznn-1+centerstep]];
	                jj=y[i,j,k,nn-1+centerstep]+dt/2*vyy[x[i,j,k,nn-1+centerstep],y[i,j,k,nn-1+centerstep],z[i,j,kznn-1+centerstep]];
	                kk=y[i,j,k,nn-1+centerstep]+dt/2*vzz[x[i,j,k,nn-1+centerstep],y[i,j,k,nn-1+centerstep],z[i,j,kznn-1+centerstep]];
	                x[i,j,k,nn+centerstep]=x[i,j,k,nn-1+centerstep]+dt/2*vxx[ii,jj,kk];
	                y[i,j,k,nn+centerstep]=y[i,j,k,nn-1+centerstep]+dt/2*vyy[ii,jj,kk];
	                z[i,j,k,nn+centerstep]=z[i,j,k,nn-1+centerstep]+dt/2*vzz[ii,jj,kk];
	                # Backward integration  
	                ii=x[i,j,k,centerstep-nn+1]-dt/2*vxx[x[i,j,k,centerstep-nn+1],y[i,j,k,centerstep-nn+1]];
	                jj=y[i,j,k,centerstep-nn+1]-dt/2*vyy[x[i,j,k,centerstep-nn+1],y[i,j,k,centerstep-nn+1]];
	                kk=y[i,j,k,centerstep-nn+1]-dt/2*vzz[x[i,j,k,centerstep-nn+1],y[i,j,k,centerstep-nn+1]];
	                x[i,j,k,centerstep-nn]=x[i,j,k,centerstep-nn+1]-dt/2*vxx[ii,jj,kk];
	                y[i,j,k,centerstep-nn]=y[i,j,k,centerstep-nn+1]-dt/2*vyy[ii,jj,kk];
	                z[i,j,k,centerstep-nn]=z[i,j,k,centerstep-nn+1]-dt/2*vzz[ii,jj,kk];
	            end
	        end       
	        
	        # KH (Jan 18 2020)
	        # Remember
	        #    d[T]   [ 0  k  0][T]
	        #   --[N] = [-k  0  t][N]
	        #   dt[B]   [ 0 -t  0][B]

	        # So we have
	        # curvature k = |dT/dt|
	        # torsion   t = |dN/dt+kT| or t = |dN/dt x T|
	        
	        Tx=zeros(nstep-4);
	        Ty=zeros(nstep-4);
	        Tz=zeros(nstep-4);
	        Nx=zeros(nstep-8);
	        Ny=zeros(nstep-8);
	        Nz=zeros(nstep-8);
	        
	        # Tangent 
	        centerstepx=div(nstep-4,2)+1;
	        for tt in centerstep-2:centerstep+2
	            Tx[tt-(centerstep-2)+1] = 1/(12*dt)*(-x[i,j,z,tt+2]+8*x[i,j,z,tt+1]-8*x[i,j,z,tt-1]+x[i,j,z,tt-2]);
	            Ty[tt-(centerstep-2)+1] = 1/(12*dt)*(-y[i,j,z,tt+2]+8*y[i,j,z,tt+1]-8*y[i,j,z,tt-1]+y[i,j,z,tt-2]);
	            Tz[tt-(centerstep-2)+1] = 1/(12*dt)*(-z[i,j,z,tt+2]+8*z[i,j,z,tt+1]-8*z[i,j,z,tt-1]+z[i,j,z,tt-2]);
	            TT=sqrt(Tx[tt-(centerstep-2)+1]^2+Ty[tt-(centerstep-2)+1]^2+Tz[tt-(centerstep-2)+1]^2);
	            Tx[tt-(centerstep-2)+1]=Tx[tt-(centerstep-2)+1]/TT;
	            Ty[tt-(centerstep-2)+1]=Ty[tt-(centerstep-2)+1]/TT;
	            Tz[tt-(centerstep-2)+1]=Tz[tt-(centerstep-2)+1]/TT;
	        end 
	        # Normal 
	        for tt in centerstep-4:centerstep+4
	            Nx[tt-(centerstep-4)+1] = 1/(12*dt)*(-Tx[tt+2]+8*Tx[tt+1]-8*Tx[tt-1]+Tx[tt-2]);
	            Ny[tt-(centerstep-4)+1] = 1/(12*dt)*(-Ty[tt+2]+8*Ty[tt+1]-8*Ty[tt-1]+Ty[tt-2]);
	            Nz[tt-(centerstep-4)+1] = 1/(12*dt)*(-Tz[tt+2]+8*Tz[tt+1]-8*Tz[tt-1]+Tz[tt-2]);
	            NN=sqrt(Nx[tt-(centerstep-4)+1]^2+Ny[tt-(centerstep-4)+1]^2+Nz[tt-(centerstep-4)+1]^2);
	            Nx[tt-(centerstep-4)+1]=Nx[tt-(centerstep-4)+1]/NN;
	            Ny[tt-(centerstep-4)+1]=Ny[tt-(centerstep-4)+1]/NN;
	            Nz[tt-(centerstep-4)+1]=Nz[tt-(centerstep-4)+1]/NN;
	        end
	        # Derivative of T
	        # dT/dt = kN
	        Txx = 1/(12*dt)*(-Tx[centerstepx+2]+8*Tx[centerstepx+1]-8*Tx[centerstepx-1]+Tx[centerstepx-2]);
	        Tyy = 1/(12*dt)*(-Ty[centerstepx+2]+8*Ty[centerstepx+1]-8*Ty[centerstepx-1]+Ty[centerstepx-2]);
	        Tzz = 1/(12*dt)*(-Tz[centerstepx+2]+8*Tz[centerstepx+1]-8*Tz[centerstepx-1]+Tz[centerstepx-2]);
	        curvature[i,j,k]=sqrt(Txx.^2.0.+Tyy.^2.0.+Tzz.^2.0);
	        # Derivative of N
	        # dN/dt = -kT+tB
	        Nxx = 1/(12*dt)*(-Nx[centerstepx]+8*Nx[centerstepx-1]-8*Nx[centerstepx-3]+Nx[centerstepx-4]);
	        Nyy = 1/(12*dt)*(-Ny[centerstepx]+8*Ny[centerstepx-1]-8*Ny[centerstepx-3]+Ny[centerstepx-4]);
	        Nzz = 1/(12*dt)*(-Nz[centerstepx]+8*Nz[centerstepx-1]-8*Nz[centerstepx-3]+Nz[centerstepx-4]);
	        
	        Bx=Nxx+curvature[i,j,k]*Tx[centerstepx]
	        By=Nyy+curvature[i,j,k]*Ty[centerstepx]
	        Bz=Nzz+curvature[i,j,k]*Tz[centerstepx]
	        torsion[i,j,k]=sqrt(Bx.^2.0.+By.^2.0.+Bz.^2.0)
	        
	    end
	    return x,y,z,curvature,torsion
	end

	function structure_curvature(A::Mat;period=0)
		#==
			Input
				A binary matrix A
		==#
		nx,ny=size(A);
		dt0=min(1e-3,1/nx);
		Ax,Ay=sobel_conv_2d(A);
		AA=sqrt.(Ax.^2.0+Ay.^2.0);
		Ax=Ax./AA;
		Ay=Ay./AA;
		# Regulate the gradient vectors
		Ax[isinf.(Ay)].=0;
		Ay[isinf.(Ay)].=0;
		# curvature computation
		x,y,curvature=RK2_curvature(A,-Ay,Ax,9,dt0,period);
		return curvature
	end

	function gradient_amplitude(I::Mat;l=10)
		#==
		l: The lengthscale of the cube (unit: pc)
		the gradient amplitude is normalized by its mean
		==#
		nx,ny=size(I);
		II=I./mean(I);
		dx=l/nx;
		dy=l/ny;
		Ix=(circshift(II,[-1,0]).+circshift(II,[1,0]).-2.0.*I)./dx/2;
		Iy=(circshift(II,[0,-1]).+circshift(II,[0,1]).-2.0.*I)./dy/2;
		Ia=sqrt.(Ix.^2.0.+Iy.^2.0);
		return Ia
	end

	# building blocks: direcrional derivative


	function dd(B::Cube;dir=1,ord=1,dx=1)
	    if (ord==1)
	        if (dir==1)
	            return (circshift(B,[1,0,0]).-B)./dx
	        elseif (dir==2)
	            return (circshift(B,[0,1,0]).-B)./dx
	        elseif (dir==3)
	            return (circshift(B,[0,0,1]).-B)./dx
	        end
	    elseif (ord==2)
	        if (dir==1)
	            return (circshift(B,[1,0,0]).-circshift(B,[-1,0,0]))./dx./2
	        elseif (dir==2)
	            return (circshift(B,[0,1,0]).-circshift(B,[0,-1,0]))./dx./2
	        elseif (dir==3)
	            return (circshift(B,[0,0,1]).-circshift(B,[0,0,-1]))./dx./2
	        end
	    end
	end

	function directional_derivative(Bx::Cube,By::Cube,Bz::Cube;order=1,dx=1)
	    Bxx=Bx.*dd(Bx,dir=1,ord=order,dx=dx).+Bx.*dd(By,dir=1,ord=order,dx=dx)+Bx.*dd(Bz,dir=1,ord=order,dx=dx)
	    Byy=By.*dd(Bx,dir=2,ord=order,dx=dx).+By.*dd(By,dir=2,ord=order,dx=dx)+By.*dd(Bz,dir=2,ord=order,dx=dx)
	    Bzz=Bz.*dd(Bx,dir=3,ord=order,dx=dx).+Bz.*dd(By,dir=3,ord=order,dx=dx)+Bz.*dd(Bz,dir=3,ord=order,dx=dx)
	    return Bxx,Byy,Bzz
	end

	function cross_product(Ax::Cube,Ay::Cube,Az::Cube,Bx::Cube,By::Cube,Bz::Cube)
	    Cx=Ay.*Bz.-Az.*By;
	    Cy=Az.*Bx.-Ax.*Bz;
	    Cz=Ax.*By.-Ay.*Bx;
	    return Cx,Cy,Cz
	end

	function amp(Ax::Cube,Ay::Cube,Az::Cube)
	    return sqrt.(Ax.^2.0.+Ay.^2.0.+Az.^2.0);
	end

	function directional_derivative(Ax::Cube,Ay::Cube,Az::Cube,Bx::Cube,By::Cube,Bz::Cube;order=1,dx=1)
	    Bxx=Ax.*dd(Bx,dir=1,ord=order,dx=dx).+Ax.*dd(By,dir=1,ord=order,dx=dx)+Ax.*dd(Bz,dir=1,ord=order,dx=dx)
	    Byy=Ay.*dd(Bx,dir=2,ord=order,dx=dx).+Ay.*dd(By,dir=2,ord=order,dx=dx)+Ay.*dd(Bz,dir=2,ord=order,dx=dx)
	    Bzz=Az.*dd(Bx,dir=3,ord=order,dx=dx).+Az.*dd(By,dir=3,ord=order,dx=dx)+Az.*dd(Bz,dir=3,ord=order,dx=dx)
	    return Bxx,Byy,Bzz
	end

	function dot_product(Ax::Cube,Ay::Cube,Az::Cube,Bx::Cube,By::Cube,Bz::Cube)
	    return Ax.*Bx.+Ay.*By.+Az.*Bz
	end

	function covariant_derivative(Ax::Cube,Ay::Cube,Az::Cube,Bx::Cube,By::Cube,Bz::Cube;order=1,dx=1)
	    Axx,Ayy,Azz=directional_derivative(Ax,Ay,Az,order=order,dx=dx);
	    A=dot_product(Ax,Ay,Az,Axx,Ayy,Azz);
	    Bxx,Byy,Bzz=directional_derivative(Ax,Ay,Az,Bx,By,Bz,order=order,dx=dx);
	    return Bxx.-A.*Ax,Byy.-A.*Ay,Bzz.-A.*Az;
	end


	function grad(A::Cube;order=1,dx=1)
	    Ax=dd(A,dir=1,ord=order,dx=dx)
	    Ay=dd(A,dir=2,ord=order,dx=dx)
	    Az=dd(A,dir=3,ord=order,dx=dx)
   	    return Ax,Ay,Az
	end
	


end # module LazGA