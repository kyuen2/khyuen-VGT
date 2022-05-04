module LazNewCore

using HDF5,PyPlot,Statistics,LsqFit,PyCall,FFTW,StatsBase,Images
using LazCore,LazType,LazThermal,LazIO,LazThermal_Kritsuk
using LazRHT_investigation,LazCFA
using SpecialFunctions

using Base.Threads # CPU Parallelism

	##############################################################################
	#
	# Copyright (c) 2020
	# Ka Ho Yuen, Alex Lazarian and Dmitri Pogosyan
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
	 KH: This is v1.3+ only
	==#

	# A new set of functions due to YLP20.

	export union_pixels,mask,new_sba
	export idx2ij,idx2ijk
	export vonMises,recursive_vonMises,fit_vgt_multiple_vonMises,vonMisesba
	export sban2d_vonmise
	# testing functions
	export Sturges,angle_regulation,angle_disp,twodgaussian


	function idx2ij(idx::Number,nx::Number)
	    j=div(idx-1,nx)+1
	    i=mod1(idx-(j-1)*nx,nx);
	    return i,j
	end

	function idx2ijk(idx::Number,nx::Number)
	    k=div(idx-1,nx^2)+1
	    j=div(idx-(k-1)*nx^2-1,nx)+1;
	    i=mod1(idx-(k-1)*nx^2-(j-1)*nx,nx);
	    return i,j,k
	end

	function union_pixels(d::Cube,threshold::Number)
	    nx,ny,nv=size(d)
	    dd=similar(d);
	    dd[isnan.(d)].=0;
	    dd[d.<threshold].=0;
	    dd[d.>threshold].=1;
	    dx=ones(nx,ny);			  	
	    # Postunion map, preinitialized as all one;
	    for k in 1:nv, j in 1:ny, i in 1:nx
	        dx[i,j]=dx[i,j]*dd[i,j,k]
	    end
	    return dx
	end

	function mask(I::Mat,thres::Number,dn::Number)
	    nx,ny=size(I)
	    Imask=zeros(div(nx,dn),div(ny,dn))
	    for i in 1:div(nx,dn), j in 1:div(ny,dn)
	        is=(i-1)*dn+1;
	        ie=i*dn;
	        js=(j-1)*dn+1;
	        je=j*dn;
	        if (mean(I[is:ie,js:je])>thres)
	            Imask[i,j]=1;
	        end
	    end
	    return Imask            
	end

	function new_sba(d, block_size::Number;threed=true)
	    nx=size(d)[1]
	    ny=size(d)[2]
	    if (threed)
	        nv=size(d)[3]
	    else
	        nv=1;
	    end
	    d_Q=zeros(div(nx,block_size),div(ny,block_size));
	    d_U=zeros(div(nx,block_size),div(ny,block_size));
	    
	    for ii in 1:div(nx,block_size), jj in 1:div(ny,block_size)
	        if (threed)
	            for kv in 1:nv
	               is=(ii-1)*block_size+1;
	               ie=ii*block_size;
	               js=(jj-1)*block_size+1;
	               je=jj*block_size;
	               dx,dy=sobel_conv_2d(d[:,:,kv])
	               d_theta=cmean(dy[is:ie,js:je]./dx[is:ie,js:je])
	               dd=mean(d[is:ie,js:je,kv])
	               d_Q[ii,jj]+=dd.*cos.(2.0.*d_theta)
	               d_U[ii,jj]+=dd.*sin.(2.0.*d_theta)
	            end
	        else
	           is=(ii-1)*block_size+1;
	           ie=ii*block_size;
	           js=(jj-1)*block_size+1;
	           je=jj*block_size;
	           dx,dy=sobel_conv_2d(d[:,:])
	           d_theta=cmean(dy[is:ie,js:je]./dx[is:ie,js:je])
	           dd=mean(d[is:ie,js:je])
	           d_Q[ii,jj]=dd.*cos.(2.0.*d_theta)
	           d_U[ii,jj]=dd.*sin.(2.0.*d_theta) 
	        end
	    end
	    return d_Q,d_U        
	end


#==
    The periodic gradient function
==#
	function vonMises(
    x::Vec,       # Angles, in radian
    p::Vec        # Parameters
    )
	    #==
	    p[1] = a constant
	    p[2] = μ
	    p[3] = κ
	    notice 1/κ ~ σ^2
	    ==#
	    return p[1].*exp.(p[3].*cos.(x.-p[2]))./(2.0.*pi.*besseli(0,p[3]));
	end

	function recursive_vonMises(number_of_gaussian::Number);
	 vonMises(x,p)=p[1].*exp.(p[3].*cos.(x.-p[2]))./(2.0.*pi.*besseli(0,p[3]));
	 vonMisesnew(x,p)=p[1].*exp.(p[3].*cos.(x.-p[2]))./(2.0.*pi.*besseli(0,p[3]));
	 for i in 2:number_of_gaussian
	  vonMisesnew = let vonMisesold=vonMisesnew
	   (x,p)->vonMises(x,p[(i-1)*3+1:i*3]).+vonMisesold(x,p[1:(i-1)*3]);
	  end
	 end
	 return vonMisesnew
	end



	#==
	New Gradient Algorithm
	==#

	function fit_vgt_multiple_vonMises(
	    fitxx, # velocity array
	    fityy, # the spectral line array
	    i     # number of gaussian
	    )
	    axxx1=subplot(111, projection="polar")
	    plot(fitxx,fityy,color="k")
	    fitxxx=fitxx
	    fityyy=(fityy.-minimum(fityy))./maximum(fityy)
	    try 
	    	fits=curve_fit(recursive_vonMises(i),fitxx,fityy,[ones(3*i).*0.5...])
		    vv=(diff(fitxx)[1]/diff(fitxxx)[1]);
		    anglesarray=zeros(0)
		    for j in 1:i
		    scatter(fitxx,vonMises(fitxxx,fits.param[j*3-2:j*3]),color=([j/i,0,1-j/i]),s=3)
		        fits.param[3+(j-1)*3]>0 ? angles=mod(fits.param[2+(j-1)*3],2*pi) : angles=mod(fits.param[2+(j-1)*3]-pi,2*pi)
		    text(1.10,1.05-0.10*j, L"$\mu=$"*string(round(angles*180.0./pi,digits=2))
		        *"\n"*L"$\sigma=$"*string(round(180.0./pi./sqrt(abs(fits.param[3+(j-1)*3])),digits=2))
		        ,color=([j/i,0,1-j/i]),fontsize=8,transform=axxx1.transAxes,horizontalalignment="right")
		        axvline(angles,color=([j/i,0,1-j/i]),linestyle="--")
		        push!(anglesarray,angles)
		    end
		    xlabel(L"$\theta$",fontsize=12)
		    #ylabel("N",fontsize=12)
		    R2=1-sum((fits.resid).^2.0)./sum((fityy.-mean(fityy)).^2.0)
		    text(0.06,0.96,"R2="*string(round(R2,digits=3)),color="k",fontsize=10,transform=axxx1.transAxes,horizontalalignment="left")
		    axxx1.set_theta_zero_location("N")
		    axxx1.set_theta_direction(-1)
		    axxx1.set_yticklabels([])
		    #==
			Outputs are
			1. Gradient histogram weights
			2. angles, in radian
			3. angle dispersion, in radian
			4. Coefficient of Determination
		    ==#
		    return fits.param[1:3:end],anglesarray,1.0./sqrt.(abs.(fits.param[3:3:end])),R2
	    catch e
	    	A=0;
	    	B=0;
	    	C=0;
	    	D=0;
	    	if (i>1)
	    		A,B,C,D=fit_vgt_multiple_vonMises(fitxx,fityy,i-1)
	    	else
	    		println("The fitting has a problem even when i=1")
	    	end
	    	return A,B,C,D;
	    end
	end

	function twodgaussian(nx::Number,ny::Number,l_0::Number,h_0::Number,θ::Number)
	    A=zeros(nx,ny)
	    @threads for ii in 1:nx*ny
	    	i,j=idx2ij(ii,nx)
	        idx=i-div(nx,2)
	        jdx=j-div(ny,2)
	        A[i,j]=1.0./(2*pi).*exp(-idx^2.0/2/l_0^2.0-jdx^2.0/2/h_0^2.0)
	    end
	    A=rotate_2d(A,θ)
	    return Array(A)
	end

	function Sturges(kernelsize::Number)
		# The effective number of data is given by π(kernelsize)^2/4. 
		# Sturges' formula from wikipedia: https://en.wikipedia.org/wiki/Histogram#:~:text=The%20bins%20(intervals)%20must%20be,to%20display%20%22relative%22%20frequencies.
		N=round(Int,log(pi*kernelsize*kernelsize/4)/log(2))
		return max(N,30)
	end

	function angle_regulation(C,A;halfpolar=false)
		# C = count
		# A = angles, in radian
		if (halfpolar)
			# tangent regulation trick
			AA=atan.(tan.(A))
		else
			AA=A
		end
		return angle(sum(C.*exp.(im.*AA)))
	end

	function angle_disp(C,σ)
		# C = count
		# σ = angle dispersion, in radian
		return sqrt(sum(C.*σ.^2.0)./sum(C))
	end

	function sban2d_vonmise(V::Mat,blocksize::Number;halfpolar=false,number_of_gaussian=2)
		vx,vy=sobel_conv_2d(V)
		va=atan.(vy,vx)
		nx,ny=size(V)
		nnx=div(nx,blocksize)
		nny=div(ny,blocksize)
		va_vgt=zeros(nnx,nny);
		va_σ  =zeros(nnx,nny);
		va_R2 =zeros(nnx,nny);

		# the recursion begin here.
		for ii in 1:nnx, jj in 1:nny
			is=(ii-1)*blocksize+1
			ie=(ii-0)*blocksize+0
			js=(jj-1)*blocksize+1
			je=(jj-0)*blocksize+0
			vaa=va[is:ie,js:je]
			binnum=Sturges(blocksize)
			# No weight
			Xn=plt.hist(vaa[:].*180.0./pi,bins=binnum,label="CGW");

			# Bins and count
			XX=0.5.*(Xn[2][1:end-1]+Xn[2][2:end]);
			Xb=Xn[1];

			# Fit the hell out.
			C,A,σ,R2=fit_vgt_multiple_vonMises(XX,Xb.*pi./180,number_of_gaussian)

			# Sum with directions
			va_vgt[ii,jj]=angle_regulation(C,A;halfpolar=halfpolar)
			va_σ[ii,jj]  =angle_disp(C,σ)
			va_R2[ii,jj] =R2	
		end	
		return va_vgt,va_σ,va_R2
	end

	function vonMisesba(V::Mat,kernelsize::Number;halfpolar=false,number_of_gaussian=2)

		vx,vy=sobel_conv_2d(V)
		va=atan.(vy,vx)
		nx,ny=size(V)
		va_vgt=zeros(size(va))
		va_σ=zeros(size(va))
		va_R2=zeros(size(va))

		# This line can be initialized outside the loop
		G1=twodgaussian(nx,ny,kernelsize/2,kernelsize/2,0);

		# the recursion begin here.
		#@threads 
		for idx in 1:nx*ny
			i,j=idx2ij(idx,nx)


			G1X=circshift(G1,[div(nx,2)+1,div(nx,2)+1]);
			binnum=Sturges(kernelsize)
			# The below line is for periodc map
			Xn=plt.hist(circshift(va,[-div(kernelsize,2)-i,-div(kernelsize,2)-j])[:].*180.0./pi,bins=binnum,weights=G1X[:],label="CGW");

			# Bins and count
			XX=0.5.*(Xn[2][1:end-1]+Xn[2][2:end]);
			Xb=Xn[1];

			# Fit the hell out.
			C,A,σ,R2=fit_vgt_multiple_vonMises(XX.*pi./180,Xb,number_of_gaussian)

			# Sum with directions
			va_vgt[i,j]=angle_regulation(C,A;halfpolar=halfpolar)
			va_σ[i,j]  =angle_disp(C,σ)
			va_R2[i,j] =R2
		end
		return va_vgt,va_σ,va_R2
	end


end #module LazNewCore