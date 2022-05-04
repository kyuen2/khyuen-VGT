module LazRHT
using PyCall,FITSIO,LazType,Statistics,FFTW,LazPyWrapper,LinearAlgebra,Printf
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
	 rht.py must be in your local folder!
	 Please vist Susan Clark's homepage to obtain the rht.py file.
	==#

	export RHT_wrapper,USM_wrapper,BIT_wrapper
	export RHTs_wrapper,l,NCC,bitmap
	export USM_width,USM,USM_hist,indical_derivative

	function RHT_wrapper(A::Mat)
		f=FITS("a.fits","w")
		write(f,A);
		close(f)
		run(`python rht.py a.fits`);
		ff=FITS("a_kh00.fits");
		bs=65;
		Ax=read(ff[1]);
		hi=read(ff[2],"hi");
		hj=read(ff[2],"hj");
		htheta=read(ff[2],"hthets")
		close(ff);
		run(`rm a.fits a_kh00.fits`)
		return Ax[bs+1:end-bs,bs+1:end-bs],hi,hj,htheta;
	end

	function RHTs_wrapper(A::Mat,WLEN::Number,SMR::Number,THR::Number)
		aa=rand(1:999999)
		aaa=@sprintf("%6.6i",aa)
		f=FITS("a$aaa.fits","w")
		write(f,A);
		close(f)
		WLENS=string(WLEN);
		SMRS=string(SMR);
		THRS=string(THR);
		run(`python rht.py -w $WLENS -s $SMRS -t $THRS a$aaa.fits`);
		ff=FITS("a"*aaa*"_kh00.fits");
		af="a"*aaa*"_kh00.fits";
		bs=WLEN;
		Ax=read(ff[1]);
		hi=read(ff[2],"hi");
		hj=read(ff[2],"hj");
		htheta=read(ff[2],"hthets")
		close(ff);
		run(`rm a$aaa.fits $af `)
		return Ax[bs+1:end-bs,bs+1:end-bs],hi,hj,htheta;
	end

	function l(A::Mat,B::Mat)
        function t(A::Mat)
            return A.-mean(A)
        end
        # KH : get rid of these mistakes
        A[isnan.(A)].=0;
        B[isnan.(B)].=0;
        A[isinf.(A)].=0;
        B[isinf.(B)].=0;
        return sum(t(A).*t(B))./std(A)./std(B)./length(A)
    end

    function l(kT::Cube,Ms::Cube)
        return mean((kT.-mean(kT)).*(Ms.-mean(Ms)))/(std(kT)*std(Ms))
    end

    function NCC(A::Mat,B::Mat)
    	return l(A,B)
    end

	function umask(v::Mat,radius,periodic)
	dA=1;
	nx,ny=size(v);
	dx=dA^(1/2);
	Nx=2*nx;
	Ny=2*ny;
	G_r = zeros(Nx,Ny);
	# Here one can invert any kinds of function you want.
	count=0
	for i in 1:Nx
		for j in 1:Ny
			xx = (i-nx-1)*dx;
			yy = (j-ny-1)*dx;
			rr = sqrt(xx^2+yy^2);
			if (rr<radius)
				G_r[i,j] = 1
				count+=1;
			end
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
	rho_r[1:nx,1:ny]=v;
	if (periodic>0)
		rho_r[nx+1:Nx,1:ny]=v;
		rho_r[1:nx,ny+1:Ny]=v;
		rho_r[nx+1:Nx,ny+1:Ny]=v;
	end
	rho_k=fft(rho_r);
	phi_k=G_k.*rho_k.*dA;
	rho_r=0;
	rho_k=0;
	G_k=0;
	phi_r=ifft(phi_k);
	phi_k=0;
	phi_rr=real(phi_r[1:nx,1:ny]);
	phi_r=0;
	phi_rr=phi_rr./count;
	# Denoise algorithm
	phi_rr[abs.(phi_rr).<1e-9].=0
	return phi_rr;
	end

	function bitmap(A::Mat,threshold::Number);
		AA=zeros(size(A));
        AA[A.>threshold].=1;
        AA[A.<threshold].=0;
        return AA
    end
	
	function USM_wrapper(A::Mat)
		return bitmap(A-umask(A,15,1),0);
	end

	function USM(A::Mat,thres::Number)
		return bitmap(A-umask(A,thres,1),0)
	end

	function USM_hist(A::Mat)
		width=zeros(0);;
		ct=zeros(0);
		nx,ny=size(A)
		for i in 1:div(nx,2)
			push!(width,i)
			AA=USM(A,i);
			push!(ct,length(findall(AA.>0))./length(AA))
		end
		return width,ct
	end

	function indical_derivative(a::Vec,b::Vec)
		ad=0.5*(diff(a)[1:end-1] + diff(a)[2:end]);
		bd=0.5*(diff(b)[1:end-1] + diff(b)[2:end]);
		return ad./bd
	end

	function USM_width(A::Mat)
		tolerance = 1e-3
		w,c=USM_hist(A);
		logw=log.(w);
		logc=log.(c);
		dlcdlw=indical_derivative(logc,logw);
		return w[minimum(findall(abs.(dlcdlw).<tolerance))]/size(A)[1];
	end


	function clump_eigenfunctions(A::Mat)
		A_map,A_count=plabel(A);
		nx,ny=size(A_map);
		X,Y=meshgrid(1:ny,1:nx);
		A_weighted_width = zeros(0);
		A_weight = zeros(0);
		for i in 1:A_count
			An=zeros(size(A));
			An[A_map.==i].=1;
			An_x=sum(An.*X)/sum(An);
			An_y=sum(An.*Y)/sum(An);
			An_xx=sum(An.*(X-An_x).^2)/sum(An);
			An_yy=sum(An.*(Y-An_y).^2)/sum(An);
			An_xy=sum(An.*(X-An_x).*(Y-An_y))/sum(An);
			eigval=eigvals(reshape([An_xx,An_xy,An_xy,An_yy],2,2));
			push!(A_weighted_width,minimum(eigval));
			push!(A_weight,sum(An))
		end
		return sum(A_weighted_width.*A_weight)/sum(A_weight);
	end

	function BIT_wrapper(A::Mat)
		return bitmap(A,mean(A))
	end
	
end # module LazRHT