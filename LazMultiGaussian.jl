module LazMultiGaussian

using HDF5,PyPlot,Statistics,LsqFit,PyCall,FFTW,StatsBase,Images
using LazCore,LazType,LazThermal,LazIO,LazThermal_Kritsuk
using LazRHT_investigation,LazCFA


using LazABA # for the multigaussian definition.
using PyPlot # We will plot the figure


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

	export fit_multigaussian,decompose_multigaussian

	# The Multi Gaussian decomposition algorithm originally used for Snez's homework.

	# The simple Gaussian function is not explicitly exported, let us just put ir here.
	Gaus(x,p)=p[1]*exp.(.-(x.-p[2]).^2 .*p[3]);

	# Some constant, in SI for less confusion.

	kb_SI = 1.38064852e-23
	mh_SI = 1.6735575e-27
	kb_SI/mh_SI;

	function fit_multigaussian(
		fitxx, # velocity array
		fityy, # the spectral line array
		i::Number)
	    axxx1=subplot(111)
	    plot(fitxx,fityy,color="k")
	    fitxxx=Array(1:length(fityy))./length(fityy).-0.5
	    fityyy=(fityy.-minimum(fityy))./maximum(fityy)
	    fits=curve_fit(recursive_gaussian(i),fitxxx,fityyy,[ones(3*i).*0.5...])
	    vv2=(diff(fitxx)[1]/diff(fitxxx)[1])^2;
	    for j in 1:i
	        scatter(fitxx,minimum(fityy).+maximum(fityy).*Gaus(fitxxx,fits.param[j*3-2:j*3]),color=([j/i,0,1-j/i]),s=3)
	        text(0.55,0.90-0.05*j, L"$T_"*"$j"*L"=$"*string(round(Int,1/2/fits.param[j*3]*1000000/(kb_SI/mh_SI)*vv2))*"K",
	            color=([j/i,0,1-j/i]),fontsize=8,transform=axxx1.transAxes,horizontalalignment="left")
	    end
        xlabel("velocity (km/s)",fontsize=12)
        ylabel(L"$\sum_{HVC} T_B$ (K)",fontsize=12)
	    R2=1-sum((fits.resid).^2.0)./sum((fityyy.-mean(fityyy)).^2.0)
	    text(0.55,0.90,"Number of Gaussian=$i\nR2="*string(round(R2,digits=3)),color="k",fontsize=10,transform=axxx1.transAxes,horizontalalignment="left")
	    return fits
	end

	function decompose_multigaussian(d,n::Number)
		nnx,nny,nnv=size(d);
		nn=n*3
		spectralline_parameter=zeros(nnx,nny,n*3)


		for i in 1:nnx, j in 1:nny
		    fitxxx=Array(1:length(fitxx))./length(fitxx).-0.5;
		    fityyy=(d[i,j,:].-minimum(d[i,j,:]))./(maximum(d[i,j,:]).-minimum(d[i,j,:]));
		    fits=curve_fit(recursive_gaussian(3),fitxxx,fityyy,[ones(nn).*0.5...])
		    param=zeros(nn);
		    spectralline_parameter[i,j,:].=param;
		end

		# reconstruction of ppv under two gaussians

		dd=zeros(nnx,nny,nnv,n)
		for k in 1:n
			for i in 1:nnx, j in 1:nny
			    fitxxx=Array(1:length(fitxx))./length(fitxx).-0.5;
			    if (spectralline_parameter[i,j,3*k].>0)
			        dd[i,j,:,k].=minimum(d[i,j,:]).+maximum(d[i,j,:]).*Gaus(fitxxx,spectralline_parameter[i,j,(k-1)*3+1:k*3])
			    end
			end
		end
		return dd
	end

end # module LazMultiGaussian