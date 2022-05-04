module LazTsallis
using PyCall, Images, HDF5, Statistics, StatsBase;
using LazCore, LazType
using LsqFit
using Base.Threads

	##############################################################################
	#
	# Copyright (c) 2020
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
	 KH: This is developed in v1.3
	==#


	function Tsallis_Function(Deltaf,p)
	    # p[1] = A
	    # p[2] = q
	    # p[3] = w
	    return real.(p[1].*Complex.(1.0.+(p[2]-1).*Deltaf.^2.0./p[3].^2).^(-1/(p[2]-1)))
	end

	function Tsallis_Fitting(d::Mat;dims=1,lag=10,a_guess=Float64(length(d)),q_guess=5.0,w_guess=1.0)
		# Gradients (lags)
		if (dims==1)
			gradd=circshift(d,[lag,0]).-d;
		elseif (dims==2)
			gradd=circshift(d,[0,lag]).-d;
		end

		# Normaliza to z-score
		# last pargraph of Sec 2 of EL10
		gradd_z = (gradd.-mean(gradd))./std(gradd)

		# 400 bins roughly
		gradd_histogram=fit(Histogram,gradd_z[:],-2:0.01:2)
		x=Array(-2:0.01:2);
		xx=0.5.*(x[1:end-1].+x[2:end])
		yy=gradd_histogram.weights;

		fit_gradd_histogram=curve_fit(Tsallis_Function,xx,yy,[a_guess,q_guess,w_guess])


		A=fit_gradd_histogram.param[1];
		q=fit_gradd_histogram.param[2];
		w=fit_gradd_histogram.param[3];

		return xx,yy,A,q,w
		    
	end

	function Tsallis_Fitting_QU(Q::Mat,U::Mat;lag=10,a_guess=Float64(length(Q)),q_guess=5.0,w_guess=1.0)
		# Gradients (lags)
		gradQ2=(circshift(Q,[lag,0]).-Q).^2.0 .+ (circshift(Q,[0,lag]).-Q).^2.0;
		gradU2=(circshift(U,[lag,0]).-U).^2.0 .+ (circshift(U,[0,lag]).-U).^2.0;

		gradP = sqrt.(gradQ2.+gradU2)
		# Normaliza to z-score
		# last pargraph of Sec 2 of EL10
		gradP_z = (gradP.-mean(gradP))./std(gradP)

		# 400 bins roughly
		gradd_histogram=fit(Histogram,gradP_z[:],-2:0.01:2)
		x=Array(-2:0.01:2);
		xx=0.5.*(x[1:end-1].+x[2:end])
		yy=gradd_histogram.weights;

		fit_gradd_histogram=curve_fit(Tsallis_Function,xx,yy,[a_guess,q_guess,w_guess])


		A=fit_gradd_histogram.param[1];
		q=fit_gradd_histogram.param[2];
		w=fit_gradd_histogram.param[3];

		return xx,yy,A,q,w
		    
	end


	function Tsallis_Fitting(d::Cube;dims=1,lag=10,a_guess=Float64(length(d)),q_guess=5.0,w_guess=1.0)
		# Gradients (lags)
		if (dims==1)
			gradd=circshift(d,[lag,0,0]).-d;
		elseif (dims==2)
			gradd=circshift(d,[0,lag,0]).-d;
		elseif (dims==3)
			gradd=circshift(d,[0,0,lag]).-d;
		end

		# Normaliza to z-score
		# last pargraph of Sec 2 of EL10
		gradd_z = (gradd.-mean(gradd))./std(gradd)

		# 400 bins roughly
		gradd_histogram=fit(Histogram,gradd_z[:],-2:0.01:2)
		x=Array(-2:0.01:2);
		xx=0.5.*(x[1:end-1].+x[2:end])
		yy=gradd_histogram.weights;

		fit_gradd_histogram=curve_fit(Tsallis_Function,xx,yy,[a_guess,q_guess,w_guess])


		A=fit_gradd_histogram.param[1];
		q=fit_gradd_histogram.param[2];
		w=fit_gradd_histogram.param[3];

		return xx,yy,A,q,w
		    
	end


end # module LazTsallis