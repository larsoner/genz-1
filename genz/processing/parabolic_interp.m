%Copyright (c) 2015 Washington University 
%Created by: Anish Mitra
%

%Washington University hereby grants to you a non-transferable, non-exclusive, royalty-free, 
%non-commercial, research license to use and copy the computer code that may be downloaded within 
%this site (the "Software").  You agree to include this license and the above copyright notice in 
%all copies of the Software.  The Software may not be distributed, shared, or transferred to any third party.  
%This license does not grant any rights or licenses to any other patents, copyrights, or other forms of 
%intellectual property owned or controlled by Washington University.  
%If interested in obtaining a commercial license, please contact Washington University's Office of Technology 
%Management (otm@dom.wustl.edu).

%YOU AGREE THAT THE SOFTWARE PROVIDED HEREUNDER IS EXPERIMENTAL AND IS PROVIDED "AS IS", 
%WITHOUT ANY WARRANTY OF ANY KIND, EXPRESSED OR IMPLIED, INCLUDING WITHOUT LIMITATION WARRANTIES 
%OF MERCHANTABILITY OR FITNESS FOR ANY PARTICULAR PURPOSE, OR NON-INFRINGEMENT OF ANY THIRD-PARTY PATENT, 
%COPYRIGHT, OR ANY OTHER THIRD-PARTY RIGHT.  IN NO EVENT SHALL THE CREATORS OF THE SOFTWARE 
%OR WASHINGTON UNIVERSITY BE LIABLE FOR ANY DIRECT, INDIRECT, SPECIAL, OR CONSEQUENTIAL DAMAGES 
%ARISING OUT OF OR IN ANY WAY CONNECTED WITH THE SOFTWARE, THE USE OF THE SOFTWARE, OR THIS AGREEMENT, 
%WHETHER IN BREACH OF CONTRACT, TORT OR OTHERWISE, EVEN IF SUCH PARTY IS ADVISED OF THE POSSIBILITY OF SUCH 
%DAMAGES. 




%This function uses parabolic interpolation to find the lag using the
%extremum of the lagged cross correlation.

%The function first uses the
%sign of the cross correlation at zero to decide whether to find a maximum
%or minmum. Next, we look for the global max/min.

%lcc is the empirical lagged covariance curve, lags is a vector with the timepoints in each temporal direction (e.g. -8:2:8 for +/- 8 seconds with a 2 second TR). 
%I have set a boundary condition such that any lag greater than 5 seconds is recorded as a NaN-- this is based on our experience that giant lags tend to be noise. You can relax or abolish this 
%boundary condition if you like.

function [peak_lag,peak_cov] = parabolic_interp_lite(lcc,tr)
	s = size(lcc);
	peak_lag = single(nan(1,s(1)));
	peak_cov = peak_lag;
    
    lcc = lcc';
    
    % find index of extremum (max or min determined by sign at zero-lag)
	[~,I]= max(bsxfun(@times,lcc,sign(lcc(single(s(2)+1)/2,:))),[],1);
    
    % ensure extremum is not at an endpoint (this would preclude parabolic interpolation)
	use = I>1 & I<s(2);
	lcc = lcc(:,use);

    % place peaks at center
	x0 = I(use) - (s(2)+1)/2;

    % set up three-point ccf for interpolation (y1,y2,y3)
	i = sub2ind([size(lcc),sum(use)],I(use),1:sum(use));
	lcc = [lcc(i-1);lcc(i);lcc(i+1)];
    
    % fit parabola: tau = TR * (y1-y3) / (2*(y1-2y2+y3))
	b = (lcc(3,:) - lcc(1,:))/2;
    a = (lcc(1,:) + lcc(3,:) - 2*lcc(2,:))/2;
	peak_lag(use) =  (-b./(2*a));
    
    % construct parabola to get covariance (y = ax^2 + bx + c)
	peak_cov(use) = a.*(peak_lag(use).^2) + b.*peak_lag(use) + lcc(2,:);
    
    % put back TR information
	peak_lag(use) = (peak_lag(use) + x0)*tr;
	
end
