lag_lim = 1;             % lag limit (in seconds)
tr = .005;                  % The sampling rate, here 2.4 seconds between frames of fMRI data
lags = -10:tr:10;          % UPDATE BASED ON HOW MANY SHIFTS YOU WANT
nframes = 1001;          % Number of time points in the dataset
data = [wernicke; broca];
ds = data.';                % Must be in dimensions time x voxels
s = 0:1000;
figure(1);
plot(s, wernicke, 'b', s, broca, 'r', 'LineWidth',2);
Cov = lagged_cov(ds(:,1), ds(:,2), max(lags));
figure(2);
plot(Cov, ':*g', 'LineWidth', 2);


% normalize based on entire run
for k = 1:numel(lags)
Cov(:,k) = Cov(:,k)/(nframes - abs(lags(k)));
end
figure(3);
plot(Cov, '--oc', 'LineWidth', 2);

% Parabolic interpolation to get peak lag/correlation
[pl,pc] = parabolic_interp(Cov,tr);

function  r = lagged_cov(ts1,ts2,L)
	% a signal time x ROI
	% L - number of TR shifts in each direction    

    if size(ts1) ~= size(ts2)
        error('Data matrices must be same dimensions for lagged_cov_lite.m');
    end
    
    data_length = (size(ts1,2)*(size(ts1,2)+1))/2;
    r = single(zeros(data_length,2*L+1));
    k = 1;
	
    for i = -L:L
		tau = abs(i);

        if i >= 0
            ts1_lagged = ts1(1:end-tau,:);
            ts2_lagged = ts2(1+tau:end,:);
        else
            ts1_lagged = ts1(1+tau:end,:);
            ts2_lagged = ts2(1:end-tau,:);
        end
        
        r(:,k) = vec_mat_flip(ts1_lagged'*ts2_lagged,'TD',1);
        
        k = k+1;
    end
end

function array_out = vec_mat_flip(array_in,type,dgnl)

% Transforms a(n) (anti-) symmetric matrix to a vector consisting of upper triangular without diagonal, or vice versa
% Useful for saving memory and limiting matrix comparisons to unique, non-trivial elements
 
% INPUTS:
% array_in (single/double) = input matrix/vector
% type (char) = 'TD' for time delay or 'ZL' for zero-lag FC (only needed
%                 for going from vector to matrix)
% dgnl (int) = 1 to keep diagonal (mat to vec) or indicates that diagonal
%                is represented (vec to mat). dgnl = 0 by default

% OUTPUT:
% array_out = vector or square matrix, depending on input

array_in = squeeze(array_in);
if nargin<3
    dgnl = 0;
end

% Matrix to vector
if size(array_in,1) == size(array_in,2)
    if dgnl
        array_out = array_in(triu(true(size(array_in))));
    else
        array_out = array_in(triu(true(size(array_in)),1));
    end

% Vector to matrix
elseif size(array_in,1) == 1 || size(array_in,2) == 1
    mat_sz = roots([.5,-.5,-length(array_in)]);
    mat_sz = single(mat_sz(mat_sz > 0));
    
    if dgnl
        mat_sz = mat_sz-1;
    end
    
    inds_mat = zeros(mat_sz,mat_sz);
    for i = 1:length(inds_mat)
        start = (i-1) * length(inds_mat) + 1;
        stop = start + length(inds_mat) - 1;
        inds_mat(i,:) = start:stop;
    end
    array_out = zeros(size(inds_mat));
    
    if dgnl
        array_out(inds_mat==triu(inds_mat)) = array_in;
    else
        array_out(inds_mat==triu(inds_mat,1)) = array_in;
    end
    
    if ischar(type)
        if strcmpi(type,'TD')
            array_out = array_out - array_out';
        elseif strcmpi(type,'ZL')
            array_out = array_out + array_out';
        else
            error('"type" must be ''TD'' or ''ZL''')
        end
    else
        error('"type" must be a type char')
    end
    
    if dgnl
        array_out = array_out - diag(diag(array_out))/2;
    elseif strcmpi(type,'ZL')
        array_out(logical(eye(size(array_out)))) = 1;
    end
else
    error('Input must be vector or square matrix!');
end
end
