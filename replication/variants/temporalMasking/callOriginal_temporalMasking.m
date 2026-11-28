function varargout = callOriginal_temporalMasking(fname, varargin)
%CALLORIGINAL_TEMPORALMASKING  Call the original (shadowed) function FNAME.
% Takes this variant folder off the path for the call, then puts it back.
d = fileparts(mfilename('fullpath'));
rmpath(d);
restore = onCleanup(@() addpath(d, '-begin'));
[varargout{1:nargout}] = feval(fname, varargin{:});
end
