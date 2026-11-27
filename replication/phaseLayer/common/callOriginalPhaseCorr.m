function varargout = callOriginalPhaseCorr(fname, varargin)
%CALLORIGINALPHASECORR  Call the repository's own version of a function shadowed by phaseLayer/common.
d = fileparts(mfilename('fullpath'));
rmpath(d);
restore = onCleanup(@() addpath(d, '-begin'));
[varargout{1:nargout}] = feval(fname, varargin{:});
end
