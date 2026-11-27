function varargout = callOriginal_charm(fname, varargin)
%CALLORIGINAL_CHARM  Call the repository's own version of a function the CHARM variants shadow.
% Temporarily takes charmCommon off the path, calls the original, then puts it back on top.
d = fileparts(mfilename('fullpath'));
rmpath(d);
restore = onCleanup(@() addpath(d, '-begin'));
[varargout{1:nargout}] = feval(fname, varargin{:});
end
