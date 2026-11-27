function varargout = callOriginal_phased(fname, varargin)
%CALLORIGINAL_PHASED  Call the repository's own version of a function the Phased LSTM variants shadow.
% Temporarily takes phased/common off the path, calls the original, then puts it back on top.
d = fileparts(mfilename('fullpath'));
rmpath(d);
restore = onCleanup(@() addpath(d, '-begin'));
[varargout{1:nargout}] = feval(fname, varargin{:});
end
