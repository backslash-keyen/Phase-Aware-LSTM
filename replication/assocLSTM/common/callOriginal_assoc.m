function varargout = callOriginal_assoc(fname, varargin)
%CALLORIGINAL_ASSOC  Call the repository's own version of a function the al_ variants shadow.
% Temporarily takes this folder off the path, calls the original, then puts it back on top.
d = fileparts(mfilename('fullpath'));
rmpath(d);
restore = onCleanup(@() addpath(d, '-begin'));
[varargout{1:nargout}] = feval(fname, varargin{:});
end
