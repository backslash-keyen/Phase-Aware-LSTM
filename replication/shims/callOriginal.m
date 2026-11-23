function varargout = callOriginal(fname, varargin)
%CALLORIGINAL  Call the repository's own version of a shadowed function.
% The shims in this folder share names with original functions. This
% temporarily takes the shim folder off the path, calls the original,
% and puts the shim folder back on top.
shimDir = fileparts(mfilename('fullpath'));
rmpath(shimDir);
restore = onCleanup(@() addpath(shimDir, '-begin'));
[varargout{1:nargout}] = feval(fname, varargin{:});
end
