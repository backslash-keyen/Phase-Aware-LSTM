function varargout = callOriginal_ver5FastPhaseLSTM(fname, varargin)
%CALLORIGINAL_VER5FASTPHASELSTM  Call the repository's own version of a shadowed function.
% Temporarily takes this variant folder off the path, calls the original,
% then puts the folder back on top.
d = fileparts(mfilename('fullpath'));
rmpath(d);
restore = onCleanup(@() addpath(d, '-begin'));
[varargout{1:nargout}] = feval(fname, varargin{:});
end
