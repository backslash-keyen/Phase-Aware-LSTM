function varargout = callOriginal_fastPhaseLSTM(fname, varargin)
%CALLORIGINAL_FASTPHASELSTM  Call the repository's own version of a shadowed function.
% Temporarily takes this variant folder off the path, calls the original,
% then puts the folder back on top.
d = fileparts(mfilename('fullpath'));
rmpath(d);
restore = onCleanup(@() addpath(d, '-begin'));
[varargout{1:nargout}] = feval(fname, varargin{:});
end
