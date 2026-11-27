function varargout = callOriginal_miHPCOptions(fname, varargin)
%CALLORIGINAL_MIHPCOPTIONS  Call the repository's own version of a shadowed function.
% Temporarily takes this variant folder off the path, calls the original,
% then puts the folder back on top. The ported Graz_EEGNet/Graz_Transformer
% files sit in this folder, so they are added back on a second path entry
% for the duration of the call.
d = fileparts(mfilename('fullpath'));
rmpath(d);
restore = onCleanup(@() addpath(d, '-begin'));
ported = fullfile(d, 'ported');
addpath(ported, '-end');
cleanPorted = onCleanup(@() rmpath(ported));
[varargout{1:nargout}] = feval(fname, varargin{:});
end
