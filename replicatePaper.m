%REPLICATEPAPER  One-click replication of every table in the paper.
% Additive entry point for the replication; projecitonTL_mainScript.m is untouched.
% Set the options below, then run this script from anywhere.
%
% See replication/README.md for what each option does and what each table needs.

dryRun   = true;    % true: only check data and pretrained models, train nothing
seed     = 0;       % master seed recorded in the manifest

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'replication'));
runAll('dryRun', dryRun, 'seed', seed);
