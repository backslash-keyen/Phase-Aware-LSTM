function runAll(varargin)
%RUNALL  Whole replication in one call: train everything, then build tables.
%
%   runAll()                      % shipped code only
%   runAll('dryRun', true)        % plan + data/pretrained checks, no training
%
% Any option of runReplication is accepted. Steps:
%   1. runReplication  - source-domain experiments, then target-domain ones
%   2. collectResults  - per-fold metrics -> results/fold_metrics.csv
%   3. buildTables     - results/tables/table1..table10 (.csv .md .tex + stats)
% Rerunning skips experiments that are already done, so an interrupted run
% resumes where it stopped.
runReplication(varargin{:});
p = inputParser; p.KeepUnmatched = true;
addParameter(p, 'dryRun', false); addParameter(p, 'tables', {});
parse(p, varargin{:});
if p.Results.dryRun
    fprintf('[replication] dry run: no tables built.\n');
    return
end
buildTables('tables', p.Results.tables);
end
