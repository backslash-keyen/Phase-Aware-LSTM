function [sha, dirty] = gitCommit(projectLoc)
%GITCOMMIT  Commit of the code that produced a run (empty if git is unavailable).
sha = ''; dirty = false;
[st, out] = system(sprintf('git -C "%s" rev-parse HEAD', projectLoc));
if st == 0, sha = strtrim(out); end
[st, out] = system(sprintf('git -C "%s" status --porcelain --untracked-files=no', projectLoc));
if st == 0, dirty = ~isempty(strtrim(out)); end
end
