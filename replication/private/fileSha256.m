function h = fileSha256(path)
%FILESHA256  SHA-256 of a file, via Java (no toolbox needed).
md  = java.security.MessageDigest.getInstance('SHA-256');
fid = fopen(path, 'r');
if fid < 0, h = ''; return; end
c = onCleanup(@() fclose(fid));
while true
    buf = fread(fid, 2^24, '*uint8');
    if isempty(buf), break; end
    md.update(buf);
end
d = typecast(md.digest(), 'uint8');
h = lower(reshape(dec2hex(d, 2)', 1, []));
end
