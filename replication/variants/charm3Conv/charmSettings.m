function s = charmSettings()
%CHARMSETTINGS  Which CHARM the variant charm3Conv builds (read by replication/variants/charmCommon).
s = struct('name', 'charm3Conv', 'layer', '3conv', 'inputNorm', false, 'sourceTrained', false);
end
