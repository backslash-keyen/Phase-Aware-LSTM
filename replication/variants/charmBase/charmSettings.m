function s = charmSettings()
%CHARMSETTINGS  Which CHARM the variant charmBase builds (read by replication/variants/charmCommon).
s = struct('name', 'charmBase', 'layer', 'base', 'inputNorm', false, 'sourceTrained', false);
end
