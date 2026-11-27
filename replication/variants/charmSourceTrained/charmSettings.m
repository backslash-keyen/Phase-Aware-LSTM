function s = charmSettings()
%CHARMSETTINGS  Which CHARM the variant charmSourceTrained builds (read by replication/variants/charmCommon).
s = struct('name', 'charmSourceTrained', 'layer', 'base', 'inputNorm', false, 'sourceTrained', true);
end
