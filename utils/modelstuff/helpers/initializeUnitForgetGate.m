function B = initializeUnitForgetGate(nWeights,numHiddenUnits)
    B = zeros(nWeights*numHiddenUnits, 1);
    B(numHiddenUnits+1:2*numHiddenUnits) = 1; % Forget gate bias set to 1
end