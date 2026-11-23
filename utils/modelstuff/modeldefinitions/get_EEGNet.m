function layers = get_EEGNet(model)

layers = [imageInputLayer([model.nchans model.signalLength],"Name","input","Normalization","none")
    convolution2dLayer([1 ceil(model.signalLength/4)],4,"Name","conv2d_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm1")
    groupedConvolution2dLayer([model.nchans 1],4,"channel-wise","Name","grouped_conv")
    batchNormalizationLayer("Name","batchnorm2")
    eluLayer(1,"Name","elu1")
    averagePooling2dLayer([1 4],"Name","avgpool1","Stride",[1 4])
    dropoutLayer(0.5,"Name","dropout1")
    convolution2dLayer([1 1],8,"Name","pointwise_conv2d","Padding","same")
    batchNormalizationLayer("Name","batchnorm3")
    eluLayer(1,"Name","elu2")
    averagePooling2dLayer([1 5],"Name","avgpool2","Stride",[1 5])
    dropoutLayer(0.5,"Name","dropout2")
    flattenLayer("Name","flatten")
    fullyConnectedLayer(model.nclass,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer()];

end 