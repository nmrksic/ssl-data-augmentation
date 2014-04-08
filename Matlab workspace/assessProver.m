function [PreAugmentation, PostAugmentation, ceiling] = assessProver(TrainFM, TrainLabels, TestFM, TestLabels, kkdthreshold)

% assessMultiClass wants the unlabeled data FM as input as well. 
% it reconstructs the whole thing inside

samples = [0.05, 0.1, 0.2, 0.33, 0.5, 0.8, 0.5, 0.8];

PreAugmentation = zeros(6, 1);
PostAugmentation = zeros(6, 1);

for i = 1:6
    
    sample = samples(i);

    sampledEntries = randsample(1:size(TrainFM, 1), fix(sample * size(TrainFM, 1)));

    reduxFM = TrainFM(sampledEntries, :);
    reduxLabels = TrainLabels(sampledEntries); 

    sampler = zeros(1, size(TrainFM, 1));
    sampler(sampledEntries) = 1;
    sampler = inverseMatrix(sampler);
    sampler = LabelToArray(sampler);

    sampleUnlabeled = setdiff(1:size(TrainFM, 1), sampledEntries);

    isequal(sampler, sampleUnlabeled)

    UnlabeledFM = TrainFM( sampleUnlabeled , :);

    [A,B] = assessMultiClass( reduxFM, reduxLabels, TestFM, TestLabels, UnlabeledFM, kkdthreshold)

    PreAugmentation(i, 1) = B(1,3);
    PostAugmentation(i, 1) = A(1,3);
    
end

newLabelling = MultiClassSVM(TrainFM, TrainLabels, TestFM, 0);

ceiling = quality(newLabelling, TestLabels);
ceiling = ceiling(1,3);