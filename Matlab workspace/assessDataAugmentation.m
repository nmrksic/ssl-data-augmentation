function [ResultsBayes, ResultsSEM, ResultsRocSVM, entitySetExpansionQuality] = assessDataAugmentation(TrainFM, TrainLabels, TestFM, TestLabels, kkt_threshold, max_iter, dist_bias, bidirection_pull, useBayes, useSEM, useRocSVM)

% N.B: Should get full single, not sparse matrices as input. From a memory
% perspective, this is necessarilly better, as we'll have to create the full 
% versions for SVMs anyway.

% bias means how to change the ratio of Positives to Negatives. 
% For example, dist_bias = 2 means that P:N ratio in bias will be twice
% that of the trainset. 0.5 means the opposite.

% the following matrix will contain the performances pre/post-augmentation:
ResultsSVM = zeros(13,3); % 1-3 reduxes, 4 original, 5,6,7 first redux with three different augmentations, 8,9,10 second, 11, 12, 13 third.

newNegBayes = []; % to make sure that they're empty unless we specifically want bidirectional expansion!
newNegSEM = [];
newNegRocSVM = [];

disp('starting algorithm')

SVMCLassifer = svmtrain(full(TrainFM), TrainLabels ,'kktviolationlevel', kkt_threshold, 'options', statset('MaxIter', max_iter, 'Display', 'off' ));
Results = svmclassify(SVMCLassifer, full(TestFM));
ResultsSVM(4,:) = quality(Results, TestLabels);

% Now, we need to create redux sets, with their labels, respective unlabeled set - its feature matrix, and its (true) labels to assess quality of entity set expansion (aug2).

%LabelsPrecision = zeros(1, length(TrainLabels));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sample = 0.1; % first reduced (redux) training set

[reduxFM, reduxLabels, reduxAugLabels, reduxAugLabelsNegative] = createRedux(TrainFM, TrainLabels, sample, dist_bias);

[newPosBayes, newPosSEM, newPosRocSVM] = augmentData(TrainFM, reduxAugLabels, LabelToArray(reduxAugLabelsNegative), useBayes, useSEM, useRocSVM, kkt_threshold, max_iter);

disp('size of newposbayes')
size(newPosBayes)
disp('tag')


disp('size of newpossem')
size(newPosSEM)
disp('tag')

if(bidirection_pull==1)
    
    [newNegBayes, newNegSEM, newNegRocSVM] = augmentData(TrainFM, reduxAugLabelsNegative, 1:sum(reduxAugLabels), useBayes, useSEM, useRocSVM, kkt_threshold, max_iter);

    interBayes = intersect(newPosBayes, newNegBayes);
    interSem = intersect(newPosSEM, newNegSEM);
    interRocSVM = intersect(newPosRocSVM, newNegRocSVM);

    newPosBayes = setdiff(newPosBayes, interBayes);
    newNegBayes = setdiff(newNegBayes, interBayes);
    newPosSEM = setdiff(newPosSEM, interSem);
    newNegSEM = setdiff(newNegSEM, interSem);
    newPosRocSVM = setdiff(newPosRocSVM, interRocSVM);
    newNegRocSVM = setdiff(newNegRocSVM, interRocSVM);
    
end

LabelsPrecision = TrainLabels - reduxAugLabels; % just leave the non-remaining positives.

%disp('augmented redux10')

SVMClassifierRedux10 = svmtrain(full(reduxFM), reduxLabels,'kktviolationlevel', kkt_threshold, 'options', statset('MaxIter', max_iter, 'Display', 'off' ));

Results = svmclassify(SVMClassifierRedux10, TestFM);

ResultsSVM(1,:) = quality(Results, TestLabels);

clear SVMClassifierRedux10;

% Now, we are to use all the new positives to retrain, and reassess!


% Bayes:

if(useBayes)

    LabelsPrecNew = zeros(1, length(TrainLabels));
    LabelsPrecNew(newPosBayes) = 1;
    entitySetExpansionQuality(1, :) = quality(LabelsPrecNew, LabelsPrecision);

    reduxx = zeros(size(reduxFM, 1) + length(newPosBayes) + length(newNegBayes), size(reduxFM, 2), 'single');
    reduxx(1:size(reduxFM, 1), :) = reduxFM;
    reduxx(size(reduxFM,1) + 1 : size(reduxFM,1) + length(newPosBayes), :) = TrainFM(newPosBayes, :);
    if(not(isempty(newNegBayes)))
        reduxx( size(reduxFM,1) + length(newPosBayes) + 1 : size(reduxx, 1), :) = TrainFM(newNegBayes, :);
    end

    reduxxLabels = ones(1, size(reduxx, 1));   
    reduxxLabels(size(reduxFM,1) + length(newPosBayes) + 1 : size(reduxx, 1)) = 0;
    reduxxLabels(1:size(reduxFM, 1)) = reduxLabels;

 %   disp('bayes10 test set made')

    SVMCLassifer = svmtrain(full(reduxx), reduxxLabels, 'kktviolationlevel', kkt_threshold, 'options', statset('MaxIter', max_iter, 'Display', 'off' ));
    Results = svmclassify(SVMCLassifer, full(TestFM));
    ResultsSVM(5,:) = quality(Results, TestLabels);

%    disp('bayes redux10 done')

    clear SVMCLassifer;
    clear reduxx;
end

% SEM:
if(useSEM)
    
    LabelsPrecNew = zeros(1, length(TrainLabels));
    LabelsPrecNew(newPosSEM) = 1;
    entitySetExpansionQuality(2, :) = quality(LabelsPrecNew, LabelsPrecision);

    reduxx = zeros(size(reduxFM, 1) + length(newPosSEM) + length(newNegSEM), size(reduxFM, 2), 'single');
    reduxx(1:size(reduxFM, 1), :) = reduxFM;
    reduxx(size(reduxFM,1) + 1 : size(reduxFM,1) + length(newPosSEM), :) = TrainFM(newPosSEM, :);
    if(not(isempty(newNegSEM)))
        reduxx( size(reduxFM,1) + length(newPosSEM) + 1 : size(reduxx, 1), :) = TrainFM(newNegSEM, :);
    end
    reduxxLabels = ones(1, size(reduxx, 1));   
    reduxxLabels(size(reduxFM,1) + length(newPosSEM) + 1 : size(reduxx, 1)) = 0;
    reduxxLabels(1:size(reduxFM, 1)) = reduxLabels;

   % disp('sem10 test set made')

    lenLabels = size(reduxxLabels)
    sumLabels = sum(reduxxLabels)

    SVMCLassifer = svmtrain(full(reduxx), reduxxLabels, 'kktviolationlevel', kkt_threshold, 'options', statset('MaxIter', max_iter, 'Display', 'off' ));
    Results = svmclassify(SVMCLassifer, full(TestFM));
    ResultsSVM(6,:) = quality(Results, TestLabels);

    clear SVMCLassifer;
    clear reduxx;

   % disp('sem redux10 done')

end


% RocSVM:

if(useRocSVM)

    LabelsPrecNew = zeros(1, length(TrainLabels));
    LabelsPrecNew(newPosRocSVM) = 1;
    entitySetExpansionQuality(3, :) = quality(LabelsPrecNew, LabelsPrecision);

    reduxx = zeros(size(reduxFM, 1) + length(newPosRocSVM) + length(newNegRocSVM), size(reduxFM, 2), 'single');
    reduxx(1:size(reduxFM, 1), :) = reduxFM;
    reduxx(size(reduxFM,1) + 1 : size(reduxFM,1) + length(newPosRocSVM), :) = TrainFM(newPosRocSVM, :);
    if(not(isempty(newNegRocSVM)))
        reduxx( size(reduxFM,1) + length(newPosRocSVM) + 1 : size(reduxx, 1), :) = TrainFM(newNegRocSVM, :);
    end
    reduxxLabels = ones(1, size(reduxx, 1));   
    reduxxLabels(size(reduxFM,1) + length(newPosRocSVM) + 1 : size(reduxx, 1)) = 0;
    reduxxLabels(1:size(reduxFM, 1)) = reduxLabels;

    %disp('rocSVM10 test set made')

    lenLabels = size(reduxxLabels)
    sumLabels = sum(reduxxLabels)

    SVMCLassifer = svmtrain(full(reduxx), reduxxLabels, 'kktviolationlevel', kkt_threshold, 'options', statset('MaxIter', max_iter, 'Display', 'off' ));
    Results = svmclassify(SVMCLassifer, full(TestFM));
    ResultsSVM(7,:) = quality(Results, TestLabels);

    clear SVMCLassifer;
    clear reduxx;

    %disp('rocsvm redux10 done')

end


clear reduxFM;
clear reduxAugLabels;
clear reduxLabels;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('starting redux33')

sample = 0.33;

[reduxFM, reduxLabels, reduxAugLabels, reduxAugLabelsNegative] = createRedux(TrainFM, TrainLabels, sample, dist_bias);

[newPosBayes, newPosSEM, newPosRocSVM] = augmentData(TrainFM, reduxAugLabels, LabelToArray(reduxAugLabelsNegative), useBayes, useSEM, useRocSVM, kkt_threshold, max_iter);

disp('size of newposbayes')
size(newPosBayes)
disp('tag')

disp('size of newpossem')
size(newPosSEM)
disp('tag')

if(bidirection_pull==1)
    
    [newNegBayes, newNegSEM, newNegRocSVM] = augmentData(TrainFM, reduxAugLabelsNegative, 1:sum(reduxAugLabels),  useBayes, useSEM, useRocSVM, kkt_threshold, max_iter);

    interBayes = intersect(newPosBayes, newNegBayes);
    interSem = intersect(newPosSEM, newNegSEM);
    interRocSVM = intersect(newPosRocSVM, newNegRocSVM);

    newPosBayes = setdiff(newPosBayes, interBayes);
    newNegBayes = setdiff(newNegBayes, interBayes);
    newPosSEM = setdiff(newPosSEM, interSem);
    newNegSEM = setdiff(newNegSEM, interSem);
    newPosRocSVM = setdiff(newPosRocSVM, interRocSVM);
    newNegRocSVM = setdiff(newNegRocSVM, interRocSVM);

end

%disp('augmented redux33')

SVMClassifierRedux = svmtrain(full(reduxFM), reduxLabels, 'kktviolationlevel', kkt_threshold, 'options', statset('MaxIter', max_iter, 'Display', 'off' ));

Results = svmclassify(SVMClassifierRedux, TestFM);

ResultsSVM(2,:) = quality(Results, TestLabels);

LabelsPrecision = TrainLabels - reduxAugLabels; % just leave the non-remaining positives.

clear SVMClassifierRedux33;

%disp('svm redux33 done')

% Bayes:
 if(useBayes)
 
    LabelsPrecNew = zeros(1, length(TrainLabels));
    LabelsPrecNew(newPosBayes) = 1;
    entitySetExpansionQuality(4, :) = quality(LabelsPrecNew, LabelsPrecision);

    reduxx = zeros(size(reduxFM, 1) + length(newPosBayes) + length(newNegBayes), size(reduxFM, 2), 'single');
    reduxx(1:size(reduxFM, 1), :) = reduxFM;
    reduxx(size(reduxFM,1) + 1 : size(reduxFM,1) + length(newPosBayes), :) = TrainFM(newPosBayes, :);
    if(not(isempty(newNegBayes)))
        reduxx( size(reduxFM,1) + length(newPosBayes) + 1 : size(reduxx, 1), :) = TrainFM(newNegBayes, :);
    end

    reduxxLabels = ones(1, size(reduxx, 1));   
    reduxxLabels(size(reduxFM,1) + length(newPosBayes) + 1 : size(reduxx, 1)) = 0;
    reduxxLabels(1:size(reduxFM, 1)) = reduxLabels;

   % disp('bayes set33 made')

    SVMCLassifer = svmtrain(reduxx, reduxxLabels, 'kktviolationlevel', kkt_threshold, 'options', statset('MaxIter', max_iter, 'Display', 'off' ));
    Results = svmclassify(SVMCLassifer, TestFM);
    ResultsSVM(8,:) = quality(Results, TestLabels);

    clear SVMCLassifer;
    clear reduxx;
    ResultsSVM(8,:)
    % disp('bayes svm33 done')
 
 end

% SEM:

if(useSEM)

    LabelsPrecNew = zeros(1, length(TrainLabels));
    LabelsPrecNew(newPosSEM) = 1;
    entitySetExpansionQuality(5, :) = quality(LabelsPrecNew, LabelsPrecision);

    reduxx = zeros(size(reduxFM, 1) + length(newPosSEM) + length(newNegSEM), size(reduxFM, 2), 'single');
    reduxx(1:size(reduxFM, 1), :) = reduxFM;
    reduxx(size(reduxFM,1) + 1 : size(reduxFM,1) + length(newPosSEM), :) = TrainFM(newPosSEM, :);
    if(not(isempty(newNegSEM)))
        reduxx( size(reduxFM,1) + length(newPosSEM) + 1 : size(reduxx, 1), :) = TrainFM(newNegSEM, :);
    end

    reduxxLabels = ones(1, size(reduxx, 1));   
    reduxxLabels(size(reduxFM,1) + length(newPosSEM) + 1 : size(reduxx, 1)) = 0;
    reduxxLabels(1:size(reduxFM, 1)) = reduxLabels;

  %  disp('sem sem33 set done')

    SVMCLassifer = svmtrain(full(reduxx), reduxxLabels, 'kktviolationlevel', kkt_threshold, 'options', statset('MaxIter', max_iter, 'Display', 'off' ));
    Results = svmclassify(SVMCLassifer, full(TestFM));
    ResultsSVM(9,:) = quality(Results, TestLabels);

    clear SVMCLassifer;
    clear reduxx;

    
   % disp('sem33 svm done')

end


% RocSVM:

if(useRocSVM)
    
    LabelsPrecNew = zeros(1, length(TrainLabels));
    LabelsPrecNew(newPosRocSVM) = 1;
    entitySetExpansionQuality(6, :) = quality(LabelsPrecNew, LabelsPrecision);

    reduxx = zeros(size(reduxFM, 1) + length(newPosRocSVM) + length(newNegRocSVM), size(reduxFM, 2), 'single');
    reduxx(1:size(reduxFM, 1), :) = reduxFM;
    reduxx(size(reduxFM,1) + 1 : size(reduxFM,1) + length(newPosRocSVM), :) = TrainFM(newPosRocSVM, :);
    if(not(isempty(newNegRocSVM)))
        reduxx( size(reduxFM,1) + length(newPosRocSVM) + 1 : size(reduxx, 1), :) = TrainFM(newNegRocSVM, :);
    end

    reduxxLabels = ones(1, size(reduxx, 1));   
    reduxxLabels(size(reduxFM,1) + length(newPosRocSVM) + 1 : size(reduxx, 1)) = 0;
    reduxxLabels(1:size(reduxFM, 1)) = reduxLabels;

   % disp('rocSVM33 test set made')

    SVMCLassifer = svmtrain(full(reduxx), reduxxLabels, 'kktviolationlevel', kkt_threshold, 'options', statset('MaxIter', max_iter, 'Display', 'off' ));
    Results = svmclassify(SVMCLassifer, full(TestFM));
    ResultsSVM(10,:) = quality(Results, TestLabels);

    clear SVMCLassifer;
    clear reduxx;

    % disp('rocSVM redux33 done')

end

clear reduxFM;
clear reduxAugLabels;
clear reduxLabels;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sample = 0.8;

[reduxFM, reduxLabels, reduxAugLabels, reduxAugLabelsNegative] = createRedux(TrainFM, TrainLabels, sample, dist_bias);

[newPosBayes, newPosSEM, newPosRocSVM] = augmentData(TrainFM, reduxAugLabels, LabelToArray(reduxAugLabelsNegative), useBayes, useSEM, useRocSVM, kkt_threshold, max_iter);

disp('size of newposbayes')
size(newPosBayes)
disp('tag')

disp('size of newpossem')
size(newPosSEM)
disp('tag')

if(bidirection_pull==1)
    
    [newNegBayes, newNegSEM, newNegRocSVM] = augmentData(TrainFM, reduxAugLabelsNegative, 1:sum(reduxAugLabels), useBayes, useSEM, useRocSVM, kkt_threshold, max_iter);

    interBayes = intersect(newPosBayes, newNegBayes);
    interSem = intersect(newPosSEM, newNegSEM);
    interRocSVM = intersect(newPosRocSVM, newNegRocSVM);

    newPosBayes = setdiff(newPosBayes, interBayes);
    newNegBayes = setdiff(newNegBayes, interBayes);
    newPosSEM = setdiff(newPosSEM, interSem);
    newNegSEM = setdiff(newNegSEM, interSem);
    newPosRocSVM = setdiff(newPosRocSVM, interRocSVM);
    newNegRocSVM = setdiff(newNegRocSVM, interRocSVM);
end

% disp('augmentation80 done')

SVMClassifierRedux = svmtrain(reduxFM, reduxLabels, 'kktviolationlevel', kkt_threshold, 'options', statset('MaxIter', max_iter, 'Display', 'off' ));

Results = svmclassify(SVMClassifierRedux, TestFM);

ResultsSVM(3,:) = quality(Results, TestLabels);

LabelsPrecision = TrainLabels - reduxAugLabels; % just leave the non-remaining positives.

%disp('redux80 svm done')

clear SVMClassifierRedux;

% Bayes:
if(useBayes)
    
    LabelsPrecNew = zeros(1, length(TrainLabels));
    LabelsPrecNew(newPosBayes) = 1;
    entitySetExpansionQuality(7, :) = quality(LabelsPrecNew, LabelsPrecision);

    reduxx = zeros(size(reduxFM, 1) + length(newPosBayes) + length(newNegBayes), size(reduxFM, 2), 'single');
    reduxx(1:size(reduxFM, 1), :) = reduxFM;
    reduxx(size(reduxFM,1) + 1 : size(reduxFM,1) + length(newPosBayes), :) = TrainFM(newPosBayes, :);
    if(not(isempty(newNegBayes)))
        reduxx( size(reduxFM,1) + length(newPosBayes) + 1 : size(reduxx, 1), :) = TrainFM(newNegBayes, :);
    end

    reduxxLabels = ones(1, size(reduxx, 1));   
    reduxxLabels(size(reduxFM,1) + length(newPosBayes) + 1 : size(reduxx, 1)) = 0;
    reduxxLabels(1:size(reduxFM, 1)) = reduxLabels;

    % disp('bayes80 test set made')
    
    SVMCLassifer = svmtrain(full(reduxx), reduxxLabels, 'kktviolationlevel', kkt_threshold, 'options', statset('MaxIter', max_iter, 'Display', 'off' ));

    Results = svmclassify(SVMCLassifer, full(TestFM));
    ResultsSVM(11,:) = quality(Results, TestLabels);

    clear SVMCLassifer;
    clear reduxx;

    % disp('bayes svm done')

end

% SEM:
if(useSEM)
    
    LabelsPrecNew = zeros(1, length(TrainLabels));
    LabelsPrecNew(newPosSEM) = 1;
    entitySetExpansionQuality(8, :) = quality(LabelsPrecNew, LabelsPrecision);

    reduxx = zeros(size(reduxFM, 1) + length(newPosSEM) + length(newNegSEM), size(reduxFM, 2), 'single');
    reduxx(1:size(reduxFM, 1), :) = reduxFM;
    reduxx(size(reduxFM,1) + 1 : size(reduxFM,1) + length(newPosSEM), :) = TrainFM(newPosSEM, :);

    if(not(isempty(newNegSEM)))
        reduxx( size(reduxFM,1) + length(newPosSEM) + 1 : size(reduxx, 1), :) = TrainFM(newNegSEM, :);
    end

    reduxxLabels = ones(1, size(reduxx, 1));   
    reduxxLabels(size(reduxFM,1) + length(newPosSEM) + 1 : size(reduxx, 1)) = 0;
    reduxxLabels(1:size(reduxFM, 1)) = reduxLabels;

    size(newPosSEM);

    % disp('sem80 set made')

    SVMCLassifer = svmtrain(full(reduxx), reduxxLabels, 'kktviolationlevel', kkt_threshold, 'options', statset('MaxIter', max_iter, 'Display', 'off' ));
    Results = svmclassify(SVMCLassifer, full(TestFM));
    ResultsSVM(12,:) = quality(Results, TestLabels);

    clear SVMCLassifer;
    clear reduxx;

    % disp('sem80 svm done')

end
    

% RocSVM:

if(useRocSVM)

    LabelsPrecNew = zeros(1, length(TrainLabels));
    LabelsPrecNew(newPosRocSVM) = 1;
    entitySetExpansionQuality(9, :) = quality(LabelsPrecNew, LabelsPrecision);

    %Augmented 3. redux:    
    reduxx = zeros(size(reduxFM, 1) + length(newPosRocSVM) + length(newNegRocSVM), size(reduxFM, 2), 'single');
    reduxx(1:size(reduxFM, 1), :) = reduxFM;
    reduxx(size(reduxFM,1) + 1 : size(reduxFM,1) + length(newPosRocSVM), :) = TrainFM(newPosRocSVM, :);
    if(not(isempty(newNegRocSVM)))
        reduxx( size(reduxFM,1) + length(newPosRocSVM) + 1 : size(reduxx, 1), :) = TrainFM(newNegRocSVM, :);
    end
    reduxxLabels = ones(1, size(reduxx, 1));   
    reduxxLabels(size(reduxFM,1) + length(newPosRocSVM) + 1 : size(reduxx, 1)) = 0;
    reduxxLabels(1:size(reduxFM, 1)) = reduxLabels;

    % disp('RocSVM80 test set made')

    SVMCLassifer = svmtrain(full(reduxx), reduxxLabels, 'kktviolationlevel', kkt_threshold, 'options', statset('MaxIter', max_iter, 'Display', 'off' ));
    Results = svmclassify(SVMCLassifer, full(TestFM));

    ResultsSVM(13,:) = quality(Results, TestLabels);

    clear SVMCLassifer;
    clear reduxx;

    % disp('RocSVM redux80 done')

end

clear reduxFM;
clear reduxAugLabels;
clear reduxLabels;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ResultsBayes = zeros(7, 3);

ResultsBayes(1,:) = ResultsSVM(1,:); % result of first redux
ResultsBayes(2, :) = ResultsSVM(5, :); % augmentation of first Bayes
ResultsBayes(3, :) = ResultsSVM(2, :); % second redux
ResultsBayes(4, :) = ResultsSVM(8, :); % augmentation of second redux
ResultsBayes(5, :) = ResultsSVM(3, :); % third redux's result
ResultsBayes(6, :) = ResultsSVM(11, :);
ResultsBayes(7, :) = ResultsSVM(4, :) % full train used


ResultsSEM = zeros(7,3);

ResultsSEM(1,:) = ResultsSVM(1,:); % result of first redux
ResultsSEM(2, :) = ResultsSVM(6, :); % augmentation of first SEM
ResultsSEM(3, :) = ResultsSVM(2, :); % second redux
ResultsSEM(4, :) = ResultsSVM(9, :); % augmentation of second redux
ResultsSEM(5, :) = ResultsSVM(3, :); % third redux's result
ResultsSEM(6, :) = ResultsSVM(12, :); % third aug
ResultsSEM(7, :) = ResultsSVM(4, :); % full train used


ResultsRocSVM = zeros(7,3);

ResultsRocSVM(1,:) = ResultsSVM(1,:); % result of first redux
ResultsRocSVM(2, :) = ResultsSVM(7, :); % augmentation of first SEM
ResultsRocSVM(3, :) = ResultsSVM(2, :); % second redux
ResultsRocSVM(4, :) = ResultsSVM(10, :); % augmentation of second redux
ResultsRocSVM(5, :) = ResultsSVM(3, :); % third redux's result
ResultsRocSVM(6, :) = ResultsSVM(13, :); % third aug
ResultsRocSVM(7, :) = ResultsSVM(4, :); % full train used