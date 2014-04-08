function precision = precisionAtN(rankedIndices, trueLabels, N)

numCorrect = 0;

for i = 1:N
    if( trueLabels(rankedIndices(i)) == 1 )
        numCorrect = numCorrect + 1;
    end
end

precision = numCorrect / N;