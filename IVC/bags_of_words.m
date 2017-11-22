function histImage = bags_of_words(image_samples, vocab)
fprintf('\nCompute bag of words\n');

[~, num_words] = size(vocab);
histImage = zeros(length(image_samples), num_words);

for i =1:length(image_samples)
    in_image = single(image_samples{i,1});
    
    histSingleImage = zeros(1,num_words);    
    [~, desc] = vl_dsift(in_image);
    [~, numOfFeatyres] = size(desc);
    D = vl_alldist2(vocab, single(desc));
    for j = 1:numOfFeatyres
        [~,index] = min(D(:,j));
        histSingleImage(index) = histSingleImage(index) + 1;
    end
    
    histImage(i,:) = histSingleImage / sum(histSingleImage);
end
end