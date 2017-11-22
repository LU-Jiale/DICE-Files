%% Initialisation
%==========================================================================
% Add the path of used library.
% - The function of adding path of liblinear and vlfeat is included.
%==========================================================================
clear all
clc
run ICV_setup

% Hyperparameter of experiments
resize_size=[64 64];


%% Part II: Face Verification: 
%==========================================================================
% The aim of this task is to verify whether the two given people in the
% images are the same person. We train a binary classifier to predict
% whether these two people are actually the same person or not.
% - Extract the features
% - Get a data representation for training
% - Train the verifier and evaluate its performance
%==========================================================================
Xtr = [];
Xva = [];

disp('Verification:Extracting features..')

load('./data/face_verification/face_verification_va.mat')
load('./data/face_verification/face_verification_tr.mat')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loading the training data
% -tr_img_pair/va_img_pair:
% The data is store in a N-by-4 cell array. The first dimension of the cell
% array is the first cropped face images. The second dimension is the name 
% of the image. Similarly, the third dimension is another image and the
% fourth dimension is the name of that image.
% -Ytr/Yva: is the label of 'same' or 'different'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%dataNum1 = (length(tr_img_pair)-5);
dataNum1 = 100;
XtrNN = cell(1984*2, dataNum1);
YtrNN = cell(1,dataNum1);

% You should construct the features in here. (read, resize, extract)
parfor i =1:dataNum1
    tempgroup1 = [];
    for j = 1:2
        temp3 = vl_hog(single(tr_img_pair{i+j-1,1})/255, 8);
        temp4 = vl_hog(single(tr_img_pair{i+j-1,3})/255, 8);
        tempgroup1 = [tempgroup1 [temp3(:); temp4(:)]];
        
    end
    XtrNN(:,i) = num2cell(tempgroup1,2);
    YtrNN{i} = [Ytr(i), Ytr(i+1), Ytr(i+2),Ytr(i+3), Ytr(i+4)];
end
for i =1:1800
    temp1 = vl_hog(single(tr_img_pair{i,1})/255, 8);
    temp2 = vl_hog(single(tr_img_pair{i,3})/255, 8);
    Xtr = [Xtr; temp1(:)' - temp2(:)'];
end

%dataNum2 = (length(va_img_pair)-5);
dataNum2 =  50;
YvaNN = cell(1,dataNum2);
XvaNN = cell(1984*2,dataNum2);

parfor i =1:dataNum2
    tempgroup1 = [];
    for j = 1:5
        temp3 = vl_hog(single(va_img_pair{i+j-1,1})/255,8);
        temp4 = vl_hog(single(va_img_pair{i+j-1,3})/255,8);
        tempgroup1 = [tempgroup1 [temp3(:); temp4(:)]];
    end
    XvaNN(:,i) = num2cell(tempgroup1,2);
    YvaNN{i} = [Yva(i), Yva(i+1), Yva(i+2),Yva(i+3), Yva(i+4)];
end
for i =1:400
    temp1 = vl_hog(single(va_img_pair{i,1})/255, 8);
    temp2 = vl_hog(single(va_img_pair{i,3})/255, 8);
    Xva = [Xva; temp1(:)' - temp2(:)'];
end


randomindex = randperm(dataNum1);
XtrN = Xtr(randomindex,:);
XtrN = XtrN';
YtrN = Ytr(randomindex,:);
YtrN = ((YtrN +1 ) / 2)';
randomindex = randperm(dataNum2);
XvaN = Xva(randomindex,:);
XvaN = XvaN';
YvaN = Yva(randomindex,:);
YvaN = ((YvaN +1 ) / 2)';
%% Train neruen network
net=patternnet(4);
net = train(net,XtrNN, YtrNN,'useParallel', 'yes', 'useGPU', 'yes');
%% Evaluate
y = net(XvaN);
