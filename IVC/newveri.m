function  newveri(tr_img_pair)
fprintf('\nConv method\n')

num = size(tr_img_pair,1);
for i = 890:910
    temp1 = single(tr_img_pair{i,1})/255;
    temp2 = single(tr_img_pair{i,3})/255;
    
    leftEye = temp1(1:32,1:32);
    rightEye = temp1(1:32,33:64);
    mouthLeft = temp1(40:62,10:30);   
    mouthRight = temp1(40:62,34:54);   
    nose = temp1(15:45,20:44);
    
    temp_eL = (leftEye-temp2(1:32,1:32));
    temp_eR = normxcorr2(rightEye,temp2);
    temp_nose = normxcorr2(nose,temp2);
    temp_mouthL = normxcorr2(mouthLeft,temp2);
    temp_mouthR = normxcorr2(mouthRight,temp2);
    temp_face = normxcorr2(temp1,temp2);

    
    imshow(temp_eL);
    pause(1);  
   
 
end




