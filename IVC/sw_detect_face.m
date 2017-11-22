function [ patches,bbox_location ] = sw_detect_face( real_image,window_size, scale, scale_end, stride)
% sw_multiscale_detect_face
% - This is a function to proposed the potential face images via moving the
% sliding window. 
%==========================================================================
% Output:
%   - patches: a cell to store every window_size proposed images. The size
%               of save images are H*W*N, where N is the number of sliding
%   - bbox_location: bounding box [x,y,height,width]
%--------------------------------------------------------------------------
% Input:
%   - real_image : The original images without resize
%   - window_size: The proposed sliding window size
%   - scale      : The scale of for each original image
%   - stride     : The steps between each save images
%==========================================================================

window_r = window_size(1);
window_c = window_size(2);
scale_count = 1;
% Multi-scale sliding window
for s = scale:0.1:scale_end
    resize_image = imresize(real_image, s);
    [irow, icol] = size(resize_image);

    rMax = round((irow-window_r+1) / stride - 0.5);
    cMax =  round((icol-window_c+1) / stride - 0.5);

    single_patches = zeros(window_r, window_c, rMax*cMax, 'uint8');
    single_bbox_location = zeros(rMax*cMax, 4, 'uint8');

    % Iteratively save the patches.

    count = 1;
    for i = 1:rMax
        for j = 1:cMax
            r = 1 + (i-1)*stride;
            c = 1 + (j-1)*stride;
            single_patches(:,:,count) = resize_image(r:r+window_r-1, c:c+window_c-1);
            single_bbox_location(count,:) = [round(r/s),round(c/s), ...
                round(window_r) / s, round(window_c / s)]; % top-left y,x, height, width
            count = count+1;
        end
    end
    patches{scale_count} = single_patches;
    bbox_location{scale_count} = single_bbox_location;
    scale_count = scale_count + 1;
end


end

