%%
clc;clear;
box = load('assignment_1_box.mat');
box = box.pcl_train;

% display the points as a point cloud and as an image
%%
% extract a frame
frameNum = 15;
rgb = box{frameNum}.Color;     % Extracting the colour data
point = box{frameNum}.Location;     % Extracting the xyz data
r_point = point;
r_rgb = rgb;
for i1 = 1:length(point)
    % remove all points that are not near box
    if -0.9>point(i1,1) || point(i1,1)>-0.5
        r_point(i1,:) = [NaN NaN NaN];
        r_rgb(i1,:) = [0 0 0];
    end
    if -0.5>point(i1,2) || point(i1,2)>-0.1
        r_point(i1,:) = [NaN NaN NaN];
        r_rgb(i1,:) = [0 0 0];
    end
    if 0.6>point(i1,3) || point(i1,3)>1
        r_point(i1,:) = [NaN NaN NaN];
        r_rgb(i1,:) = [0 0 0];
    end
    % remove hand pixels
    if 35<rgb(i1,1) && rgb(i1,1)<140 && 10<rgb(i1,2) && rgb(i1,2)<100 && 0<rgb(i1,3) && rgb(i1,3)<85 
        r_point(i1,:) = [NaN NaN NaN];
        r_rgb(i1,:) = [0 0 0];
    end
end
pc = pointCloud(r_point, 'Color', r_rgb);     % Creating a point-cloud variable
% remove noise
denoise = pcdenoise(pc);
% remove NaN value
n_pc = removeInvalidPoints(denoise);
    % display the point cloud and corresponding image
%figure(frameNum)
%figure
%imag2d(r_rgb) % Shows the 2D images, NOTE: this image contains noise, once we remove the noise,
% the size of that point cloud changes, so that we cannot use imag2d()
%figure(100+frameNum)
figure
showPointCloud(n_pc) % show the denoised version
%figure
%showPointCloud(pc) % show the version without denoising
%pause

