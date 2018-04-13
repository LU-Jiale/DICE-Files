%% Load the training data
clear
clc
box = load('cloud.mat');
load('R.mat');

%% Uncomment to load the test file
%box = load('assignment_1_test.mat');
%box = box.pcl_test;
% global model planelist planenorm 
xyz_cutting = [0.25,0.25,0.25];
% display the points as a point cloud and as an image
model = zeros(4,3);
linePoint=zeros(3,1);
planenorm = zeros(3,3);
modelNum = 0;
point_pre=[];
point_fuse = [];
rgb_fuse = [];
corner1 = [];
corner2 = [];
s=[];

% display the points as a point cloud and as an image
model = zeros(4,3);
linePoint=zeros(3,1);
planenorm = zeros(3,3);

list = [31,30,28,27,24,22,21,18,15,12,9];
Rot=[R3031;R2830;R2728;R2427;R1518;R1215];
Tran=[T3031;T2830;T2728;T2427;T1518;T1215]*200.0;
previous_frame = 0;
point1 = box.cloud31;
point2 = box.cloud30;
point3 = box.cloud28;
point4 = box.cloud27;
point5 = box.cloud18;
point6 = box.cloud15;
point7 = box.cloud6;
list = {point1,point2,point3,point4,point5,point6,point7};
for frameNum = 1:4
    point = [list{frameNum}; point_fuse];
    [k,d]=size(point);
    rgb=zeros(k,d);
    %% fit plane
    figure(1778365)
    clf
    hold on
%     [k,d]=size(point_fuse);    
%     rgb=[rgb;zeros(k,d)];
    remaining = point;
%     remaining=[remaining;point_fuse];
    plot3(remaining(:,1),remaining(:,2),remaining(:,3),'k.')
    [NPts,W] = size(remaining);
    planelist = zeros(20,4);
    
%     find surface patches
%     here just get 4 first planes with more than 1000 data points 
%     Use normarlised rgb
    rgb_remaining = uint8((double(rgb)./sum(rgb,2)) * 255);
    planeNum = 0;
    planepointList = {};
    rgbpointList = {};
    while length(remaining) > 200
        % select a random small surface patch
        [oldlist, oldrgblist, plane] = select_patch(remaining, rgb_remaining,160);
        if(isempty(plane))
            break;
        end       
        planeNum = planeNum + 1;
        
        stillgrowing = 1;        
        while stillgrowing
            % find neighbouring points that lie in plane
            stillgrowing = 0;
            [newlist,rgb_list, remaining, rgb_remaining] = getallpoints(...
                plane,oldlist, oldrgblist,remaining,rgb_remaining, NPts);
            [NewL,W] = size(newlist);
            [OldL,W] = size(oldlist);
            figure(1778365)
            if planeNum == 1
                plot3(newlist(:,1),newlist(:,2),newlist(:,3),'r.')
                save1=newlist;
                rgbpointList{1}=rgb_list;
            elseif planeNum==2
                plot3(newlist(:,1),newlist(:,2),newlist(:,3),'b.')
                planepointList{2}=newlist;
                rgbpointList{2}=rgb_list;
            elseif planeNum == 3
                plot3(newlist(:,1),newlist(:,2),newlist(:,3),'g.')
                planepointList{3}=newlist;
                rgbpointList{3}=rgb_list;
            else
                plot3(newlist(:,1),newlist(:,2),newlist(:,3),'m.')
                planepointList{4}=newlist;
                rgbpointList{4}=rgb_list;
            end
            pause(2)
                
            if NewL > OldL + 50
                % refit plane
                [newplane,fit] = fitplane(newlist);
                [newplane',fit,NewL];
                planelist(planeNum,:) = newplane';
                if fit > 0.2*NewL     % bad fit - stop growing
                    break
                end
                stillgrowing = 1;
                oldlist = newlist;
                oldrgblist = rgb_list;
                plane = newplane;
            end
        end
        % delete the surface with less than 100 points
        if length(newlist) < 1000
            plot3(newlist(:,1),newlist(:,2),newlist(:,3),'k.')
            planelist(planeNum,:) = [];
            planeNum = planeNum - 1;        
        end
        pause(1)        
        ['**************** Segmentation Completed']     
    end
    
        % calculate mse
        q=planelist(1,:)/norm(planelist(1,1:3));
        xyz_new=zeros(length(save1),4);
        xyz_new(:,1:3)=save1;
        e=xyz_new*q';
        RMS=sqrt(mean(power(e,2)));
        s=[s,RMS];

    previous_frame = frameNum;
    point_fuse=[point_fuse; point];
    point_fuse = point_fuse*Rot(frameNum*3-2:frameNum*3,:)+Tran(frameNum,:);
    figure(110)
    plot3(point_fuse(:,1),point_fuse(:,2),point_fuse(:,3),'r.')
%     pause
end

    point_fuse = point_fuse*R2224 + T2224;
    point_fuse = point_fuse*R2122 + T2122;
    point_fuse = point_fuse*R1821 + T1821;
    figure(110)
    plot3(point_fuse(:,1),point_fuse(:,2),point_fuse(:,3),'r.')
    
for frameNum = 5:6
    point = list{frameNum};
    [k,d]=size(point);
    rgb=zeros(k,d);
    %% fit plane
    figure(1778365)
    clf
    hold on
%     [k,d]=size(point_fuse);    
%     rgb=[rgb;zeros(k,d)];
    remaining = point;
%     remaining=[remaining;point_fuse];
    plot3(remaining(:,1),remaining(:,2),remaining(:,3),'k.')
    [NPts,W] = size(remaining);
    planelist = zeros(20,4);
    
%     find surface patches
%     here just get 4 first planes with more than 1000 data points 
%     Use normarlised rgb
    rgb_remaining = uint8((double(rgb)./sum(rgb,2)) * 255);
    planeNum = 0;
    planepointList = {};
    rgbpointList = {};
    while length(remaining) > 200
        % select a random small surface patch
        [oldlist, oldrgblist, plane] = select_patch(remaining, rgb_remaining,160);
        if(isempty(plane))
            break;
        end       
        planeNum = planeNum + 1;
        
        stillgrowing = 1;        
        while stillgrowing
            % find neighbouring points that lie in plane
            stillgrowing = 0;
            [newlist,rgb_list, remaining, rgb_remaining] = getallpoints(...
                plane,oldlist, oldrgblist,remaining,rgb_remaining, NPts);
            [NewL,W] = size(newlist);
            [OldL,W] = size(oldlist);
            figure(1778365)
            if planeNum == 1
                plot3(newlist(:,1),newlist(:,2),newlist(:,3),'r.')
                save1=newlist;
                rgbpointList{1}=rgb_list;
            elseif planeNum==2
                plot3(newlist(:,1),newlist(:,2),newlist(:,3),'b.')
                planepointList{2}=newlist;
                rgbpointList{2}=rgb_list;
            elseif planeNum == 3
                plot3(newlist(:,1),newlist(:,2),newlist(:,3),'g.')
                planepointList{3}=newlist;
                rgbpointList{3}=rgb_list;
            else
                plot3(newlist(:,1),newlist(:,2),newlist(:,3),'m.')
                planepointList{4}=newlist;
                rgbpointList{4}=rgb_list;
            end
            pause(2)
                
            if NewL > OldL + 50
                % refit plane
                [newplane,fit] = fitplane(newlist);
                [newplane',fit,NewL];
                planelist(planeNum,:) = newplane';
                if fit > 0.2*NewL     % bad fit - stop growing
                    break
                end
                stillgrowing = 1;
                oldlist = newlist;
                oldrgblist = rgb_list;
                plane = newplane;
            end
        end
        % delete the surface with less than 100 points
        if length(newlist) < 1000
            plot3(newlist(:,1),newlist(:,2),newlist(:,3),'k.')
            planelist(planeNum,:) = [];
            planeNum = planeNum - 1;        
        end
        pause(1)        
        ['**************** Segmentation Completed']     
    end
    
        % calculate mse
        q=planelist(1,:)/norm(planelist(1,1:3));
        xyz_new=zeros(length(save1),4);
        xyz_new(:,1:3)=save1;
        e=xyz_new*q';
        RMS=sqrt(mean(power(e,2)));
        s=[s,RMS];

    previous_frame = frameNum;
    point_fuse=[point_fuse; point];
    point_fuse = point_fuse*Rot(frameNum*3-2:frameNum*3,:)+Tran(frameNum,:);
    figure(110)
    plot3(point_fuse(:,1),point_fuse(:,2),point_fuse(:,3),'r.')
%     pause
end
point_fuse = point_fuse*R2224 + T2224;
point_fuse = point_fuse*R2122 + T2122;
point_fuse = point_fuse*R1821 + T1821;
figure(110)
plot3(point_fuse(:,1),point_fuse(:,2),point_fuse(:,3),'r.')
    
% ;R912;R69;
