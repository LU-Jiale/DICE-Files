% find a candidate planar patch
% if not enough remaining points, returns empty fitlist
% if cannot find a good patch, returns empty fitlist
function [fitlist,rgblist, plane] = select_patch(points, RGB, residThre)

  [L,D] = size(points);
  tmpnew = zeros(L,3);
  tmprest = zeros(L,3);
  rgblist = zeros(L,3);
  rgbrest = zeros(L,3);

  % if not enough remaining points, return empty list
  if L < 200
    fitlist=[];
    plane=[];
    rgblist=[];
    return
  end

  % try different random points until a successful plane is found
  pointlist = randperm(L);
  success = 0;
  for n = 1 : L
    idx = pointlist(n);
    pnt = points(idx,:);
%     rgb_random = RGB(idx,:);
    % find points in the neighborhood of the given point
    DISTTOL = 5.0;
    fitcount = 0;
    restcount = 0;
    for i = 1 : L
      dist = norm(points(i,:) - pnt);
      if dist < DISTTOL
        fitcount = fitcount + 1;
        tmpnew(fitcount,:) = points(i,:);
        rgblist(fitcount,:) = RGB(i,:);
      else
        restcount = restcount + 1;
        tmprest(restcount,:) = points(i,:);
        rgbrest(restcount,:) = RGB(i,:);
      end
    end
    oldlist = tmprest(1:restcount,:);
    rgbrest = rgbrest(1:restcount,:);
    if fitcount > 100
      % fit a plane
      [plane,resid] = fitplane(tmpnew(1:fitcount,:));

      if resid < residThre
        fitlist = tmpnew(1:fitcount,:);
        rgblist = rgblist(1:fitcount,:);
        return
      end
    end
  end  
  fitlist=[];  % failed to find a plane
  plane=[];
