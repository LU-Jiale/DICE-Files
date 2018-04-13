function [Rot,trans] = trarot(model,planelist)
    %   estimate rotation
      M = zeros(3,3);
      D = zeros(3,3);
      for i = 1 : 3
        n = model(i,1:3);
        M(:,i) = n';
        d = planelist(i,1:3);
        D(:,i) = d';
      end
      [U,DG,V] = svd(D*M');
      Rot = U*V';

    % estimate translation
    L = zeros(3,3);
    N = zeros(3,1);
    for i = 1 : 3
        d = planelist(i,1:3);
        dt = d';
        n = planelist(i,1:3);    
        b = -n'*planelist(i,4);

        a = [0; 0; -( model(i,4)/ model(i,3))];
        %     a(1) = model(pairs(i,1),1,1);
        %     a(2) = model(pairs(i,1),1,2);
        %     a(3) = model(pairs(i,1),1,3);
        ra = Rot*a;
        L = L + dt*d;
        N = N + (dt*d)*(ra-b);
    end
    trans = -inv(L)*N;
end
