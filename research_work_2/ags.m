function [xmin, n, c, rec] = ags(theta, p, a, c, eps, upsilon, m, xi)
    % otheta = theta;
    if nargin<6
        eps = 1.0e-08;
    end
    w = (3-sqrt(5))/2;
    fb = zeros(1, m);
    btheta = theta + (a+(c-a)*w)*p;
    for k=1:m
      [~, fb(k)] = runNet([xi(:, k); btheta; upsilon(k)]');
    end
    fb = mean(fb);
    fx = zeros(1, m);
    xtheta = theta + a+(c-a)*(1-w)*p;
    for k=1:m
      [~, fx(k)] = runNet([xi(:, k); xtheta; upsilon(k)]');
    end
    fx = mean(fx);
    atheta = theta + a*p;
    fa = zeros(1, m);
    for k=1:m
      [~, fa(k)] = runNet([xi(:, k); atheta; upsilon(k)]');
    end
    fa = mean(fa);
    ctheta = theta + c*p;
    fc = zeros(1,m);
    for k=1:m
      [~, fc(k)] = runNet([xi(:, k); ctheta; upsilon(k)]');
    end
    fc = mean(fc); 
    %if min(fb, fx) >= min(fc, fa)
    %   disp 'need to bracket an optimal solution'
    %   xmin = NaN;
    %   rec = [];
    %   return
    %end
    
    %this keeps track of number of iteration the code takes to determine
    %the interval of the golden section search.
    n = 1;
    while(min(fb, fx) >= min(fa, fc))
        if(fc > fa) 
            % Means c is too large and lets shrink the interval to make the
            % min of fb or fx greater than fa or fc.
            c = c/2;
        end
        if(fc <= fa)
            % Means c is too small and there will be divergence. Increasing
            % the interval gap will fix it.
            c = c*2;
        end
        btheta = theta + (a+(c-a)*w)*p;
        for k=1:m
          [~, fb(k)] = runNet([xi(:, k); btheta; upsilon(k)]');
        end
        fb = mean(fb);
        xtheta = theta + a+(c-a)*(1-w)*p;
        for k=1:m
          [~, fx(k)] = runNet([xi(:, k); xtheta; upsilon(k)]');
        end
        fx = mean(fx);
        fc = zeros(1,m);
        ctheta = theta + c*p;
        for k=1:m
         [~, fc(k)] = runNet([xi(:, k); ctheta; upsilon(k)]');
        end
        fc = mean(fc);
        if(n>300 || log10(c)>6 || log10(c)<-6)
            xmin = inf;
            rec = zeros(10, 4);
            return;
        end
%         rec_find_c(n, :) = [a, c, (a+(c-a)*w), a+(c-a)*(1-w), fc];
%         if(n>300)
%             xmin = inf;
%             rec = zeros(10, 4);
%             return;
%         end
        %each time c is modified, n is incremented to measure the
        %computational complexity
        n = n + 1;
    end
    rec = zeros(200, 4);
    runc = 1;
    while c-a>eps && runc <= 200
        rec(runc, :) = [a, c, fb, fx];
        if fb<fx
            c = a+(c-a)*(1-w);
            fx = fb;
            btheta = theta + (a+(c-a)*w)*p;
            for k=1:m
                [~, fb(k)] = runNet([xi(:, k); btheta; upsilon(k)]');
            end
            fb = mean(fb);
        else
            a = a+(c-a)*w;
            fb = fx;
            xtheta = theta + (a+(c-a)*(1-w))*p;
            for k=1:m
              [~, fx(k)] = runNet([xi(:, k); xtheta; upsilon(k)]');
            end
            fx = mean(fx);
        end
        runc = runc+1;
    end
    xmin = (a+c)/2;
    rec = rec(1:k-1, :);
end

% test function:
%   f0 = @(x) x^2;
% narrowing of the interval:
%   [x, rec] = fminGS(f0, -1, 3, 1.0e-40);
%   t = 1:size(rec, 1);
%   plot(1:t, rec(:, 1), t, rec(:, 2))
% to see more detail:
%   semilogy(t, rec(:, 2)-rec(:, 1))
% to understand the slope:
%   hold on
%   semilogy(t, 0.61803.^t) or semilogy(t, exp(t*log(0.61803)))
% => convergence rate of -log(0.61803)
% function values:
%   semilogy(t, rec(:, 3))
%  => twice the slope as the objective is quadratic
%   f1 = @(x) abs(x);
%   [x, rec] = fminGS(f1, -1, 3, 1.0e-40);
%  => algorithm performance is invariant to strictly monotonically
%     increasing transformations of the objective
% Or is it?
%   f2 = @(x) 1+abs(x);
%   [x, rec] = fminGS(f2, -1, 3, 1.0e-40);
%   t = 1:size(rec, 1);
%   semilogy(t, rec(:, 3)-1)
%
%   f3 = @(x) (x-1.2)^2;
%   [x, rec] = fminGS(f3, -1, 3, 1.0e-40);
%   semilogy(t, rec(:, 3))
% => numerical accuracy issue; 1.2+1.0e-16 is the same as 1.2; double
% floating point accuracy gives you approximately 15 digits
